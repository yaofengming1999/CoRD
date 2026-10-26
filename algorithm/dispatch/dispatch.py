from __future__ import annotations

from pathlib import Path

from algorithm.dispatch.return_policies import (
    HeuristicDroneReturnPolicy,
    NearestDroneReturnPolicy,
    PreplannedDroneReturnPolicy,
    RandomDroneReturnPolicy,
)
from algorithm.R_rule_based import (
    FullChargeFifoOrderBatteryPolicy,
    HeuristicOrderBatteryPolicy,
    HungarianOnTimeGmvOrderBatteryPolicy,
    LowestFeasibleOrderBatteryPolicy,
    RandomOrderBatteryPolicy,
)
from simulator.model.enums import VehicleStatus
from simulator.model.types import StateDict, TaskLevelActionDict


ORDER_BATTERY_POLICY_NAMES = (
    "heuristic",
    "random",
    "lowest_feasible",
    "full_charge_fifo",
    "hungarian_ontime_gmv",
    "rbg_svm_auction",  # L1 (planned): two-stage like R*, return stays a rule
    "drone_trajectory_joint_plan_ppo",
    "global_ppo",
    "drone_ordered_counterfactual_ppo",
    "moe_pointer",
    "drone_synchronous_mappo",
    "staged_order_launchpad_battery_ppo",
    "independent_order_launchpad_battery_ppo",  # F1b: o, l, b drawn independently
    "d2_separable_ppo",
    "d2_separable_team_reward_ppo",  # AS5: F2 with the team reward on per-drone trajectories
    "drone_synchronous_mappo_own_reward",  # AS4: AS1 with per-drone credit
    "d2_separable_battery_blind_ppo",      # AS6: no battery state, random battery
    "ob_l_one_decision_ppo",               # F0OneDecision: (order, battery) | return pad
    "f2pro_ppo",                           # F2pro: o | l | b, interaction on (o,l) only
    "f0_dispatch_only_ppo",  # F0: learned (order, battery); the return is a second agent
    # planned (L1/L2/L3/L5) -- see algorithm/L*_*.py stubs
    "msra",
    "mapdp",
    "hrl4ag",
)
_PREPLANNED_POLICY_NAMES = frozenset(ORDER_BATTERY_POLICY_NAMES[6:]) - {"f0_dispatch_only_ppo"}
_TRAINABLE_POLICY_NAMES = _PREPLANNED_POLICY_NAMES | {"rbg_svm_auction", "f0_dispatch_only_ppo"}
# return policies that carry their own weights (F0's second agent)
_TRAINABLE_RETURN_POLICY_NAMES = frozenset({"f0_learned_ppo"})


class DroneDecisionDispatcher:
    """Task-level dispatcher for the drone-only model.

    Coordinates the three actions exposed by the operational model:
      1. order-battery dispatch (match a waiting order to a launchpad battery;
         the simulator assigns a homogeneous idle drone there), and
      2. drone return (pick a feasible launchpad for a delivered drone), and
      3. optional idle drone rebalancing with an explicitly selected battery.
    Return and rebalance share one policy because both choose a drone's target
    launchpad after checking empty-flight battery feasibility.
    Each decision is a swappable policy; the dispatcher only orchestrates them
    and must not mutate state directly.
    """

    def __init__(self, config: dict, seed: int = 0):
        self.config = config
        self.cfg = config.get("drone_decision", {})
        self.order_battery_policy_name = str(self.cfg.get("order_battery_policy", "heuristic"))
        self.drone_return_policy_name = str(self.cfg.get("drone_return_policy", "heuristic"))
        if (self.order_battery_policy_name in _PREPLANNED_POLICY_NAMES) != (
            self.drone_return_policy_name == "preplanned"
        ):
            raise ValueError(
                "joint-plan PPO policies and preplanned must be selected together "
                "because the dispatch action owns the return decision"
            )
        if (
            self.drone_return_policy_name == "preplanned"
            and bool(self.cfg.get("rebalance", {}).get("enabled", False))
        ):
            raise ValueError("preplanned return does not support idle rebalance")
        self.order_battery_policy = self._build_order_battery_policy(config, seed)
        self.drone_return_policy = self._build_drone_return_policy(config, seed)

    def plan(self, state: StateDict) -> list[TaskLevelActionDict]:
        used_orders: set[str] = set()
        used_batteries: set[str] = set()
        actions = self.order_battery_policy.plan(state, used_orders, used_batteries)

        dispatched_drones = self._drones_consumed_by_dispatch(state, actions)
        return_actions = self.drone_return_policy.plan(state, dispatched_drones, used_batteries)
        actions.extend(return_actions)
        return actions

    def _drones_consumed_by_dispatch(
        self,
        state: StateDict,
        actions: list[TaskLevelActionDict],
    ) -> set[str]:
        orders = state.get("orders", {})
        launchpads = state.get("launchpads", {})
        drones = state.get("drones", {})
        available_by_launchpad: dict[str, list[str]] = {}
        for launchpad_id, launchpad in launchpads.items():
            available_by_launchpad[str(launchpad_id)] = [
                str(drone_id)
                for drone_id in sorted(launchpad.get("drone_ids", []))
                if drone_id in drones
                and drones[drone_id].get("status") == VehicleStatus.IDLE.value
                and not drones[drone_id].get("route_plan")
                and drones[drone_id].get("current_battery_id") is None
            ]
        consumed: set[str] = set()
        for action in actions:
            order = orders.get(str(action.get("order_id", "")))
            if order is None:
                continue
            launchpad_id = str(order.get("origin_launchpad_id"))
            candidates = available_by_launchpad.get(launchpad_id, [])
            requested_drone_id = action.get("drone_id")
            if requested_drone_id is not None and str(requested_drone_id) in candidates:
                candidates.remove(str(requested_drone_id))
                consumed.add(str(requested_drone_id))
            elif candidates:
                consumed.add(candidates.pop(0))
        return consumed

    def finish_episode(
        self,
        on_time_gmv: float,
        *,
        order_rewards: dict[str, float] | None = None,
        gradient_steps: int = 1,
        **kwargs,
    ) -> dict[str, float]:
        stats = self.order_battery_policy.finish_episode(
            on_time_gmv,
            order_rewards=order_rewards,
            gradient_steps=gradient_steps,
            **kwargs,
        )
        stats.update(self.drone_return_policy.finish_episode(
            on_time_gmv,
            order_rewards=order_rewards,
            gradient_steps=gradient_steps,
            **kwargs,
        ))
        return stats

    def clear_episode(self) -> None:
        self.order_battery_policy.clear_episode()
        self.drone_return_policy.clear_episode()

    def save(self, path: str) -> None:
        saved = False
        if self.order_battery_policy_name in _TRAINABLE_POLICY_NAMES:
            self.order_battery_policy.save(str(self._order_battery_model_path(path)))
            saved = True
        if self.drone_return_policy_name in _TRAINABLE_RETURN_POLICY_NAMES:
            self.drone_return_policy.save(str(self._drone_return_model_path(path)))
            saved = True
        if not saved:
            raise NotImplementedError("No trainable policy to save")

    def load(self, path: str) -> None:
        order_battery_path = self._order_battery_model_path(path)
        if (
            self.order_battery_policy_name in _TRAINABLE_POLICY_NAMES
            and order_battery_path.is_file()
        ):
            self.order_battery_policy.load(str(order_battery_path))
        drone_return_path = self._drone_return_model_path(path)
        if (
            self.drone_return_policy_name in _TRAINABLE_RETURN_POLICY_NAMES
            and drone_return_path.is_file()
        ):
            self.drone_return_policy.load(str(drone_return_path))

    def _build_order_battery_policy(self, config: dict, seed: int):
        if self.order_battery_policy_name == "heuristic":
            return HeuristicOrderBatteryPolicy(config)
        if self.order_battery_policy_name == "random":
            return RandomOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "lowest_feasible":
            return LowestFeasibleOrderBatteryPolicy(config)
        if self.order_battery_policy_name == "full_charge_fifo":
            return FullChargeFifoOrderBatteryPolicy(config)
        if self.order_battery_policy_name == "hungarian_ontime_gmv":
            return HungarianOnTimeGmvOrderBatteryPolicy(config)
        if self.order_battery_policy_name == "drone_trajectory_joint_plan_ppo":
            from algorithm.AS2_single_drone_ppo import AS2SingleDronePPOOrderBatteryPolicy

            return AS2SingleDronePPOOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "global_ppo":
            from algorithm.AS3_global_drones_ppo import AS3GlobalDronesPPOOrderBatteryPolicy

            return AS3GlobalDronesPPOOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "drone_ordered_counterfactual_ppo":
            from algorithm.F3_full_combination import F3FullCombinationOrderBatteryPolicy

            return F3FullCombinationOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "rbg_svm_auction":
            from algorithm.L1_rbg import RBGSVMAuctionOrderBatteryPolicy
            return RBGSVMAuctionOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "msra":
            from algorithm.L2_msra import MSRAOrderBatteryPolicy
            return MSRAOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "mapdp":
            from algorithm.L3_mapdp import MAPDPOrderBatteryPolicy
            return MAPDPOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "hrl4ag":
            from algorithm.L5_hrl4ag import HRL4AGOrderBatteryPolicy
            return HRL4AGOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "moe_pointer":
            from algorithm.L4_moe_pointer import MoEPointerOrderBatteryPolicy
            return MoEPointerOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "drone_synchronous_mappo":
            from algorithm.AS1_mappo import (
                AS1MAPPOOrderBatteryPolicy,
            )

            return AS1MAPPOOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "staged_order_launchpad_battery_ppo":
            from algorithm.F1_staged_olb import F1StagedOLBOrderBatteryPolicy

            return F1StagedOLBOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "independent_order_launchpad_battery_ppo":
            from algorithm.F1b_independent_olb import F1bIndependentOLBOrderBatteryPolicy

            return F1bIndependentOLBOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "d2_separable_ppo":
            from algorithm.F2_separable import F2SeparableOrderBatteryPolicy

            return F2SeparableOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "f2pro_ppo":
            from algorithm.F2pro.policy import F2ProPolicy

            return F2ProPolicy(config, seed=seed)
        if self.order_battery_policy_name == "ob_l_one_decision_ppo":
            from algorithm.F0_one_decision import F0OneDecisionOrderBatteryPolicy

            return F0OneDecisionOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "d2_separable_battery_blind_ppo":
            from algorithm.AS6_battery_blind import AS6BatteryBlindOrderBatteryPolicy

            return AS6BatteryBlindOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "drone_synchronous_mappo_own_reward":
            from algorithm.AS4_sync_own_reward import AS4SyncOwnRewardOrderBatteryPolicy

            return AS4SyncOwnRewardOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "d2_separable_team_reward_ppo":
            from algorithm.AS5_team_reward import AS5TeamRewardOrderBatteryPolicy

            return AS5TeamRewardOrderBatteryPolicy(config, seed=seed)
        if self.order_battery_policy_name == "f0_dispatch_only_ppo":
            from algorithm.F0_two_stage import F0DispatchOrderBatteryPolicy

            return F0DispatchOrderBatteryPolicy(config, seed=seed)
        raise ValueError(f"Unknown order_battery_policy: {self.order_battery_policy_name}")

    def _build_drone_return_policy(self, config: dict, seed: int):
        if self.drone_return_policy_name == "heuristic":
            return HeuristicDroneReturnPolicy(config)
        if self.drone_return_policy_name == "nearest":
            return NearestDroneReturnPolicy(config)
        if self.drone_return_policy_name == "random":
            return RandomDroneReturnPolicy(config, seed=seed)
        if self.drone_return_policy_name == "preplanned":
            return PreplannedDroneReturnPolicy(config)
        if self.drone_return_policy_name == "f0_learned_ppo":
            from algorithm.F0_two_stage import F0LearnedDroneReturnPolicy

            return F0LearnedDroneReturnPolicy(config, seed=seed)
        raise ValueError(f"Unknown drone_return_policy: {self.drone_return_policy_name}")

    def _order_battery_model_path(self, path: str) -> Path:
        return Path(path).with_suffix(".order_battery.pt")

    def _drone_return_model_path(self, path: str) -> Path:
        return Path(path).with_suffix(".drone_return.pt")
