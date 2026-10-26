"""F0 Two-Stage -- the same two learned decisions, but never taken together.

F2 treats one dispatch as a single choice over (order, return pad, battery): the
battery it picks has to carry the drone out AND back to the pad the same action
named. F0 splits that choice between two agents that never see each other's
menu:

  * the DISPATCH agent picks (order, battery) only. Its rows are F2's rows with
    the return axis collapsed to the energy-minimal pad (``collapse_return``), so
    it sees the same features and the same feasibility test -- "some return
    exists" -- but l is not part of its action.
  * the RETURN agent picks the pad, later, when the drone is idle at a customer
    endpoint and the battery is already spent down to whatever the delivery left.

Both are PPO with the serialized per-drone credit assignment F2 uses: one
trajectory per physical drone, carrying only the rewards of the orders that drone
flew. The return agent's decisions sit in the same drone's trajectory, so a pad
that sets the drone up for its next order is what earns the credit.

F0 is therefore the ablation for the joint action itself, not for the scorer
(F1/F3) or for the agent structure (AS1-AS3): it keeps both, and only stops the
two decisions from being made as one. The metric that reads the difference is
``E3_battery/avg_delivery_margin_ratio`` against
``E3_battery/avg_roundtrip_margin_ratio`` -- see MetricsRecorder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithm.common.geometry_utils import distance_m
from algorithm.common.mappo_utils import temporal_team_rewards
from algorithm.common.torch_learner_base import BaseTorchMAPPOLearner
from algorithm.dispatch.return_policies import HeuristicDroneReturnPolicy
from algorithm.F2.policy import F2Policy
from simulator.model.enums import VehicleStatus
from simulator.model.types import StateDict, TaskLevelActionDict


# ------------------------------------------------------------------ #
# stage 1: dispatch without the return
# ------------------------------------------------------------------ #
class F0DispatchOrderBatteryPolicy(F2Policy):
    """F2's menu and learner with the return axis removed from the action."""

    plans_return = False
    collapse_return = True
    dispatch_action_prefix = "f0_two_stage_dispatch"
    joint_plan_config_key = "order_battery_f0_two_stage_ppo"
    joint_plan_feature_schema = "dispatch_only_order_battery_f0_v2"   # v2 hides the return pad
    dispatch_candidate_error = "F0 dispatch candidate is missing order or battery"


# ------------------------------------------------------------------ #
# stage 2: the return, learned
# ------------------------------------------------------------------ #
RETURN_CANDIDATE_DIM = 11
RETURN_CONTEXT_DIM = 7


@dataclass(frozen=True)
class _ReturnSample:
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    actor_context: np.ndarray
    action: int
    log_prob: float
    value: float


@dataclass(frozen=True)
class _ReturnDecision:
    time: int
    drone_id: str
    sample: _ReturnSample


class _PooledValueCritic(nn.Module):
    """V(s) from the masked-mean of the pad rows and the context; no pi dependence."""

    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.candidate_net = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.context_net = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, candidate_features, mask, context):
        rows = self.candidate_net(candidate_features)
        weight = mask.unsqueeze(-1).float()
        pooled = (rows * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return self.head(torch.cat([pooled, self.context_net(context)], dim=-1)).squeeze(-1)


class F0ReturnLearner(BaseTorchMAPPOLearner):
    """PPO over the feasible return pads, on the base class's candidate actor."""

    candidate_empty_error = "F0 return PPO requires at least one feasible launchpad"
    actor_context_label = "F0 return context"

    def __init__(self, *, context_dim: int = RETURN_CONTEXT_DIM, **kwargs):
        super().__init__(actor_context_dim=int(context_dim), **kwargs)
        self.critic = _PooledValueCritic(
            self.candidate_dim, self.actor_context_dim, self.hidden_dim,
        ).to(self.device)
        self._initialize_optimizers()

    def act(self, candidate_features: np.ndarray, context: np.ndarray, *, training: bool):
        if len(candidate_features) == 0:
            raise ValueError(self.candidate_empty_error)
        if len(candidate_features) > self.max_candidates:
            raise ValueError(self._candidate_overflow_error(len(candidate_features)))
        step = self._actor_step(candidate_features, context, training=training)
        with torch.no_grad():
            value = self.critic(
                step.candidate_features_t, step.candidate_mask_t, step.actor_context_t,
            )
        return step.action, _ReturnSample(
            step.candidate_features, step.candidate_mask, step.actor_context,
            step.action, step.log_prob, float(value.item()),
        )

    # ---- rollout plumbing (the base class runs the PPO loop) ----
    def _sample_rollout_fields(self, sample: _ReturnSample) -> dict[str, object]:
        return {
            "candidate_features": sample.candidate_features,
            "candidate_mask": sample.candidate_mask,
            "actor_context": sample.actor_context,
        }

    def _rollout_array_fields(self) -> dict[str, tuple[str, torch.dtype]]:
        return {
            "candidate_features": ("candidate_features", torch.float32),
            "mask": ("candidate_mask", torch.bool),
            "context": ("actor_context", torch.float32),
        }

    def _actor_forward(self, batch):
        return self.actor(batch["candidate_features"], batch["mask"], batch["context"])

    def _critic_forward(self, batch):
        return self.critic(batch["candidate_features"], batch["mask"], batch["context"])

    def _checkpoint_metadata(self) -> dict[str, object]:
        return {
            "feature_schema": self.feature_schema,
            "candidate_dim": self.candidate_dim,
            "context_dim": self.actor_context_dim,
            "hidden_dim": self.hidden_dim,
        }

    def _validate_checkpoint(self, payload: dict[str, object]) -> None:
        if str(payload.get("feature_schema")) != self.feature_schema:
            raise ValueError(
                f"return checkpoint schema {payload.get('feature_schema')!r} does not "
                f"match {self.feature_schema!r}"
            )
        for key, expected in (
            ("candidate_dim", self.candidate_dim),
            ("context_dim", self.actor_context_dim),
            ("hidden_dim", self.hidden_dim),
        ):
            if int(payload.get(key, expected)) != int(expected):
                raise ValueError(f"return checkpoint {key} mismatch")


class F0LearnedDroneReturnPolicy(HeuristicDroneReturnPolicy):
    """Stage 2 of F0: the return pad is sampled, not scored by the backlog rule.

    Only ``_best_return_target`` changes; feasibility, the idle-drone scan and
    the (disabled under F0) rebalance path stay the heuristic's.
    """

    feature_schema = "f0_return_pad_v2"   # v2 added the within-step incoming count

    def __init__(self, config: dict, *, seed: int = 0):
        super().__init__(config)
        decision_cfg = config.get("drone_decision", {})
        cfg = dict(decision_cfg.get("drone_return_f0_two_stage_ppo", {}))
        self.config = config
        self.training = bool(config.get("rl", {}).get("training", False))
        self.gradient_steps_per_episode = int(cfg.get("ppo_epochs", 8))
        self.learner = F0ReturnLearner(
            candidate_dim=RETURN_CANDIDATE_DIM,
            context_dim=RETURN_CONTEXT_DIM,
            max_candidates=int(cfg.get("max_return_launchpads", 512)),
            feature_schema=self.feature_schema,
            hidden_dim=int(cfg.get("hidden_dim", 128)),
            actor_lr=float(cfg.get("actor_lr", 3e-4)),
            critic_lr=float(cfg.get("critic_lr", 1e-3)),
            gamma=float(cfg.get("gamma", 0.99)),
            gae_lambda=float(cfg.get("gae_lambda", 0.95)),
            clip_ratio=float(cfg.get("clip_ratio", 0.2)),
            value_coefficient=float(cfg.get("value_coefficient", 0.5)),
            entropy_coefficient=float(cfg.get("entropy_coefficient", 0.01)),
            max_grad_norm=float(cfg.get("max_grad_norm", 10.0)),
            batch_size=int(cfg.get("batch_size", 32)),
            ppo_epochs=int(cfg.get("ppo_epochs", 8)),
            time_unit_s=float(cfg.get("time_unit_s", 60.0)),
            reward_scale=float(cfg.get("reward_scale", 25.0)),
            device=str(decision_cfg.get("device", cfg.get("device", "cuda"))),
            seed=seed,
        )
        self._episode_decisions: list[_ReturnDecision] = []
        self._orders_by_drone: dict[str, set[str]] = {}
        self._horizon = max(1.0, float(config.get("simulation", {}).get("horizon_seconds", 1)))
        self._deciding_drone_id: str | None = None
        self._state: StateDict | None = None
        self._planned_targets: dict[str, int] = {}

    # ---- the plan loop only needs the state and the acting drone in hand ----
    def plan(self, state, used_drones, used_batteries=None):
        self._state = state
        self._planned_targets = {}   # pads this step has already committed to
        self._observe_onboard_orders(state)
        return super().plan(state, used_drones, used_batteries)

    def _observe_onboard_orders(self, state: StateDict) -> None:
        """Remember which orders each drone flew, for per-drone reward attribution.

        The return decision is credited with what the drone delivers next, so the
        trajectory needs the drone -> order map. An order is only onboard for the
        steps between dispatch and delivery, so it is collected as it passes.
        """
        for drone in state.get("drones", {}).values():
            onboard = drone.get("onboard_order_ids") or []
            if onboard:
                self._orders_by_drone.setdefault(
                    str(drone.get("vehicle_id")), set()
                ).update(str(order_id) for order_id in onboard)

    def _best_return_target(self, drone, launchpads, state=None):
        state = state if state is not None else self._state
        feasible = self._feasible_return_targets(drone, launchpads)
        if not feasible:
            return None
        if len(feasible) == 1:
            only = str(feasible[0].get("launchpad_id"))
            self._planned_targets[only] = self._planned_targets.get(only, 0) + 1
            return feasible[0]
        rows = self._pad_features(drone, feasible, self._planned_targets)
        context = self._context(drone, state, len(feasible))
        index, sample = self.learner.act(rows, context, training=self.training)
        if self.training and state is not None:
            self._episode_decisions.append(_ReturnDecision(
                int(state.get("time", 0)), str(drone.get("vehicle_id")), sample,
            ))
        chosen = feasible[index]
        target_id = str(chosen.get("launchpad_id"))
        self._planned_targets[target_id] = self._planned_targets.get(target_id, 0) + 1
        return chosen

    # ---- features ----
    def _pad_features(
        self, drone: dict, launchpads: list[dict], planned: dict[str, int],
    ) -> np.ndarray:
        """One row per feasible pad, including what this step has already committed.

        ``planned`` counts the drones earlier in this step's loop that are already
        heading to each pad. Without it every returning drone of a step sees the
        identical pad state and they all pile onto whichever pad looks best -- F2
        does not have that problem, because its dispatch loop accumulates
        ``planned_targets`` and the target features expose it, so a later drone sees
        the earlier one's commitment. Leaving it out here would make F0 look worse
        than two-stage dispatch has to be.
        """
        here = drone.get("current_location")
        distances = np.asarray(
            [distance_m(here, pad.get("current_location")) for pad in launchpads],
            dtype=np.float64,
        )
        nearest = float(distances.min()) if len(distances) else 0.0
        rows = np.zeros((len(launchpads), RETURN_CANDIDATE_DIM), dtype=np.float32)
        for i, pad in enumerate(launchpads):
            backlog = len(pad.get("order_queue", []) or [])
            docked = len(pad.get("drone_ids", []) or [])
            charged = len(pad.get("charged_battery_ids", []) or [])
            charging = len(pad.get("charging_battery_ids", []) or [])
            incoming = int(planned.get(str(pad.get("launchpad_id")), 0))
            # the demand features count the committed arrivals, so a pad that two
            # earlier drones already chose no longer reads as unserved
            taken = docked + incoming
            rows[i] = (
                distances[i] / 5000.0,                                   # P1 flight distance
                distances[i] / max(1.0, nearest),                        # P2 detour vs nearest
                float(distances[i] <= nearest + 1.0),                    # P3 is the nearest pad
                backlog / 10.0,                                          # P4 waiting orders
                docked / 10.0,                                           # P5 drones already there
                charged / 10.0,                                          # P6 batteries ready now
                charging / 10.0,                                         # P7 batteries charging
                (backlog - taken) / 10.0,                                # P8 unmet demand
                float(backlog > 0 and charged > 0),                      # P9 can serve on arrival
                float(min(backlog, charged, max(0, 1 - taken))),         # P10 served only if I go
                incoming / 10.0,                                         # P11 committed this step
            )
        return rows

    def _context(self, drone: dict, state: StateDict | None, feasible_count: int) -> np.ndarray:
        if state is None:
            return np.zeros(RETURN_CONTEXT_DIM, dtype=np.float32)
        orders = state.get("orders", {})
        drones = state.get("drones", {})
        launchpads = state.get("launchpads", {})
        waiting = sum(
            1 for order in orders.values()
            if order.get("status") in {"CREATED", "WAITING", "PENDING"}
        )
        idle = sum(
            1 for other in drones.values()
            if other.get("status") == VehicleStatus.IDLE.value
            and other.get("current_launchpad_id") is not None
        )
        return np.asarray([
            float(state.get("time", 0)) / self._horizon,                 # C1 time of day
            waiting / 20.0,                                              # C2 open demand
            idle / max(1.0, len(drones)),                                # C3 idle share
            len(drones) / 20.0,                                          # C4 fleet size
            feasible_count / max(1.0, len(launchpads)),                  # C5 reachable share
            len(launchpads) / 20.0,                                      # C6 network size
            float(drone.get("remaining_capacity", 1.0) or 0.0),          # C7 free capacity
        ], dtype=np.float32)

    # ---- learning ----
    def finish_episode(
        self,
        on_time_gmv: float = 0.0,
        *,
        reward_events: list[dict] | None = None,
        episode_end_time: int | None = None,
        gradient_steps: int | None = None,
        **kwargs,
    ) -> dict[str, float]:
        decisions = len(self._episode_decisions)
        if not self.training or not self._episode_decisions:
            result = {
                "drone_rebalances": float(self._rebalances_planned),
                "drone_return_decisions": float(decisions),
                "drone_return_actor_loss": 0.0,
                "drone_return_critic_loss": 0.0,
                "drone_return_entropy": 0.0,
            }
            self.clear_episode()
            return result
        terminal = int(
            episode_end_time
            if episode_end_time is not None
            else self._episode_decisions[-1].time
        )
        by_drone: dict[str, list[_ReturnDecision]] = {}
        for decision in self._episode_decisions:
            by_drone.setdefault(decision.drone_id, []).append(decision)
        for drone_id, trajectory in by_drone.items():
            # same rule as F2: a drone's trajectory only carries the rewards of the
            # orders that drone flew, so a return is judged by what it sets up next
            own = self._orders_by_drone.get(drone_id, set())
            events = [
                event for event in (reward_events or [])
                if event.get("order_id") is not None and str(event["order_id"]) in own
            ]
            times = [decision.time for decision in trajectory]
            self.learner.store_episode(
                [decision.sample for decision in trajectory],
                temporal_team_rewards(times, events),
                times,
                terminal,
            )
        epochs = self.gradient_steps_per_episode if gradient_steps is None else int(gradient_steps)
        stats = self.learner.update(epochs=epochs)
        result = {
            "drone_rebalances": float(self._rebalances_planned),
            "drone_return_decisions": float(decisions),
            "drone_return_actor_loss": float(stats["actor_loss"]),
            "drone_return_critic_loss": float(stats["critic_loss"]),
            "drone_return_entropy": float(stats["entropy"]),
            "drone_return_rollout_size": float(stats["rollout_size"]),
        }
        self.clear_episode()
        return result

    def clear_episode(self) -> None:
        super().clear_episode()
        self._episode_decisions = []
        self._orders_by_drone = {}
        self._state = None
        self._planned_targets = {}
        self.learner.clear_rollout()

    def save(self, path: str) -> None:
        self.learner.save(path)

    def load(self, path: str) -> None:
        self.learner.load(path)
