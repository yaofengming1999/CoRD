"""F1b: IndependentOLB+SerializedMARL -- three independent heads, no conditioning.

F1 (``F1_staged_olb.py``) decodes autoregressively:

    pi(o,l,b | s) = pi_o(o | s) * pi_l(l | s,o) * pi_b(b | s,o,l)

and its later stages also *enumerate* against the earlier choice -- stage 2
lists the return targets of the chosen order, stage 3 the batteries that
confirm that specific (order, target) pair. F1b removes the conditioning on
both sides and keeps everything else:

    pi(o,l,b | s) = pi_o(o | s) * pi_l(l | s) * pi_b(b | s)

  * all three heads read the SAME state context. In F1 the return head's
    context is ``[context, selected order]`` and the battery head's is
    ``[context, selected order, selected target]``; here all three are
    ``[context]``, so no head can see what another head picked.
  * the three candidate menus are built against fixed representatives that
    no head influences: the highest-charge local battery ``b*`` (which is
    already F1's representative in stages 1 and 2) and the pruned queue's
    first feasible order ``o*``. The return menu is ``o*``'s return plans;
    the battery menu is the local batteries that can serve ``o*`` at all,
    with no target fixed.
  * the triple is assembled only after all three heads have drawn, and its
    joint feasibility is checked then. An infeasible triple does not
    dispatch: the launchpad is blocked for this epoch, exactly as on F1's
    other no-dispatch paths, and the sample still carries all three heads'
    log-probs, so the policy is credited for the choice it actually made.

Everything else is inherited from F1 unchanged: the three stages' feature
functions, the shared ``CANDIDATE_DIM`` padding, the factorised-joint-log-prob
PPO/GAE update, the SerializedMARL trajectory cut (one GAE trajectory per
physical drone carrying only its own orders' rewards), and the hyperparameters.
The F1-against-F1b difference therefore prices the sequential conditioning
alone, with the expansion cost held at three small stages in both.

Why the axis is worth a row: F3's full product is not measurably better than
the proposed factorisation on these instances, because the energy constraint
leaves barely more than one feasible battery per (order, pad) pair -- the
product it expands is nearly degenerate. That makes "how much coupling does
the decode actually need?" the live question, and F1b is its lower end: the
cheapest possible expansion, with no coupling at all.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algorithm.common.torch_learner_base import _CandidateActor
from algorithm.F1_staged_olb import (
    CANDIDATE_DIM,
    F1StagedOLBOrderBatteryPolicy,
    StagedPending,
    StagedSample,
    TorchStagedOrderLaunchpadBatteryPPOLearner,
)
from algorithm.F2 import featmask
from algorithm.F2.policy import _JointPlanCandidate, _JointPlanDecision
from simulator.model.types import StateDict, TaskLevelActionDict


class _IndependentActor(nn.Module):
    """Three heads at the same context width: no head sees another's choice.

    Constructed in the same order as F1's ``_StagedActor`` (order, return,
    battery); only the return and battery heads' context width changes, from
    ``context_dim + k * candidate_dim`` down to ``context_dim``.
    """

    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.order_actor = _CandidateActor(candidate_dim, context_dim, hidden_dim)
        self.return_actor = _CandidateActor(candidate_dim, context_dim, hidden_dim)
        self.battery_actor = _CandidateActor(candidate_dim, context_dim, hidden_dim)


class TorchIndependentOrderLaunchpadBatteryPPOLearner(
    TorchStagedOrderLaunchpadBatteryPPOLearner
):
    """F1's learner with the two conditioning concatenations removed."""

    candidate_empty_error = "independent order/launchpad/battery PPO requires a non-empty stage"
    actor_context_label = "independent order/launchpad/battery PPO context"
    actor_class = _IndependentActor

    def complete_return(self, pending, return_features, *, training):
        returns = self._stage(return_features, "return launchpad")
        # F1 concatenates pending.selected_order_features here; F1b does not.
        context = pending.order_context
        _, dist, action = self._sample(
            self.actor.return_actor, returns, context, training
        )
        index = int(action.item())
        return index, StagedPending(
            pending.order_features,
            pending.order_context,
            pending.order_action,
            pending.order_log_prob,
            pending.value,
            pending.selected_order_features,
            returns,
            index,
            float(dist.log_prob(action).item()),
            returns[index].copy(),
        )

    def complete_battery(self, pending, battery_features, *, training):
        batteries = self._stage(battery_features, "battery")
        # F1 concatenates the selected order AND target here; F1b does not.
        context = pending.order_context
        _, dist, action = self._sample(
            self.actor.battery_actor, batteries, context, training
        )
        index = int(action.item())
        return index, StagedSample(
            pending.order_features,
            pending.return_features,
            batteries,
            pending.order_context,
            pending.order_context,
            pending.order_context,
            pending.order_action,
            pending.return_action,
            index,
            True,
            True,
            pending.order_log_prob + pending.return_log_prob + float(dist.log_prob(action).item()),
            pending.value,
        )

    def complete_without_dispatch(self, pending):
        empty = np.empty((0, self.candidate_dim), dtype=np.float32)
        zeros = np.zeros((self.context_dim,), dtype=np.float32)
        return StagedSample(
            pending.order_features,
            empty,
            empty,
            pending.order_context,
            zeros,
            zeros,
            pending.order_action,
            0,
            0,
            False,
            False,
            pending.order_log_prob,
            pending.value,
        )

    def _checkpoint_metadata(self):
        return {
            "feature_schema": self.feature_schema,
            "candidate_dim": self.candidate_dim,
            "context_dim": self.context_dim,
            "transition_mode": "independent_order_launchpad_battery",
        }

    def _validate_checkpoint(self, payload):
        if payload.get("feature_schema") != self.feature_schema:
            raise ValueError("independent order/launchpad/battery feature schema is incompatible")


class F1bIndependentOLBOrderBatteryPolicy(F1StagedOLBOrderBatteryPolicy):
    """F1b: order, return pad and battery drawn independently, then assembled."""

    dispatch_action_prefix = "f1b_independent_olb_dispatch"
    joint_plan_config_key = "order_battery_f1b_independent_olb_ppo"
    joint_plan_feature_schema = "independent_order_launchpad_battery_v1"
    learner_class = TorchIndependentOrderLaunchpadBatteryPPOLearner
    candidate_dim = CANDIDATE_DIM

    def __init__(self, config: dict, *, seed: int = 0):
        super().__init__(config, seed=seed)
        self._joint_attempts = 0
        self._joint_infeasible = 0

    def _learner_kwargs(self, cfg: dict, *, seed: int) -> dict:
        kwargs = super()._learner_kwargs(cfg, seed=seed)
        kwargs["context_dim"] = featmask.active().context_dim
        return kwargs

    def clear_episode(self) -> None:
        self._joint_attempts = 0
        self._joint_infeasible = 0
        super().clear_episode()

    def finish_episode(self, *args, **kwargs) -> dict[str, float]:
        attempts, infeasible = self._joint_attempts, self._joint_infeasible
        result = super().finish_episode(*args, **kwargs)
        result["order_battery_joint_attempts"] = float(attempts)
        result["order_battery_joint_infeasible"] = float(infeasible)
        result["order_battery_joint_infeasible_rate"] = (
            float(infeasible) / float(attempts) if attempts else 0.0
        )
        return result

    # ------------------------------------------------------------------
    # Representatives: chosen without reference to any head's draw.
    # ------------------------------------------------------------------

    def _representative_order_id(
        self,
        state: StateDict,
        launchpad: dict,
        drone_id: str,
        used_orders: set[str],
        used_batteries: set[str],
        current_time: int,
        planned_departures: dict[str, int],
    ) -> str | None:
        """The pruned queue's first order that is feasible against ``b*``.

        This is the same pruned list and the same existence check F1's stage 1
        builds its menu from, so the representative is always one of the
        dispatchable options the order head sees -- it just is not the one the
        head picked.
        """
        from simulator.model.enums import OrderStatus

        orders = state.get("orders", {})
        batteries = state.get("batteries", {})
        launchpad_id = str(launchpad.get("launchpad_id"))
        active_order_id = self.menu.active_order_by_launchpad.get(launchpad_id)
        order_ids = [
            str(oid) for oid in launchpad.get("order_queue", [])
            if str(oid) not in used_orders
            and str(oid) in orders
            and orders[str(oid)].get("status") == OrderStatus.PENDING.value
            and (active_order_id is None or str(oid) == active_order_id)
        ]
        order_ids = self.menu.prune_order_ids(
            state, launchpad, order_ids, current_time, planned_departures
        )
        local_battery_ids = self._local_battery_ids(launchpad, batteries, used_batteries)
        best_battery_id = self._best_local_battery_id(local_battery_ids, batteries)
        if best_battery_id is None or self.candidate_feasible_fn is None:
            return None
        best_battery = batteries[best_battery_id]
        for order_id in order_ids:
            order = orders[order_id]
            if (
                self.menu.drone_can_carry(state, drone_id, order)
                and bool(self.candidate_feasible_fn(order, best_battery))
            ):
                return order_id
        return None

    def _marginal_battery_candidates(
        self,
        state: StateDict,
        launchpad: dict,
        representative_order_id: str,
        used_batteries: set[str],
        current_time: int,
    ) -> list[_JointPlanCandidate]:
        """Local batteries that can serve ``o*`` at all -- no target fixed.

        F1's stage 3 keeps only the batteries whose return plan to the CHOSEN
        pad is feasible. That is the conditioning this row removes, so the mask
        here stops at order-level feasibility and the features use ``o*``'s
        energy-minimal return plan as the route block.
        """
        order = state["orders"][representative_order_id]
        batteries = state.get("batteries", {})
        local_battery_ids = self._local_battery_ids(launchpad, batteries, used_batteries)
        if self.candidate_details_fn is None:
            return []
        candidates: list[_JointPlanCandidate] = []
        for battery_id in local_battery_ids:
            battery = batteries[battery_id]
            details = self.candidate_details_fn(order, battery)
            if not bool(details.get("feasible", False)):
                continue
            plans = [
                route for route in details.get("return_plans", {}).values()
                if bool(route.get("feasible", False))
            ]
            if not plans:
                continue
            route = max(plans, key=lambda r: float(r.get("reserve_after_ratio", 0.0) or 0.0))
            candidates.append(_JointPlanCandidate(
                "dispatch", representative_order_id, battery_id, None,
                self._stage3_features(battery, current_time, route),
            ))
        return candidates

    def _triple_is_feasible(
        self,
        state: StateDict,
        drone_id: str,
        order_id: str,
        battery_id: str,
        return_launchpad_id: str,
    ) -> bool:
        order = state.get("orders", {}).get(order_id)
        battery = state.get("batteries", {}).get(battery_id)
        if order is None or battery is None or self.candidate_details_fn is None:
            return False
        if not self.menu.drone_can_carry(state, drone_id, order):
            return False
        details = self.candidate_details_fn(order, battery)
        if not bool(details.get("feasible", False)):
            return False
        route = details.get("return_plans", {}).get(str(return_launchpad_id))
        return route is not None and bool(route.get("feasible", False))

    # ------------------------------------------------------------------
    # Per-drone plan loop: three independent draws, then one joint check.
    # ------------------------------------------------------------------

    def plan(
        self,
        state: StateDict,
        used_orders: set[str],
        used_batteries: set[str],
    ) -> list[TaskLevelActionDict]:
        current_time = int(state.get("time", 0))
        actions: list[TaskLevelActionDict] = []
        planned_targets: dict[str, int] = {}
        planned_departures: dict[str, int] = {}
        blocked_launchpads: set[str] = set()
        cleared_launchpads: set[str] = set()
        fleet_size = len(state.get("drones", {}))

        for agent_index, (raw_drone_id, drone) in enumerate(state.get("drones", {}).items()):
            drone_id = str(raw_drone_id)
            launchpad_id = drone.get("current_launchpad_id")
            if launchpad_id is None:
                continue
            launchpad_id = str(launchpad_id)
            if launchpad_id in blocked_launchpads:
                continue
            launchpad = state.get("launchpads", {}).get(launchpad_id)
            if launchpad is None or not self.menu.drone_is_available(state, launchpad, drone_id):
                continue
            if launchpad_id not in cleared_launchpads:
                self.menu.clear_terminal_lock(state, launchpad_id, launchpad)
                cleared_launchpads.add(launchpad_id)

            order_candidates = self._stage1_order_candidates(
                state, launchpad, drone_id, used_orders, used_batteries,
                current_time, planned_departures,
            )
            if not order_candidates:
                blocked_launchpads.add(launchpad_id)
                continue

            context = self.menu.context(
                state, launchpad, used_orders, used_batteries, 1,
                planned_targets, planned_departures,
                agent_index=agent_index, agent_count=fleet_size,
            )
            order_index, pending = self.learner.begin_order(
                np.stack([c.features for c in order_candidates]), context,
                training=self.training,
            )
            selected_order = order_candidates[order_index]
            if selected_order.kind in {"stop", "defer"}:
                sample = self.learner.complete_without_dispatch(pending)
                if self.training:
                    self._episode_decisions.append(_JointPlanDecision(
                        current_time, sample, selected_order.kind,
                        launchpad_id, drone_id, selected_order.order_id,
                    ))
                if selected_order.kind == "defer" and selected_order.order_id is not None:
                    self.menu.active_order_by_launchpad[launchpad_id] = selected_order.order_id
                blocked_launchpads.add(launchpad_id)
                continue
            order_id = selected_order.order_id
            if order_id is None:
                raise RuntimeError(self.dispatch_candidate_error)

            # The other two menus are built against the representative, never
            # against `order_id`: that is what makes the heads independent.
            representative_order_id = self._representative_order_id(
                state, launchpad, drone_id, used_orders, used_batteries,
                current_time, planned_departures,
            )
            if representative_order_id is None:
                sample = self.learner.complete_without_dispatch(pending)
                if self.training:
                    self._episode_decisions.append(_JointPlanDecision(
                        current_time, sample, "defer", launchpad_id, drone_id, order_id,
                    ))
                self.menu.active_order_by_launchpad[launchpad_id] = order_id
                blocked_launchpads.add(launchpad_id)
                continue

            return_candidates = self._stage2_return_candidates(
                state, launchpad, representative_order_id, used_batteries,
                current_time, planned_targets, planned_departures,
            )
            if not return_candidates:
                sample = self.learner.complete_without_dispatch(pending)
                if self.training:
                    self._episode_decisions.append(_JointPlanDecision(
                        current_time, sample, "defer", launchpad_id, drone_id, order_id,
                    ))
                self.menu.active_order_by_launchpad[launchpad_id] = order_id
                blocked_launchpads.add(launchpad_id)
                continue
            return_index, pending = self.learner.complete_return(
                pending, np.stack([c.features for c in return_candidates]),
                training=self.training,
            )
            return_launchpad_id = return_candidates[return_index].return_launchpad_id
            if return_launchpad_id is None:
                raise RuntimeError(self.dispatch_candidate_error)

            battery_candidates = self._marginal_battery_candidates(
                state, launchpad, representative_order_id, used_batteries, current_time,
            )
            if not battery_candidates:
                sample = self.learner.complete_without_dispatch(pending)
                if self.training:
                    self._episode_decisions.append(_JointPlanDecision(
                        current_time, sample, "defer", launchpad_id, drone_id, order_id,
                    ))
                self.menu.active_order_by_launchpad[launchpad_id] = order_id
                blocked_launchpads.add(launchpad_id)
                continue
            battery_index, sample = self.learner.complete_battery(
                pending, np.stack([c.features for c in battery_candidates]),
                training=self.training,
            )
            battery_id = battery_candidates[battery_index].battery_id
            if battery_id is None:
                raise RuntimeError(self.dispatch_candidate_error)

            # One joint check, after all three heads have committed.
            self._joint_attempts += 1
            if not self._triple_is_feasible(
                state, drone_id, order_id, battery_id, return_launchpad_id
            ):
                self._joint_infeasible += 1
                if self.training:
                    # the sample carries all three log-probs: the policy is
                    # credited for the triple it chose, which dispatched nothing
                    self._episode_decisions.append(_JointPlanDecision(
                        current_time, sample, "defer", launchpad_id, drone_id, order_id,
                    ))
                blocked_launchpads.add(launchpad_id)
                continue

            if self.training:
                self._episode_decisions.append(_JointPlanDecision(
                    current_time, sample, "dispatch", launchpad_id, drone_id, order_id,
                ))
            actions.append({
                "action_id": (
                    f"{self.dispatch_action_prefix}_{current_time}_{drone_id}_"
                    f"{order_id}_{battery_id}_{return_launchpad_id}"
                ),
                "action_type": "order_battery_dispatch",
                "order_id": order_id,
                "battery_id": battery_id,
                "target_launchpad_id": return_launchpad_id,
                "drone_id": drone_id,
                "feasibility": True,
            })
            used_orders.add(order_id)
            used_batteries.add(battery_id)
            planned_departures[launchpad_id] = planned_departures.get(launchpad_id, 0) + 1
            planned_targets[return_launchpad_id] = planned_targets.get(return_launchpad_id, 0) + 1
            if self.menu.active_order_by_launchpad.get(launchpad_id) == order_id:
                self.menu.active_order_by_launchpad.pop(launchpad_id, None)
        return actions
