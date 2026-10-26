"""Rule-based order-battery dispatch policies (main-table baselines R1-R5).

All five share one skeleton: enumerate the feasible (order, battery) pairs at
each launchpad, score them, and commit the best ones up to the number of idle
drones (greedy by score; R5 solves a max-weight matching instead). They differ
in ``_score``, and R4/R5 additionally override feasibility / selection:

    R1_urgentLower_nearest      HeuristicOrderBatteryPolicy           urgency + value - SoC - distance
    R2_random_random            RandomOrderBatteryPolicy              uniform random
    R3_fifoLower_backlogHeu     LowestFeasibleOrderBatteryPolicy      queue position, then lowest SoC
    R4_fifoFull_backlogHeu      FullChargeFifoOrderBatteryPolicy      queue position, fully charged batteries only
    R5_hungarianGmv_backlogHeu  HungarianOnTimeGmvOrderBatteryPolicy  max-weight matching on predicted on-time value

The matching return policies live in ``algorithm/dispatch/return_policies.py``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from algorithm.common.decision_metrics import OrderBatteryDecisionMetrics
from algorithm.common.geometry_utils import distance_m, safe_ratio, urgency
from simulator.model.enums import BatteryStatus, OrderStatus, VehicleStatus
from simulator.model.types import StateDict, TaskLevelActionDict


class HeuristicOrderBatteryPolicy:
    """Match launchpad orders directly to batteries using every idle drone there."""

    def __init__(self, config: dict):
        self.config = config
        self.training = False
        self.candidate_feasible_fn: Callable[[dict, dict], bool] | None = None
        self._metrics = OrderBatteryDecisionMetrics()

    def plan(
        self,
        state: StateDict,
        used_orders: set[str],
        used_batteries: set[str],
    ) -> list[TaskLevelActionDict]:
        actions: list[TaskLevelActionDict] = []
        current_time = int(state.get("time", 0))
        orders = state.get("orders", {})
        batteries = state.get("batteries", {})
        drones = state.get("drones", {})

        for launchpad in state.get("launchpads", {}).values():
            launchpad_id = str(launchpad.get("launchpad_id"))
            available_drones = sum(
                1
                for drone_id in launchpad.get("drone_ids", [])
                if drone_id in drones
                and drones[drone_id].get("status") == VehicleStatus.IDLE.value
                and not drones[drone_id].get("route_plan")
                and drones[drone_id].get("current_battery_id") is None
            )
            if available_drones <= 0:
                continue
            battery_ids = [
                str(battery_id)
                for battery_id in (
                    list(launchpad.get("charged_battery_ids", []))
                    + list(launchpad.get("charging_battery_ids", []))
                )
                if str(battery_id) not in used_batteries
            ]
            candidates: list[tuple[float, str, str]] = []
            for order_id in launchpad.get("order_queue", []):
                order_id = str(order_id)
                order = orders.get(order_id)
                if (
                    order_id in used_orders
                    or order is None
                    or order.get("status") != OrderStatus.PENDING.value
                ):
                    continue
                for battery_id in battery_ids:
                    battery = batteries.get(battery_id)
                    if battery is None or not self._candidate_feasible(order, battery, launchpad_id):
                        continue
                    candidates.append((self._score(order, battery, launchpad, current_time), order_id, battery_id))

            selected_candidates = self._select_candidates(candidates, available_drones)
            for _, order_id, battery_id in selected_candidates:
                actions.append({
                    "action_id": f"dispatch_{current_time}_{order_id}_{battery_id}",
                    "action_type": "order_battery_dispatch",
                    "order_id": order_id,
                    "battery_id": battery_id,
                    "feasibility": True,
                })
                used_orders.add(order_id)
                used_batteries.add(battery_id)
            self._metrics.record_group(
                order_ids=[candidate[1] for candidate in candidates],
                battery_ids=[candidate[2] for candidate in candidates],
                selected_count=len(selected_candidates),
                available_drones=available_drones,
            )
        return actions

    def _select_candidates(
        self,
        candidates: list[tuple[float, str, str]],
        available_drones: int,
    ) -> list[tuple[float, str, str]]:
        selected: list[tuple[float, str, str]] = []
        selected_orders: set[str] = set()
        selected_batteries: set[str] = set()
        for candidate in sorted(candidates, reverse=True):
            if len(selected) >= available_drones:
                break
            _, order_id, battery_id = candidate
            if order_id in selected_orders or battery_id in selected_batteries:
                continue
            selected.append(candidate)
            selected_orders.add(order_id)
            selected_batteries.add(battery_id)
        return selected

    def _candidate_feasible(self, order: dict, battery: dict, launchpad_id: str) -> bool:
        if battery.get("current_vehicle_id") is not None:
            return False
        if str(battery.get("current_launchpad_id")) != launchpad_id:
            return False
        if battery.get("status") not in {BatteryStatus.CHARGED.value, BatteryStatus.CHARGING.value}:
            return False
        return self.candidate_feasible_fn is None or self.candidate_feasible_fn(order, battery)

    def _score(self, order: dict, battery: dict, launchpad: dict, current_time: int) -> float:
        capacity = float(battery.get("capacity_Wh", 0.0) or 0.0)
        charge = float(battery.get("current_charge_Wh", 0.0) or 0.0)
        # Prefer the lowest feasible state of charge so high-energy batteries
        # remain available unless urgency/value justifies consuming them.
        return (
            1000.0 * urgency(order, current_time)
            + float(order.get("value", 0.0) or 0.0)
            - 100.0 * safe_ratio(charge, capacity)
            - distance_m(launchpad.get("current_location"), order.get("destination")) / 1000.0
        )

    def finish_episode(self, *args, **kwargs) -> dict[str, float]:
        summary = self._metrics.summary()
        self.clear_episode()
        return summary

    def clear_episode(self) -> None:
        self._metrics.clear()

    def save(self, path: str) -> None:
        raise NotImplementedError("HeuristicOrderBatteryPolicy has no trainable model")

    def load(self, path: str) -> None:
        return None


class RandomOrderBatteryPolicy(HeuristicOrderBatteryPolicy):
    """Pick uniformly at random among the feasible (order, battery) pairs."""

    def __init__(self, config: dict, *, seed: int = 0):
        super().__init__(config)
        self._rng = np.random.default_rng(seed)

    def _score(self, order: dict, battery: dict, launchpad: dict, current_time: int) -> float:
        return float(self._rng.random())


class LowestFeasibleOrderBatteryPolicy(HeuristicOrderBatteryPolicy):
    """Serve queued orders FIFO with the lowest-charge feasible battery."""

    def _score(self, order: dict, battery: dict, launchpad: dict, current_time: int) -> float:
        queue = [str(order_id) for order_id in launchpad.get("order_queue", [])]
        order_id = str(order.get("order_id"))
        queue_position = queue.index(order_id) if order_id in queue else len(queue)
        capacity = float(battery.get("capacity_Wh", 0.0) or 0.0)
        charge = float(battery.get("current_charge_Wh", 0.0) or 0.0)
        return -1_000_000.0 * queue_position - safe_ratio(charge, capacity)


class FullChargeFifoOrderBatteryPolicy(LowestFeasibleOrderBatteryPolicy):
    """Serve queued orders FIFO using only fully charged batteries.

    A battery is fully charged iff its status is CHARGED (the simulator sets
    ``current_charge_Wh = capacity_Wh`` at that moment); batteries still
    CHARGING are excluded even when they would already be energy-feasible.
    """

    def _candidate_feasible(self, order: dict, battery: dict, launchpad_id: str) -> bool:
        if battery.get("status") != BatteryStatus.CHARGED.value:
            return False
        return super()._candidate_feasible(order, battery, launchpad_id)


class HungarianOnTimeGmvOrderBatteryPolicy(HeuristicOrderBatteryPolicy):
    """Per-launchpad max-weight (order, battery) matching on predicted on-time value.

    Weight of a feasible pair is the order value if dispatching *now* lands
    before the deadline (ETA = t + straight-line distance / cruise speed, the
    simulator's own flight-time model), else 0. Pairs with zero weight are not
    dispatched: the objective is the on-time GMV realised by this decision, so
    a predicted-late order is left in the queue rather than flown for a
    penalised delivery. A tiny ``-eps * SoC`` term breaks ties towards the
    lowest feasible battery, mirroring the FIFO baselines' battery choice.

    The assignment is solved with the Hungarian algorithm
    (``scipy.optimize.linear_sum_assignment``) and truncated to the idle-drone
    count by weight. Because weights depend on the battery only through the
    tie-breaker, that truncation is the optimum of the drone-capped problem.
    """

    _SOC_TIE_BREAK = 1e-3

    def __init__(self, config: dict):
        super().__init__(config)
        uav_cfg = config.get("vehicles", {}).get("uavs", {})
        self._speed_mps = float(uav_cfg.get("speed_mps", 50.0 / 3.6))

    def _score(self, order: dict, battery: dict, launchpad: dict, current_time: int) -> float:
        eta = current_time + distance_m(launchpad.get("current_location"), order.get("destination")) / self._speed_mps
        if eta > float(order.get("deadline", current_time)):
            return 0.0
        capacity = float(battery.get("capacity_Wh", 0.0) or 0.0)
        charge = float(battery.get("current_charge_Wh", 0.0) or 0.0)
        return float(order.get("value", 0.0) or 0.0) - self._SOC_TIE_BREAK * safe_ratio(charge, capacity)

    def _select_candidates(
        self,
        candidates: list[tuple[float, str, str]],
        available_drones: int,
    ) -> list[tuple[float, str, str]]:
        positive = [candidate for candidate in candidates if candidate[0] > 0.0]
        if not positive or available_drones <= 0:
            return []
        order_ids = sorted({candidate[1] for candidate in positive})
        battery_ids = sorted({candidate[2] for candidate in positive})
        order_index = {order_id: i for i, order_id in enumerate(order_ids)}
        battery_index = {battery_id: j for j, battery_id in enumerate(battery_ids)}
        weights = np.zeros((len(order_ids), len(battery_ids)), dtype=float)
        for weight, order_id, battery_id in positive:
            weights[order_index[order_id], battery_index[battery_id]] = weight
        rows, cols = linear_sum_assignment(weights, maximize=True)
        matched = [
            (float(weights[i, j]), order_ids[i], battery_ids[j])
            for i, j in zip(rows, cols)
            if weights[i, j] > 0.0  # zero entries are infeasible pairs
        ]
        matched.sort(reverse=True)
        return matched[:available_drones]
