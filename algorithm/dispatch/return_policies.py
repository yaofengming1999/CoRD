from __future__ import annotations

from typing import Callable

import numpy as np

from algorithm.common.geometry_utils import distance_m, relocation_settings, relocation_worthwhile
from simulator.model.enums import VehicleStatus
from simulator.model.types import StateDict, TaskLevelActionDict


class HeuristicDroneReturnPolicy:
    def __init__(self, config: dict):
        self.training = False
        self.rebalance_enabled, self.rebalance_interval_s = relocation_settings(config)
        self._next_rebalance_due: dict[str, int] = {}
        self._rebalances_planned = 0
        self.return_feasible_fn: Callable[[dict, dict], bool] | None = None
        self.rebalance_feasible_fn: Callable[[dict, dict, dict], bool] | None = None

    def plan(
        self,
        state: StateDict,
        used_drones: set[str],
        used_batteries: set[str] | None = None,
    ) -> list[TaskLevelActionDict]:
        used_batteries = used_batteries if used_batteries is not None else set()
        current_time = int(state.get("time", 0))
        launchpads = list(state.get("launchpads", {}).values())
        batteries = state.get("batteries", {})
        actions: list[TaskLevelActionDict] = []
        for drone in state.get("drones", {}).values():
            drone_id = str(drone.get("vehicle_id"))
            if (
                drone_id in used_drones
                or drone.get("status") != VehicleStatus.IDLE.value
                or drone.get("route_plan")
                or drone.get("onboard_order_ids")
            ):
                continue
            if drone.get("current_launchpad_id") is None:
                target = self._best_return_target(drone, launchpads, state)
                if target is None:
                    continue
                actions.append({
                    "action_id": f"return_{current_time}_{drone_id}_{target['launchpad_id']}",
                    "action_type": "drone_return",
                    "drone_id": drone_id,
                    "target_launchpad_id": str(target["launchpad_id"]),
                    "feasibility": True,
                })
            else:
                if not self._rebalance_due(state, drone, launchpads, current_time):
                    continue
                choice = self._best_rebalance(drone, launchpads, batteries, used_batteries)
                if choice is None:
                    continue
                battery, target = choice
                actions.append({
                    "action_id": f"rebalance_{current_time}_{drone_id}_{battery['battery_id']}_{target['launchpad_id']}",
                    "action_type": "drone_rebalance",
                    "drone_id": drone_id,
                    "battery_id": str(battery["battery_id"]),
                    "target_launchpad_id": str(target["launchpad_id"]),
                    "feasibility": True,
                })
                used_batteries.add(str(battery["battery_id"]))
                self._rebalances_planned += 1
            used_drones.add(drone_id)
        return actions

    def _best_return_target(
        self,
        drone: dict,
        launchpads: list[dict],
        state: StateDict | None = None,
    ) -> dict | None:
        feasible = self._feasible_return_targets(drone, launchpads)
        if not feasible:
            return None
        return max(feasible, key=lambda launchpad: self._target_score(drone, launchpad))

    def _feasible_return_targets(self, drone: dict, launchpads: list[dict]) -> list[dict]:
        return [
            launchpad
            for launchpad in launchpads
            if (
                self.return_feasible_fn(drone, launchpad)
                if self.return_feasible_fn is not None
                else True
            )
        ]

    def _best_rebalance(
        self,
        drone: dict,
        launchpads: list[dict],
        batteries: dict[str, dict],
        used_batteries: set[str] | None = None,
    ) -> tuple[dict, dict] | None:
        choices = self._feasible_rebalance_choices(
            drone, launchpads, batteries, used_batteries
        )
        if not choices:
            return None
        return max(
            choices,
            key=lambda choice: (
                self._target_score(drone, choice[1]),
                -float(choice[0].get("current_charge_Wh", 0.0) or 0.0),
            ),
        )

    def _feasible_rebalance_choices(
        self,
        drone: dict,
        launchpads: list[dict],
        batteries: dict[str, dict],
        used_batteries: set[str] | None = None,
    ) -> list[tuple[dict, dict]]:
        current_launchpad_id = str(drone.get("current_launchpad_id"))
        current_launchpad = next(
            (launchpad for launchpad in launchpads if str(launchpad.get("launchpad_id")) == current_launchpad_id),
            None,
        )
        if current_launchpad is None:
            return []
        battery_ids = list(current_launchpad.get("charged_battery_ids", [])) + list(current_launchpad.get("charging_battery_ids", []))
        choices: list[tuple[dict, dict]] = []
        used_batteries = used_batteries if used_batteries is not None else set()
        for target in launchpads:
            if str(target.get("launchpad_id")) == current_launchpad_id:
                continue
            for battery_id in battery_ids:
                if str(battery_id) in used_batteries:
                    continue
                battery = batteries.get(str(battery_id))
                if battery is None:
                    continue
                if self.rebalance_feasible_fn is not None and not self.rebalance_feasible_fn(drone, battery, target):
                    continue
                choices.append((battery, target))
        return choices

    def _target_score(self, drone: dict, launchpad: dict) -> float:
        backlog = len(launchpad.get("order_queue", []))
        docked = len(launchpad.get("drone_ids", []))
        return (
            100.0 * backlog
            - 100.0 * docked
            - distance_m(drone.get("current_location"), launchpad.get("current_location"))
        )

    def _rebalance_due(self, state: StateDict, drone: dict, launchpads: list[dict], current_time: int) -> bool:
        if not self.rebalance_enabled or not relocation_worthwhile(state, drone, launchpads):
            return False
        drone_id = str(drone.get("vehicle_id"))
        due = self._next_rebalance_due.get(drone_id, current_time)
        if current_time < due:
            return False
        self._next_rebalance_due[drone_id] = current_time + self.rebalance_interval_s
        return True

    def finish_episode(self, *args, **kwargs) -> dict[str, float]:
        result = {"drone_rebalances": float(self._rebalances_planned)}
        self.clear_episode()
        return result

    def clear_episode(self) -> None:
        self._next_rebalance_due = {}
        self._rebalances_planned = 0

    def save(self, path: str) -> None:
        raise NotImplementedError("HeuristicDroneReturnPolicy has no trainable model")

    def load(self, path: str) -> None:
        return None


class NearestDroneReturnPolicy(HeuristicDroneReturnPolicy):
    """Choose the geographically nearest feasible return launchpad."""

    def _target_score(self, drone: dict, launchpad: dict) -> float:
        return -distance_m(drone.get("current_location"), launchpad.get("current_location"))


class RandomDroneReturnPolicy(HeuristicDroneReturnPolicy):
    """Choose uniformly from feasible return or rebalance actions."""

    def __init__(self, config: dict, *, seed: int = 0):
        super().__init__(config)
        self._rng = np.random.default_rng(seed)

    def _best_return_target(
        self,
        drone: dict,
        launchpads: list[dict],
        state: StateDict | None = None,
    ) -> dict | None:
        feasible = self._feasible_return_targets(drone, launchpads)
        if not feasible:
            return None
        return feasible[int(self._rng.integers(len(feasible)))]

    def _best_rebalance(
        self,
        drone: dict,
        launchpads: list[dict],
        batteries: dict[str, dict],
        used_batteries: set[str] | None = None,
    ) -> tuple[dict, dict] | None:
        choices = self._feasible_rebalance_choices(
            drone, launchpads, batteries, used_batteries
        )
        if not choices:
            return None
        return choices[int(self._rng.integers(len(choices)))]


class PreplannedDroneReturnPolicy(HeuristicDroneReturnPolicy):
    """No-op endpoint policy for dispatch actions that already include return."""

    def plan(
        self,
        state: StateDict,
        used_drones: set[str],
        used_batteries: set[str] | None = None,
    ) -> list[TaskLevelActionDict]:
        return []
