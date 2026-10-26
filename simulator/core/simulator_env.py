from __future__ import annotations

import logging
from typing import Any, Optional

from ..io import InstanceLoader
from ..metrics.recorder import MetricsRecorder
from ..model.entities import Battery, Drone, Launchpad, Order, Stop
from ..model.enums import BatteryStatus, OrderStatus, StopType, VehicleStatus
from ..model.types import StateDict, TaskLevelActionDict
from ..routing.air_router import AirRouter
from ..world.airspace import Airspace
from ..world.weather import Weather
from .state_builder import StateBuilder
from .time_advancement import next_simulation_time
from .transition_engine import TransitionEngine

logger = logging.getLogger(__name__)


class SimulatorEnv:
    """Drone-only simulator with battery dispatch, return, and rebalancing."""

    def __init__(self, config: dict):
        self.config = config
        self._sim_cfg = config.get("simulation", {})
        self.horizon = int(self._sim_cfg.get("horizon_seconds", 28800))
        self.max_horizon = int(self._sim_cfg.get("max_horizon_seconds", self.horizon))
        self.delta_t = int(self._sim_cfg.get("delta_t_seconds", 60))

        self.airspace = Airspace(config)
        self.weather = Weather(config)
        self.air_router: AirRouter | None = None
        self._route_energy_cache: dict[tuple, dict] = {}
        self.transition_engine: TransitionEngine | None = None
        self.instance_loader: InstanceLoader | None = None
        self.state_builder = StateBuilder()
        # Replay frame recording (+ save-time trajectory interpolation) is the
        # dominant episode cost and only feeds the HTML replay — off unless a
        # run explicitly asks for it via output.record_replay.
        self.metrics_recorder = MetricsRecorder(
            record_frames=bool(config.get("output", {}).get("record_replay", False))
        )
        self.selected_instance_path = None

        self.current_time = 0
        self.orders: dict[str, Order] = {}
        self._pending_orders_by_time: list[Order] = []
        self._next_order_index = 0
        self.drones: dict[str, Drone] = {}
        self.launchpads: dict[str, Launchpad] = {}
        self.batteries: dict[str, Battery] = {}
        self._capacity_batch_action_ids: set[str] = set()
        self._step_count = 0
        self._last_step_delta_t = self.delta_t
        self.last_termination_reason: str | None = None
        self._loaded = False
        self.state: StateDict = self._empty_state()

    def reset(self, seed: Optional[int] = None) -> StateDict:
        selected_seed = seed if seed is not None else int(self._sim_cfg.get("seed", 42))
        if not self._loaded:
            self.airspace.load()
            self._loaded = True

        self.instance_loader = InstanceLoader(self.config, road_network=None)
        instance = self.instance_loader.load(seed=selected_seed)
        self.max_horizon = self._resolve_max_horizon(instance.orders)
        self.selected_instance_path = instance.instance_path
        runtime = self.config.setdefault("_runtime", {})
        runtime["selected_instance_path"] = str(instance.instance_path)
        runtime["resolved_max_horizon_seconds"] = self.max_horizon
        self._apply_instance_weather(instance.weather)

        self.weather = Weather(self.config)
        self.air_router = AirRouter(self.config, self.airspace, self.weather)
        self.transition_engine = TransitionEngine(self.config, self.air_router, self.metrics_recorder)
        self._route_energy_cache: dict[tuple, dict] = {}  # per-episode (order, pad, drone type) -> routes
        self.current_time = 0
        self.orders = {}
        self._pending_orders_by_time = list(instance.orders)
        self._next_order_index = 0
        self.drones = {drone.vehicle_id: drone for drone in instance.drones}
        self.launchpads = {launchpad.launchpad_id: launchpad for launchpad in instance.launchpads}
        self.batteries = {battery.battery_id: battery for battery in instance.batteries}
        self._configure_launchpad_capacity()
        self._step_count = 0
        self._last_step_delta_t = self.delta_t
        self.last_termination_reason = None

        self.weather.reset()
        self.metrics_recorder.reset()
        self._release_orders(0)
        self._assert_capacity_invariant("reset")
        self._refresh_state()
        self.metrics_recorder.record_frame(
            self.current_time,
            self.drones,
            self.orders,
            launchpads=self.launchpads,
            batteries=self.batteries,
            weather_state=self.weather.get_state(),
            stats=self.state["stats"],
            frame_kind="reset",
        )
        return self.state

    def step(self, action: Optional[dict] = None) -> tuple[StateDict, float, bool, dict]:
        if self.transition_engine is None:
            raise RuntimeError("SimulatorEnv.reset() must be called before step()")
        advance = next_simulation_time(
            config=self.config,
            current_time=self.current_time,
            max_horizon=self.max_horizon,
            default_delta_t=self.delta_t,
            pending_orders_by_time=self._pending_orders_by_time,
            next_order_index=self._next_order_index,
            drones=self.drones,
            orders=self.orders,
            batteries=self.batteries,
        )
        interval_start = self.current_time
        interval_end = advance.next_time
        new_orders = self._release_orders(interval_start)
        self.transition_engine.advance(
            self.drones,
            self.orders,
            interval_start,
            advance.delta_t,
            None,
            self.launchpads,
            self.batteries,
        )
        self.weather.step(interval_start)
        self.metrics_recorder.log_step(
            interval_start,
            self.drones,
            self.orders,
            launchpads=self.launchpads,
            batteries=self.batteries,
            weather_state=self.weather.get_state(),
            interval_end=interval_end,
        )

        self._last_step_delta_t = advance.delta_t
        self.current_time = interval_end
        self._step_count += 1
        new_orders.extend(self._release_orders(self.current_time))
        self._refresh_state()
        self.metrics_recorder.record_frame(
            self.current_time,
            self.drones,
            self.orders,
            launchpads=self.launchpads,
            batteries=self.batteries,
            weather_state=self.weather.get_state(),
            stats=self.state["stats"],
            frame_kind="step_start",
        )
        reward = self._compute_step_reward(interval_start, interval_end)
        done, termination_reason = self._episode_done()
        self.last_termination_reason = termination_reason
        if done:
            self._assert_episode_closed(termination_reason)
        return self.state, reward, done, {
            "step": self._step_count,
            "new_orders": len(new_orders),
            "time_advance_mode": self._time_advancement_mode(),
            "time_advance_reason": advance.reason,
            "delta_t": advance.delta_t,
            "termination_reason": termination_reason,
        }

    def apply_actions(self, actions: list[TaskLevelActionDict]) -> None:
        normalized_actions = [
            normalized
            for action in actions
            if (normalized := self._normalize_action(action)) is not None
            and normalized.get("feasibility", True)
        ]
        # Validate the complete movement batch before mutating any simulator
        # state. A policy must never silently lose a return/rebalance action
        # because the target was already committed full.
        self._validate_capacity_action_batch(normalized_actions)
        self._capacity_batch_action_ids = {
            str(action.get("action_id"))
            for action in normalized_actions
            if action.get("action_type") in {"drone_return", "drone_rebalance"}
            or (
                action.get("action_type") == "order_battery_dispatch"
                and action.get("target_launchpad_id") is not None
            )
        }

        applied: list[TaskLevelActionDict] = []
        try:
            for normalized in normalized_actions:
                action_type = normalized["action_type"]
                accepted = False
                if action_type == "order_battery_dispatch":
                    accepted = self._apply_order_battery_dispatch(normalized)
                elif action_type == "drone_return":
                    accepted = self._apply_drone_return(normalized)
                elif action_type == "drone_rebalance":
                    accepted = self._apply_drone_rebalance(normalized)
                if accepted:
                    applied.append(normalized)
            self._assert_capacity_invariant("action batch")
            if applied:
                self.metrics_recorder.log_actions(self.current_time, self._step_count, applied)
                self._refresh_state()
        finally:
            self._capacity_batch_action_ids = set()

    def get_state(self) -> StateDict:
        return self.state

    def _apply_order_battery_dispatch(self, action: TaskLevelActionDict) -> bool:
        order = self.orders.get(str(action.get("order_id", "")))
        battery = self.batteries.get(str(action.get("battery_id", "")))
        if order is None or battery is None or order.status != OrderStatus.PENDING:
            return False
        launchpad = self.launchpads.get(order.origin_launchpad_id)
        if launchpad is None or order.order_id not in launchpad.order_queue:
            return False
        requested_drone_id = action.get("drone_id")
        candidate_drone_ids = (
            {str(requested_drone_id)}
            if requested_drone_id is not None
            else None
        )
        drone = self._available_drone_at_launchpad(
            launchpad,
            order,
            candidate_drone_ids,
        )
        if drone is None or not self._can_select_launchpad_battery(battery, launchpad):
            return False
        feasible_returns = self._energy_feasible_task_return_launchpads(
            drone,
            order,
            launchpad,
            battery,
        )
        if not feasible_returns:
            return False
        target_id = action.get("target_launchpad_id")
        return_target = (
            self.launchpads.get(str(target_id))
            if target_id is not None
            else None
        )
        if target_id is not None and (
            return_target is None
            or return_target.launchpad_id
            not in {candidate.launchpad_id for candidate in feasible_returns}
        ):
            return False
        dispatch_metrics = self.order_battery_candidate_details(
            {"order_id": order.order_id},
            {"battery_id": battery.battery_id},
        )
        return_plan = (
            dispatch_metrics.get("return_plans", {}).get(return_target.launchpad_id)
            if return_target is not None
            else None
        )
        if return_target is not None and return_plan is None:
            return False
        planned_return_metrics = (
            self._accepted_preplanned_return_metrics(
                order,
                launchpad,
                return_target,
                battery,
                dispatch_metrics,
                return_plan,
            )
            if return_target is not None and return_plan is not None
            else None
        )
        if not self._mount_launchpad_battery(drone, launchpad, battery):
            return False

        launchpad.order_queue.remove(order.order_id)
        launchpad.drone_ids.remove(drone.vehicle_id)
        launchpad.service_events += 1
        drone.current_launchpad_id = None
        drone.onboard_order_ids.append(order.order_id)
        drone.remaining_capacity -= order.size
        drone.route_plan.append(Stop(
            stop_id=f"{action['action_id']}_delivery",
            location=order.destination,
            stop_type=StopType.DELIVERY,
            related_order_id=order.order_id,
            service_time_seconds=int(self.config.get("service_times", {}).get("delivery_seconds", 0)),
        ))
        if return_target is not None:
            self._reserve_return_target(return_target)
            drone.route_plan.append(Stop(
                stop_id=f"{action['action_id']}_return",
                location=return_target.location,
                stop_type=StopType.DRONE_RETURN,
                related_launchpad_id=return_target.launchpad_id,
            ))
        drone.status = VehicleStatus.MOVING
        order.status = OrderStatus.IN_TRANSIT
        order.assigned_drone_id = drone.vehicle_id
        order.selected_battery_id = battery.battery_id
        if return_target is not None:
            order.return_launchpad_id = return_target.launchpad_id
        order.dispatch_time = self.current_time
        # Which return plan the round-trip numbers are read from: the pad this action
        # committed to, or -- for a two-stage action that names none -- the
        # energy-minimal pad, which is the plan its feasibility test used. This picks
        # the return leg only. The delivery margin is the same in every plan, because
        # the loaded leg does not depend on the return pad and the reserve is a fixed
        # share of capacity (safe_return_energy_Wh is 0 on this path), so it is
        # comparable across preplanned and two-stage policies as it stands.
        energy_plan = return_plan if return_plan is not None else dispatch_metrics
        self.metrics_recorder.log_battery_dispatch_metrics(
            selected_charge_Wh=float(dispatch_metrics["available_charge_Wh"]),
            battery_capacity_Wh=battery.capacity_Wh,
            minimum_required_Wh=float(energy_plan["minimum_required_charge_Wh"]),
            delivery_energy_Wh=float(energy_plan["loaded_energy_Wh"]),
            return_energy_Wh=float(energy_plan["return_energy_Wh"]),
            feasible_return_launchpads=int(dispatch_metrics["feasible_return_count"]),
        )
        if planned_return_metrics is not None:
            self.metrics_recorder.log_return_metrics(**planned_return_metrics)
        return True

    def _apply_drone_return(self, action: TaskLevelActionDict) -> bool:
        drone = self.drones.get(str(action.get("drone_id", "")))
        target = self.launchpads.get(str(action.get("target_launchpad_id", "")))
        if drone is None or target is None:
            return False
        if (
            drone.status != VehicleStatus.IDLE
            or drone.route_plan
            or drone.onboard_order_ids
            or drone.current_launchpad_id is not None
        ):
            return False
        if not self._drone_can_reach_launchpad(drone, target):
            return False
        if (
            str(action.get("action_id")) not in self._capacity_batch_action_ids
            and not self._return_target_has_capacity(target)
        ):
            self._raise_action_capacity_error(action, drone, target)
        return_metrics = self._accepted_return_metrics(drone, target)
        self._reserve_return_target(target)
        drone.route_plan.append(Stop(
            stop_id=f"{action['action_id']}_return",
            location=target.location,
            stop_type=StopType.DRONE_RETURN,
            related_launchpad_id=target.launchpad_id,
        ))
        for order in self.orders.values():
            if (
                order.assigned_drone_id == drone.vehicle_id
                and order.status == OrderStatus.DELIVERED
                and order.return_launchpad_id is None
            ):
                order.return_launchpad_id = target.launchpad_id
        drone.status = VehicleStatus.MOVING
        self.metrics_recorder.log_return_metrics(**return_metrics)
        return True

    def _apply_drone_rebalance(self, action: TaskLevelActionDict) -> bool:
        drone = self.drones.get(str(action.get("drone_id", "")))
        battery = self.batteries.get(str(action.get("battery_id", "")))
        target = self.launchpads.get(str(action.get("target_launchpad_id", "")))
        if drone is None or battery is None or target is None:
            return False
        departure = self.launchpads.get(drone.current_launchpad_id or "")
        if (
            departure is None
            or departure.launchpad_id == target.launchpad_id
            or drone.status != VehicleStatus.IDLE
            or drone.route_plan
            or drone.onboard_order_ids
            or drone.current_battery_id is not None
            or not self._can_select_launchpad_battery(battery, departure)
        ):
            return False
        if (
            str(action.get("action_id")) not in self._capacity_batch_action_ids
            and not self._return_target_has_capacity(target)
        ):
            self._raise_action_capacity_error(action, drone, target)
        charge = self._estimated_battery_charge_at(battery, self.current_time)
        if not self._battery_can_cover_empty_leg(drone, target, charge, battery.capacity_Wh):
            return False
        distance_m = self.air_router.travel_distance(departure.location, target.location)
        energy_Wh = self.air_router.energy_cost(
            departure.location,
            target.location,
            payload_mass_kg=0.0,
            empty_mass_kg=drone.empty_mass_kg,
        )
        arrival_margin_Wh = (
            charge - energy_Wh - self.air_router.reserve_energy(battery.capacity_Wh)
        )
        if not self._mount_launchpad_battery(drone, departure, battery):
            return False
        departure.drone_ids.remove(drone.vehicle_id)
        drone.current_launchpad_id = None
        self._reserve_return_target(target)
        drone.route_plan.append(Stop(
            stop_id=f"{action['action_id']}_rebalance",
            location=target.location,
            stop_type=StopType.DRONE_RETURN,
            related_launchpad_id=target.launchpad_id,
        ))
        drone.status = VehicleStatus.MOVING
        self.metrics_recorder.log_rebalance_metrics(
            selected_charge_Wh=charge,
            battery_capacity_Wh=battery.capacity_Wh,
            distance_m=distance_m,
            energy_Wh=energy_Wh,
            arrival_margin_Wh=arrival_margin_Wh,
        )
        return True

    def order_battery_candidate_feasible(self, order_state: dict, battery_state: dict) -> bool:
        order = self.orders.get(str(order_state.get("order_id", "")))
        battery = self.batteries.get(str(battery_state.get("battery_id", "")))
        if order is None or battery is None or order.status != OrderStatus.PENDING:
            return False
        launchpad = self.launchpads.get(order.origin_launchpad_id)
        if launchpad is None:
            return False
        drone = self._available_drone_at_launchpad(launchpad, order)
        return bool(
            drone is not None
            and self._can_select_launchpad_battery(battery, launchpad)
            and self._has_feasible_task_return(drone, order, launchpad, battery)
        )

    def order_route_energy_options(self, order_state: dict) -> dict | None:
        """Battery-free energy of every (order, return launchpad) route.

        Factored counterpart of :meth:`order_battery_candidate_details`: the task
        energy, distances, times and payload/range feasibility depend only on the
        order, the departure pad and the return pad, so they are computed ONCE per
        (order, return pad) instead of once per (order, battery, return pad).
        Battery feasibility is a scalar test the caller applies against
        :meth:`battery_dispatch_state`:  charge - E_task >= reserve.
        Returns None when the order cannot be dispatched at all (not pending,
        no idle drone able to carry it).
        """
        order = self.orders.get(str(order_state.get("order_id", "")))
        if order is None or order.status != OrderStatus.PENDING:
            return None
        departure = self.launchpads.get(order.origin_launchpad_id)
        if departure is None:
            return None
        drone = self._available_drone_at_launchpad(departure, order)
        if drone is None or self.air_router is None:
            return None
        task_drone = {
            "current_location": departure.location,
            "current_launchpad_id": departure.launchpad_id,
            "battery_level": None,     # battery-free: only E_*, distances, times, payload/range flags are read
            "battery_capacity": None,
            "empty_mass_kg": drone.empty_mass_kg,
            "max_payload_kg": drone.max_payload_kg,
            "max_range_m": drone.max_range_m,
        }
        # Route energy is static within an episode (pads, destinations and the
        # per-instance wind do not move), so it is memoised per
        # (order, departure, drone type): an order is re-evaluated at every step
        # until it is dispatched, and would otherwise cost |L| energy calls each time.
        cache_key = (
            order.order_id, departure.launchpad_id,
            float(drone.empty_mass_kg), float(drone.max_payload_kg), float(drone.max_range_m),
        )
        routes = self._route_energy_cache.get(cache_key)
        if routes is None:
            routes = {}
            for return_launchpad in self._task_return_launchpads(departure):
                energy = self.air_router.estimate_task_energy(task_drone, order, departure, return_launchpad)
                routes[return_launchpad.launchpad_id] = {
                    "route_ok": bool(energy["payload_feasible"]) and bool(energy["range_feasible"]),
                    "E_task_Wh": float(energy["E_task_Wh"]),
                    "E_return_Wh": float(energy["E_return_Wh"]),
                    "distance_loaded_m": float(energy["distance_loaded_m"]),
                    "distance_return_m": float(energy["distance_return_m"]),
                    "time_return_s": float(energy["time_return_s"]),
                }
            self._route_energy_cache[cache_key] = routes
        return {"departure_launchpad_id": departure.launchpad_id, "routes": routes}

    def battery_dispatch_state(self, battery_state: dict, launchpad_state: dict) -> dict | None:
        """Per-battery quantities the factored dispatch menu needs (None = not selectable)."""
        battery = self.batteries.get(str(battery_state.get("battery_id", "")))
        launchpad = self.launchpads.get(str(launchpad_state.get("launchpad_id", "")))
        if battery is None or launchpad is None or self.air_router is None:
            return None
        if not self._can_select_launchpad_battery(battery, launchpad):
            return None
        capacity = float(battery.capacity_Wh)
        return {
            "charge_Wh": self._estimated_battery_charge_at(battery, self.current_time),
            "capacity_Wh": capacity,
            # same reserve estimate_task_energy applies (safe_return_energy_Wh = 0)
            "reserve_Wh": float(self.air_router.energy_model.reserve_energy_Wh(capacity, 0.0)),
        }

    def order_battery_candidate_details(self, order_state: dict, battery_state: dict) -> dict:
        """Energy details for one policy-visible (order, battery) edge."""
        empty = {
            "feasible": False,
            "available_charge_Wh": 0.0,
            "minimum_required_charge_Wh": 0.0,
            "task_energy_Wh": 0.0,
            "loaded_energy_Wh": 0.0,
            "return_energy_Wh": 0.0,
            "task_energy_ratio": 0.0,
            "reserve_after_ratio": 0.0,
            "feasible_return_count": 0,
            "feasible_return_ratio": 0.0,
            "min_return_distance_m": 0.0,
            "max_return_distance_m": 0.0,
            "return_launchpad_ids": [],
            "return_plans": {},
        }
        order = self.orders.get(str(order_state.get("order_id", "")))
        battery = self.batteries.get(str(battery_state.get("battery_id", "")))
        if order is None or battery is None or order.status != OrderStatus.PENDING:
            return empty
        departure = self.launchpads.get(order.origin_launchpad_id)
        if departure is None or not self._can_select_launchpad_battery(battery, departure):
            return empty
        drone = self._available_drone_at_launchpad(departure, order)
        if drone is None or self.air_router is None:
            return empty
        charge = self._estimated_battery_charge_at(battery, self.current_time)
        options = self._task_return_energy_options(drone, order, departure, battery)
        if not options:
            return {**empty, "available_charge_Wh": charge}
        minimum_energy = min(options, key=lambda option: float(option[1]["E_task_Wh"]))[1]
        distances = [float(energy["distance_return_m"]) for _, energy in options]
        capacity = max(1.0, float(battery.capacity_Wh))
        return_plans = {
            launchpad.launchpad_id: {
                "feasible": True,
                "return_launchpad_id": launchpad.launchpad_id,
                "minimum_required_charge_Wh": (
                    float(energy["E_task_Wh"]) + float(energy["B_reserve_Wh"])
                ),
                "task_energy_Wh": float(energy["E_task_Wh"]),
                # the delivery leg on its own: what a dispatch-only agent would have to
                # cover if it were not also responsible for the flight back to l
                "loaded_energy_Wh": float(energy["E_loaded_Wh"]),
                "return_energy_Wh": float(energy["E_return_Wh"]),
                "task_energy_ratio": float(energy["E_task_Wh"]) / capacity,
                "return_energy_ratio": float(energy["E_return_Wh"]) / capacity,
                "reserve_after_ratio": float(energy["reserve_after_ratio"]),
                "arrival_margin_Wh": (
                    float(energy["B_after_Wh"]) - float(energy["B_reserve_Wh"])
                ),
                "distance_loaded_m": float(energy["distance_loaded_m"]),
                "distance_return_m": float(energy["distance_return_m"]),
                "time_return_s": float(energy["time_return_s"]),
            }
            for launchpad, energy in options
        }
        return {
            "feasible": True,
            "available_charge_Wh": charge,
            "minimum_required_charge_Wh": (
                float(minimum_energy["E_task_Wh"]) + float(minimum_energy["B_reserve_Wh"])
            ),
            # same three quantities for the energy-minimal return, which is the plan a
            # two-stage policy is implicitly feasible against when it names no pad
            "task_energy_Wh": float(minimum_energy["E_task_Wh"]),
            "loaded_energy_Wh": float(minimum_energy["E_loaded_Wh"]),
            "return_energy_Wh": float(minimum_energy["E_return_Wh"]),
            "task_energy_ratio": float(minimum_energy["E_task_Wh"]) / capacity,
            "reserve_after_ratio": float(minimum_energy["reserve_after_ratio"]),
            "feasible_return_count": len(options),
            "feasible_return_ratio": len(options) / max(1, len(self.launchpads)),
            "min_return_distance_m": min(distances),
            "max_return_distance_m": max(distances),
            "return_launchpad_ids": [launchpad.launchpad_id for launchpad, _ in options],
            "return_plans": return_plans,
        }

    def drone_return_candidate_feasible(self, drone_state: dict, launchpad_state: dict) -> bool:
        drone = self.drones.get(str(drone_state.get("vehicle_id", "")))
        launchpad = self.launchpads.get(str(launchpad_state.get("launchpad_id", "")))
        return bool(
            drone is not None
            and launchpad is not None
            and drone.current_launchpad_id is None
            and self._return_target_has_capacity(launchpad)
            and self._drone_can_reach_launchpad(drone, launchpad)
        )

    def drone_rebalance_candidate_feasible(
        self,
        drone_state: dict,
        battery_state: dict,
        launchpad_state: dict,
    ) -> bool:
        drone = self.drones.get(str(drone_state.get("vehicle_id", "")))
        battery = self.batteries.get(str(battery_state.get("battery_id", "")))
        target = self.launchpads.get(str(launchpad_state.get("launchpad_id", "")))
        if drone is None or battery is None or target is None:
            return False
        departure = self.launchpads.get(drone.current_launchpad_id or "")
        if departure is None or departure.launchpad_id == target.launchpad_id:
            return False
        charge = self._estimated_battery_charge_at(battery, self.current_time)
        return bool(
            self._return_target_has_capacity(target)
            and self._can_select_launchpad_battery(battery, departure)
            and self._battery_can_cover_empty_leg(drone, target, charge, battery.capacity_Wh)
        )

    def _available_drone_at_launchpad(
        self,
        launchpad: Launchpad,
        order: Order,
        candidate_drone_ids: set[str] | None = None,
    ) -> Drone | None:
        available_ids = (
            set(launchpad.drone_ids)
            if candidate_drone_ids is None
            else candidate_drone_ids
        )
        candidates = [
            self.drones[drone_id]
            for drone_id in sorted(available_ids)
            if drone_id in self.drones
            and self.drones[drone_id].status == VehicleStatus.IDLE
            and not self.drones[drone_id].route_plan
            and not self.drones[drone_id].onboard_order_ids
            and self.drones[drone_id].current_battery_id is None
            and self.drones[drone_id].remaining_capacity >= order.size
            and (self.drones[drone_id].max_payload_kg is None or self.drones[drone_id].max_payload_kg >= order.weight)
        ]
        return candidates[0] if candidates else None

    def _can_select_launchpad_battery(self, battery: Battery, launchpad: Launchpad) -> bool:
        if battery.current_launchpad_id != launchpad.launchpad_id or battery.current_vehicle_id is not None:
            return False
        if battery.status == BatteryStatus.CHARGED:
            return battery.battery_id in launchpad.charged_battery_ids
        if battery.status == BatteryStatus.CHARGING:
            return battery.battery_id in launchpad.charging_battery_ids
        return False

    def _mount_launchpad_battery(self, drone: Drone, launchpad: Launchpad, battery: Battery) -> bool:
        if (
            drone.current_battery_id is not None
            or drone.current_launchpad_id != launchpad.launchpad_id
            or not self._can_select_launchpad_battery(battery, launchpad)
        ):
            return False
        battery.current_charge_Wh = self._estimated_battery_charge_at(battery, self.current_time)
        if battery.battery_id in launchpad.charged_battery_ids:
            launchpad.charged_battery_ids.remove(battery.battery_id)
        if battery.battery_id in launchpad.charging_battery_ids:
            launchpad.charging_battery_ids.remove(battery.battery_id)
        battery.status = BatteryStatus.IN_USE
        battery.current_vehicle_id = drone.vehicle_id
        battery.current_launchpad_id = None
        battery.charge_started_at = None
        battery.charge_start_Wh = None
        battery.charge_complete_at = None
        drone.current_battery_id = battery.battery_id
        drone.battery_capacity = battery.capacity_Wh
        drone.battery_level = battery.current_charge_Wh
        return True

    def _energy_feasible_task_return_launchpads(
        self,
        drone: Drone,
        order: Order,
        departure: Launchpad,
        battery: Battery,
    ) -> list[Launchpad]:
        return [
            launchpad
            for launchpad, _ in self._task_return_energy_options(
                drone,
                order,
                departure,
                battery,
            )
        ]

    def _task_return_energy_options(
        self,
        drone: Drone,
        order: Order,
        departure: Launchpad,
        battery: Battery,
    ) -> list[tuple[Launchpad, dict]]:
        if self.air_router is None:
            return []
        charge = self._estimated_battery_charge_at(battery, self.current_time)
        task_drone = {
            "current_location": departure.location,
            "current_launchpad_id": departure.launchpad_id,
            "battery_level": charge,
            "battery_capacity": battery.capacity_Wh,
            "empty_mass_kg": drone.empty_mass_kg,
            "max_payload_kg": drone.max_payload_kg,
            "max_range_m": drone.max_range_m,
        }
        feasible: list[tuple[Launchpad, dict]] = []
        for return_launchpad in self._task_return_launchpads(departure):
            energy = self.air_router.estimate_task_energy(
                task_drone,
                order,
                departure,
                return_launchpad,
            )
            if energy["is_feasible"]:
                feasible.append((return_launchpad, energy))
        return feasible

    def _has_feasible_task_return(
        self,
        drone: Drone,
        order: Order,
        departure: Launchpad,
        battery: Battery,
    ) -> bool:
        """Fast boolean: is there AT LEAST ONE feasible return launchpad?

        The order-battery feasibility mask only needs a yes/no, but runs once
        per (order, battery) candidate every step, so building the full
        feasible-launchpad list there is the dominant cost. Short-circuit on
        the first feasible pad, and try the departure launchpad first — a
        drone can often return to where it launched on a short round trip, so
        the common case exits immediately.
        """
        if self.air_router is None:
            return False
        charge = self._estimated_battery_charge_at(battery, self.current_time)
        task_drone = {
            "current_location": departure.location,
            "current_launchpad_id": departure.launchpad_id,
            "battery_level": charge,
            "battery_capacity": battery.capacity_Wh,
            "empty_mass_kg": drone.empty_mass_kg,
            "max_payload_kg": drone.max_payload_kg,
            "max_range_m": drone.max_range_m,
        }
        ordered = self._task_return_launchpads(departure)
        for return_launchpad in ordered:
            if self.air_router.estimate_task_energy(task_drone, order, departure, return_launchpad)["is_feasible"]:
                return True
        return False

    def _task_return_launchpads(self, departure: Launchpad) -> list[Launchpad]:
        """Return targets considered by dispatch-time battery feasibility.

        The origin-return baseline is a strict round-trip policy, so its
        battery mask is based only on the departure pad. Other policies retain
        the full energy-feasible return action set.
        """
        return_policy = str(
            self.config.get("drone_decision", {}).get("drone_return_policy", "heuristic")
        )
        if return_policy == "origin":
            return [departure]
        ordered = [departure]
        ordered.extend(
            launchpad
            for launchpad in self.launchpads.values()
            if launchpad.launchpad_id != departure.launchpad_id
        )
        return ordered

    def _drone_can_reach_launchpad(self, drone: Drone, launchpad: Launchpad) -> bool:
        battery = self.batteries.get(drone.current_battery_id or "")
        return bool(
            battery is not None
            and battery.status == BatteryStatus.IN_USE
            and battery.current_vehicle_id == drone.vehicle_id
            and drone.battery_level is not None
            and self._battery_can_cover_empty_leg(
                drone,
                launchpad,
                drone.battery_level,
                battery.capacity_Wh,
            )
        )

    def _configure_launchpad_capacity(self) -> None:
        """Apply the runtime capacity knob to every loaded launchpad.

        ``unlimited`` is represented by ``None``. The instance files may still
        contain a historical ``drone_capacity`` field; the runtime knob is the
        experiment-level source of truth so sensitivity runs do not require
        regenerating instances.
        """
        raw = self._sim_cfg.get("launchpad_capacity", "unlimited")
        if raw is None or str(raw).strip().lower() in {"", "unlimited", "inf", "infinity"}:
            limit: int | None = None
        else:
            try:
                limit = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "simulation.launchpad_capacity must be 'unlimited' or a positive integer"
                ) from exc
            if limit <= 0:
                raise ValueError(
                    "simulation.launchpad_capacity must be 'unlimited' or a positive integer"
                )
        for launchpad in self.launchpads.values():
            launchpad.drone_capacity = limit
            # Reservations are runtime commitments, never inherited from an
            # instance snapshot at reset.
            launchpad.reserved_return_slots = 0
        self.config.setdefault("_runtime", {})["launchpad_capacity"] = (
            "unlimited" if limit is None else limit
        )

    def _capacity_is_finite(self) -> bool:
        return any(launchpad.drone_capacity is not None for launchpad in self.launchpads.values())

    def _return_target_has_capacity(
        self,
        launchpad: Launchpad,
    ) -> bool:
        limit = launchpad.drone_capacity
        if limit is None:
            return True
        used = len(launchpad.drone_ids) + launchpad.reserved_return_slots
        return used < limit

    @staticmethod
    def _reserve_return_target(launchpad: Launchpad) -> None:
        if launchpad.drone_capacity is not None:
            launchpad.reserved_return_slots += 1

    def _assert_capacity_invariant(self, phase: str) -> None:
        violations = []
        for launchpad in self.launchpads.values():
            if launchpad.drone_capacity is None:
                continue
            occupancy = len(launchpad.drone_ids) + launchpad.reserved_return_slots
            if occupancy > launchpad.drone_capacity:
                violations.append(
                    f"{launchpad.launchpad_id}: occupancy={occupancy} "
                    f"(docked={len(launchpad.drone_ids)}, "
                    f"reserved={launchpad.reserved_return_slots}) > "
                    f"capacity={launchpad.drone_capacity}"
                )
        if violations:
            self._raise_capacity_error(
                f"capacity invariant violated during {phase}: " + "; ".join(violations)
            )

    def _validate_capacity_action_batch(self, actions: list[TaskLevelActionDict]) -> None:
        """Check return/rebalance actions as one projected movement batch.

        The check is intentionally before any action is applied. A target's
        occupancy includes drones already docked and explicit return/rebalance
        commitments. A legacy dispatch only removes its drone from the
        departure pad; energy returnability alone does not reserve a future
        slot. A dispatch carrying target_launchpad_id is an explicit complete
        route and therefore projects both its departure and committed return.
        """
        if not self._capacity_is_finite() or not actions:
            return

        projected = {
            launchpad_id: len(launchpad.drone_ids) + launchpad.reserved_return_slots
            for launchpad_id, launchpad in self.launchpads.items()
        }
        projected_docked = {
            launchpad_id: set(launchpad.drone_ids)
            for launchpad_id, launchpad in self.launchpads.items()
        }
        details: dict[str, list[str]] = {}
        seen_drones: set[str] = set()
        seen_orders: set[str] = set()
        seen_batteries: set[str] = set()

        for action in actions:
            action_type = action.get("action_type")
            if action_type == "order_battery_dispatch":
                order_id = str(action.get("order_id", ""))
                battery_id = str(action.get("battery_id", ""))
                order = self.orders.get(order_id)
                battery = self.batteries.get(battery_id)
                if (
                    order is None
                    or battery is None
                    or order.status != OrderStatus.PENDING
                    or order_id in seen_orders
                    or battery_id in seen_batteries
                ):
                    continue
                departure = self.launchpads.get(order.origin_launchpad_id)
                if (
                    departure is None
                    or order_id not in departure.order_queue
                    or not self._can_select_launchpad_battery(battery, departure)
                ):
                    continue
                requested_drone_id = action.get("drone_id")
                candidate_drone_ids = (
                    {str(requested_drone_id)}
                    if requested_drone_id is not None
                    else projected_docked[departure.launchpad_id]
                )
                drone = self._available_drone_at_launchpad(
                    departure,
                    order,
                    candidate_drone_ids,
                )
                if drone is None:
                    continue
                energy_feasible = self._energy_feasible_task_return_launchpads(
                    drone,
                    order,
                    departure,
                    battery,
                )
                if not energy_feasible:
                    continue
                target_id_raw = action.get("target_launchpad_id")
                target_id = (
                    str(target_id_raw)
                    if target_id_raw is not None
                    else None
                )
                if target_id is not None and (
                    target_id not in self.launchpads
                    or target_id
                    not in {launchpad.launchpad_id for launchpad in energy_feasible}
                ):
                    continue

                projected_docked[departure.launchpad_id].remove(drone.vehicle_id)
                projected[departure.launchpad_id] -= 1
                if target_id is not None:
                    projected[target_id] = projected.get(target_id, 0) + 1
                    details.setdefault(target_id, []).append(
                        "preplanned_return:"
                        f"{drone.vehicle_id}->{target_id}:{action.get('action_id', '')}"
                    )
                seen_drones.add(drone.vehicle_id)
                seen_orders.add(order_id)
                seen_batteries.add(battery_id)
                continue

            if action_type not in {"drone_return", "drone_rebalance"}:
                continue
            drone_id = str(action.get("drone_id", ""))
            target_id = str(action.get("target_launchpad_id", ""))
            drone = self.drones.get(drone_id)
            target = self.launchpads.get(target_id)
            if drone is None or target is None:
                continue
            # Only the first movement for a drone can be accepted by the
            # executor; later duplicates are independently invalid.
            if drone_id in seen_drones:
                continue

            if action_type == "drone_return":
                if (
                    drone.status != VehicleStatus.IDLE
                    or drone.route_plan
                    or drone.onboard_order_ids
                    or drone.current_launchpad_id is not None
                    or not self._drone_can_reach_launchpad(drone, target)
                ):
                    continue
            else:
                battery_id = str(action.get("battery_id", ""))
                battery = self.batteries.get(battery_id)
                departure_id = str(drone.current_launchpad_id or "")
                if (
                    battery is None
                    or battery_id in seen_batteries
                    or not departure_id
                    or departure_id == target_id
                    or departure_id not in projected_docked
                    or drone_id not in projected_docked[departure_id]
                    or drone.status != VehicleStatus.IDLE
                    or drone.route_plan
                    or drone.onboard_order_ids
                    or drone.current_battery_id is not None
                ):
                    continue
                departure = self.launchpads[departure_id]
                if not self._can_select_launchpad_battery(battery, departure):
                    continue
                charge = self._estimated_battery_charge_at(battery, self.current_time)
                if not self._battery_can_cover_empty_leg(
                    drone,
                    target,
                    charge,
                    battery.capacity_Wh,
                ):
                    continue
                projected_docked[departure_id].remove(drone_id)
                projected[departure_id] -= 1
                seen_batteries.add(battery_id)

            projected[target_id] = projected.get(target_id, 0) + 1
            details.setdefault(target_id, []).append(
                f"{action_type}:{drone_id}->{target_id}:{action.get('action_id', '')}"
            )
            seen_drones.add(drone_id)

        violations = []
        for launchpad_id, occupancy in projected.items():
            launchpad = self.launchpads[launchpad_id]
            if launchpad.drone_capacity is not None and occupancy > launchpad.drone_capacity:
                violations.append(
                    f"{launchpad_id}: projected={occupancy}, "
                    f"capacity={launchpad.drone_capacity}, "
                    f"actions=[{', '.join(details.get(launchpad_id, []))}]"
                )
        if violations:
            self._raise_capacity_error(
                "capacity violation in one accepted action batch; no action was applied: "
                + "; ".join(violations)
            )

    def _raise_capacity_error(self, message: str) -> None:
        logger.error("Simulator capacity error: %s", message)
        raise RuntimeError(f"Simulator capacity error: {message}")

    def _raise_action_capacity_error(
        self,
        action: TaskLevelActionDict,
        drone: Drone,
        target: Launchpad,
    ) -> None:
        occupancy = len(target.drone_ids) + target.reserved_return_slots
        self._raise_capacity_error(
            f"action {action.get('action_id', '<unknown>')} "
            f"({action.get('action_type')}, drone={drone.vehicle_id}, "
            f"target={target.launchpad_id}) would exceed capacity "
            f"{target.drone_capacity}: occupancy={occupancy}"
        )

    def _accepted_preplanned_return_metrics(
        self,
        order: Order,
        departure: Launchpad,
        target: Launchpad,
        battery: Battery,
        dispatch_details: dict,
        return_plan: dict,
    ) -> dict:
        plans = dispatch_details.get("return_plans", {})
        feasible_plans = []
        for launchpad_id, plan in plans.items():
            launchpad = self.launchpads.get(str(launchpad_id))
            if launchpad is None:
                continue
            used_after_departure = (
                len(launchpad.drone_ids)
                + launchpad.reserved_return_slots
                - int(launchpad.launchpad_id == departure.launchpad_id)
            )
            if (
                launchpad.drone_capacity is None
                or used_after_departure < launchpad.drone_capacity
            ):
                feasible_plans.append(plan)
        selected_distance = float(return_plan["distance_return_m"])
        target_used_after_commitment = (
            len(target.drone_ids)
            + target.reserved_return_slots
            - int(target.launchpad_id == departure.launchpad_id)
            + 1
        )
        return {
            "feasible_launchpads": len(feasible_plans),
            "distance_m": selected_distance,
            "nearest_feasible_distance_m": min(
                (
                    float(plan["distance_return_m"])
                    for plan in feasible_plans
                ),
                default=selected_distance,
            ),
            "arrival_margin_Wh": float(return_plan["arrival_margin_Wh"]),
            "battery_capacity_Wh": battery.capacity_Wh,
            "returned_to_origin": target.launchpad_id == order.origin_launchpad_id,
            "capacity_blocked_launchpads": max(0, len(plans) - len(feasible_plans)),
            "target_occupancy_ratio": (
                target_used_after_commitment / max(1, target.drone_capacity)
                if target.drone_capacity is not None
                else 0.0
            ),
        }

    def _accepted_return_metrics(self, drone: Drone, target: Launchpad) -> dict:
        battery = self.batteries.get(drone.current_battery_id or "")
        if battery is None or drone.battery_level is None or self.air_router is None:
            raise RuntimeError(f"Accepted return for {drone.vehicle_id} has no mounted battery")

        energy_reachable = [
            launchpad
            for launchpad in self.launchpads.values()
            if self._drone_can_reach_launchpad(drone, launchpad)
        ]
        feasible = [
            launchpad
            for launchpad in energy_reachable
            if self._return_target_has_capacity(launchpad)
        ]
        distances = [
            self.air_router.travel_distance(drone.current_location, launchpad.location)
            for launchpad in feasible
        ]
        distance_m = self.air_router.travel_distance(drone.current_location, target.location)
        energy_Wh = self.air_router.energy_cost(
            drone.current_location,
            target.location,
            payload_mass_kg=0.0,
            empty_mass_kg=drone.empty_mass_kg,
        )
        arrival_margin_Wh = (
            float(drone.battery_level)
            - energy_Wh
            - self.air_router.reserve_energy(battery.capacity_Wh)
        )
        target_used = len(target.drone_ids) + target.reserved_return_slots
        delivered_order = next(
            (
                order
                for order in self.orders.values()
                if order.assigned_drone_id == drone.vehicle_id
                and order.status == OrderStatus.DELIVERED
                and order.return_launchpad_id is None
            ),
            None,
        )
        return {
            "feasible_launchpads": len(feasible),
            "distance_m": distance_m,
            "nearest_feasible_distance_m": min(distances, default=distance_m),
            "arrival_margin_Wh": arrival_margin_Wh,
            "battery_capacity_Wh": battery.capacity_Wh,
            "returned_to_origin": (
                target.launchpad_id == delivered_order.origin_launchpad_id
                if delivered_order is not None
                else None
            ),
            "capacity_blocked_launchpads": len(energy_reachable) - len(feasible),
            "target_occupancy_ratio": (
                target_used / max(1, target.drone_capacity)
                if target.drone_capacity is not None
                else 0.0
            ),
        }

    def _battery_can_cover_empty_leg(
        self,
        drone: Drone,
        destination: Launchpad,
        battery_level_Wh: float,
        battery_capacity_Wh: float,
    ) -> bool:
        if self.air_router is None:
            return False
        distance_m = self.air_router.travel_distance(drone.current_location, destination.location)
        if drone.max_range_m is not None and distance_m > drone.max_range_m:
            return False
        energy_Wh = self.air_router.energy_cost(
            drone.current_location,
            destination.location,
            payload_mass_kg=0.0,
            empty_mass_kg=drone.empty_mass_kg,
        )
        return battery_level_Wh - energy_Wh >= self.air_router.reserve_energy(battery_capacity_Wh)

    def _estimated_battery_charge_at(self, battery: Battery, at_time: int) -> float:
        if battery.status != BatteryStatus.CHARGING or battery.charge_complete_at is None:
            return float(battery.current_charge_Wh)
        if at_time >= battery.charge_complete_at:
            return float(battery.capacity_Wh)
        start_at = int(battery.charge_started_at or 0)
        start_charge = float(
            battery.charge_start_Wh
            if battery.charge_start_Wh is not None
            else battery.current_charge_Wh
        )
        if at_time <= start_at:
            return start_charge
        duration = max(1, battery.charge_complete_at - start_at)
        progress = (at_time - start_at) / duration
        return start_charge + progress * (battery.capacity_Wh - start_charge)

    def _release_orders(self, current_time: int) -> list[Order]:
        released: list[Order] = []
        while self._next_order_index < len(self._pending_orders_by_time):
            order = self._pending_orders_by_time[self._next_order_index]
            if order.created_at > current_time:
                break
            self._next_order_index += 1
            self.orders[order.order_id] = order
            launchpad = self.launchpads.get(order.origin_launchpad_id)
            if launchpad is None:
                raise RuntimeError(f"Order {order.order_id} has unknown origin launchpad {order.origin_launchpad_id}")
            if order.order_id not in launchpad.order_queue:
                launchpad.order_queue.append(order.order_id)
            self.metrics_recorder.log_order_created(current_time, self._step_count, order)
            released.append(order)
        return released

    def _normalize_action(self, action: dict[str, Any]) -> TaskLevelActionDict | None:
        if not isinstance(action, dict):
            return None
        action_type = str(action.get("action_type", ""))
        if action_type not in {"order_battery_dispatch", "drone_return", "drone_rebalance"}:
            return None
        normalized: TaskLevelActionDict = {
            "action_id": str(action.get("action_id") or f"{action_type}_{self.current_time}"),
            "action_type": action_type,
            "feasibility": bool(action.get("feasibility", True)),
        }
        for field in ("order_id", "battery_id", "drone_id", "target_launchpad_id"):
            if action.get(field) is not None:
                normalized[field] = str(action[field])
        return normalized

    def _refresh_state(self) -> None:
        stats = {
            "time": self.current_time,
            "total_orders": len(self.orders),
            "pending": sum(order.status == OrderStatus.PENDING for order in self.orders.values()),
            "delivered": sum(order.status == OrderStatus.DELIVERED for order in self.orders.values()),
            "idle_vehicles": sum(drone.status == VehicleStatus.IDLE for drone in self.drones.values()),
            "total_vehicles": len(self.drones),
        }
        self.state = self.state_builder.build(
            self.current_time,
            self.drones,
            self.orders,
            self.weather.get_state(),
            stats,
            self.launchpads,
            self.batteries,
        )

    def _empty_state(self) -> StateDict:
        return {
            "time": 0,
            "orders": {},
            "drones": {},
            "launchpads": {},
            "batteries": {},
            "weather": {},
            "stats": {
                "time": 0,
                "total_orders": 0,
                "pending": 0,
                "delivered": 0,
                "idle_vehicles": 0,
                "total_vehicles": 0,
            },
        }

    def _resolve_max_horizon(self, orders: list[Order]) -> int:
        configured = self._sim_cfg.get("max_horizon_seconds")
        if configured is not None:
            return max(int(configured), self.horizon)
        slack = int(self._sim_cfg.get("terminal_slack_seconds", 3600))
        latest_deadline = max((order.deadline for order in orders), default=self.horizon)
        return max(self.horizon, latest_deadline + max(0, slack))

    def _episode_done(self) -> tuple[bool, str | None]:
        if self.current_time >= self.max_horizon:
            return True, "max_horizon_reached"
        if self._all_orders_released() and self._all_orders_terminal() and self._all_drones_settled():
            return True, "all_orders_terminal"
        return False, None

    def _assert_episode_closed(self, reason: str | None) -> None:
        """Invariant at episode end: every order is DELIVERED or EXPIRED, i.e.
        completion_rate + expired_rate == 1. PENDING orders expire at their
        deadline and IN_TRANSIT orders always land, and max_horizon sits one
        hour past the last deadline, so any leftover is a real bug."""
        counts = {status: 0 for status in OrderStatus}
        for order in self.orders.values():
            counts[order.status] += 1
        generated = len(self.orders)
        delivered, expired = counts[OrderStatus.DELIVERED], counts[OrderStatus.EXPIRED]
        in_transit, pending = counts[OrderStatus.IN_TRANSIT], counts[OrderStatus.PENDING]
        where = f"episode ended ({reason}) at t={self.current_time}"
        assert in_transit == 0, f"{where}: {in_transit} orders still IN_TRANSIT (unfinished must be 0)"
        assert pending == 0, f"{where}: {pending} orders still PENDING (pending must be 0)"
        # delivered + expired == generated  <=>  completion_rate + expired_rate == 1
        assert delivered + expired == generated, (
            f"{where}: delivered + expired != generated ({delivered} + {expired} != {generated})"
        )

    def _all_orders_released(self) -> bool:
        return self._next_order_index >= len(self._pending_orders_by_time)

    def _all_orders_terminal(self) -> bool:
        terminal = {OrderStatus.DELIVERED, OrderStatus.EXPIRED}
        return all(order.status in terminal for order in self.orders.values())

    def _all_drones_settled(self) -> bool:
        return all(
            not drone.route_plan
            and not drone.onboard_order_ids
            and drone.current_launchpad_id is not None
            and drone.current_battery_id is None
            for drone in self.drones.values()
        )

    def _compute_step_reward(self, start_time: int, end_time: int) -> float:
        lateness_penalty_per_min = float(
            self.config["cost"]["lateness_penalty_per_min"]
        )
        reward = 0.0
        for order in self.orders.values():
            if order.delivery_time is None or not (start_time < order.delivery_time <= end_time):
                continue
            reward += order.value
            if order.delivery_time > order.deadline:
                reward -= (
                    (order.delivery_time - order.deadline) / 60.0
                    * lateness_penalty_per_min
                )
        return reward

    def _time_advancement_mode(self) -> str:
        return str(self._sim_cfg.get("time_advancement", self._sim_cfg.get("mode", "fixed_step"))).lower()

    def _apply_instance_weather(self, weather_config: dict) -> None:
        self.config["weather"] = dict(weather_config or {})
