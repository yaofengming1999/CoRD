from __future__ import annotations

import logging

from ..model.battery_types import BASE_MASS_KG as BASE_BATTERY_MASS_KG
from ..model.entities import Battery, Drone, Launchpad, Location, Order, Stop
from ..model.enums import BatteryStatus, LaunchpadStatus, OrderStatus, StopType, VehicleStatus
from ..routing.air_router import AirRouter
from ..utils.path import interpolate_along_polyline, polyline_distance_m

logger = logging.getLogger(__name__)


class TransitionEngine:
    """Advance drone-only delivery, return, and battery states."""

    def __init__(self, config: dict, air_router: AirRouter, metrics_recorder=None):
        self.config = config
        self.air_router = air_router
        self.metrics_recorder = metrics_recorder

    def advance(
        self,
        drones: dict[str, Drone],
        orders: dict[str, Order],
        current_time: int,
        delta_t: int,
        weather_factor: float | None = None,
        launchpads: dict[str, Launchpad] | None = None,
        batteries: dict[str, Battery] | None = None,
    ) -> None:
        launchpads = launchpads or {}
        batteries = batteries or {}
        interval_end = current_time + delta_t
        self._complete_charging_batteries(launchpads, batteries, current_time)
        for drone in drones.values():
            self._advance_drone(
                drone,
                orders,
                current_time,
                delta_t,
                weather_factor,
                launchpads,
                batteries,
            )
        self._complete_charging_batteries(launchpads, batteries, interval_end)

        for launchpad in launchpads.values():
            if (
                launchpad.order_queue
                or launchpad.drone_ids
                or launchpad.charging_battery_ids
            ):
                launchpad.active_time_s += delta_t
            launchpad.status = LaunchpadStatus.STATIONARY

        for order in orders.values():
            if order.status == OrderStatus.PENDING and interval_end >= order.deadline:
                order.status = OrderStatus.EXPIRED
                origin = launchpads.get(order.origin_launchpad_id)
                if origin is not None and order.order_id in origin.order_queue:
                    origin.order_queue.remove(order.order_id)

    def _advance_drone(
        self,
        drone: Drone,
        orders: dict[str, Order],
        current_time: int,
        delta_t: int,
        weather_factor: float | None,
        launchpads: dict[str, Launchpad],
        batteries: dict[str, Battery],
    ) -> None:
        interval_end = current_time + delta_t
        while True:
            if not drone.route_plan:
                self._clear_current_leg(drone)
                drone.status = VehicleStatus.WAITING if drone.available_at > interval_end else VehicleStatus.IDLE
                return

            stop = drone.route_plan[0]
            if not self._current_leg_matches(drone, stop):
                self._start_current_leg(drone, stop, orders, batteries, current_time, weather_factor)

            if drone.current_leg_started_at is not None and interval_end < drone.current_leg_started_at:
                drone.current_location = drone.current_leg_origin or drone.current_location
                drone.status = VehicleStatus.WAITING
                return
            if drone.current_stop_eta is None:
                return
            if interval_end < drone.current_stop_eta:
                drone.status = self._transit_status(stop)
                self._update_location_along_current_leg(drone, interval_end)
                return

            arrival_time = drone.current_stop_eta
            self._update_location_along_current_leg(drone, arrival_time)
            drone.current_location = stop.location
            self._execute_stop(
                drone,
                stop,
                orders,
                arrival_time,
                launchpads,
                batteries,
                weather_factor,
            )
            drone.route_plan.pop(0)
            service_delay = int(stop.service_time_seconds or 0)
            self._clear_current_leg(drone)

            if drone.route_plan:
                depart_time = arrival_time + service_delay
                next_stop = drone.route_plan[0]
                self._start_current_leg(drone, next_stop, orders, batteries, depart_time, weather_factor)
                if depart_time > interval_end:
                    drone.status = VehicleStatus.WAITING
                    return
                drone.status = self._transit_status(next_stop)
                continue

            drone.available_at = arrival_time + service_delay
            drone.status = VehicleStatus.WAITING if drone.available_at > interval_end else VehicleStatus.IDLE
            return

    def _execute_stop(
        self,
        drone: Drone,
        stop: Stop,
        orders: dict[str, Order],
        current_time: int,
        launchpads: dict[str, Launchpad],
        batteries: dict[str, Battery],
        weather_factor: float | None,
    ) -> None:
        origin = drone.current_leg_origin or drone.current_location
        payload_kg = self._onboard_payload_kg(drone, orders)
        energy_Wh = self.air_router.energy_cost(
            origin,
            stop.location,
            payload_mass_kg=payload_kg,
            empty_mass_kg=self._effective_empty_mass_kg(drone, batteries),
            weather_factor=weather_factor,
        )
        if drone.battery_level is None:
            raise RuntimeError(f"UAV {drone.vehicle_id} completed a flight leg without a battery")
        remaining_Wh = float(drone.battery_level) - energy_Wh
        if remaining_Wh < -1e-6:
            raise RuntimeError(
                f"UAV {drone.vehicle_id} completed leg {stop.stop_id} with insufficient battery: "
                f"needed {energy_Wh:.6f} Wh, had {drone.battery_level:.6f} Wh"
            )
        drone.battery_level = max(0.0, remaining_Wh)
        drone.total_distance_m += drone.current_leg_distance_m
        drone.total_energy += energy_Wh
        mounted = batteries.get(drone.current_battery_id or "")
        if mounted is not None:
            # Per-battery discharge ledger: divided by capacity this is the
            # physically meaningful full-cycle-equivalent (FCE) count.
            mounted.total_discharged_Wh += energy_Wh
        self._sync_vehicle_battery_charge(drone, batteries)

        if stop.stop_type == StopType.DELIVERY:
            order = orders.get(stop.related_order_id or "")
            if order is None or order.status != OrderStatus.IN_TRANSIT:
                raise RuntimeError(f"Delivery stop {stop.stop_id} has no in-transit order")
            order.status = OrderStatus.DELIVERED
            order.delivery_time = current_time
            if order.order_id in drone.onboard_order_ids:
                drone.onboard_order_ids.remove(order.order_id)
                drone.remaining_capacity += order.size
            if self.metrics_recorder is not None:
                self.metrics_recorder.log_drone_event("delivery", current_time, drone, order=order)
            return

        if stop.stop_type != StopType.DRONE_RETURN:
            raise RuntimeError(f"Unsupported drone stop type: {stop.stop_type.value}")
        launchpad = launchpads.get(stop.related_launchpad_id or "")
        if launchpad is None:
            raise RuntimeError(f"Return stop {stop.stop_id} references an unknown launchpad")
        if launchpad.reserved_return_slots > 0:
            launchpad.reserved_return_slots -= 1
        if (
            launchpad.drone_capacity is not None
            and len(launchpad.drone_ids) >= launchpad.drone_capacity
        ):
            message = (
                "Return launchpad "
                f"{launchpad.launchpad_id} is full on arrival: "
                f"docked={len(launchpad.drone_ids)}, "
                f"reserved={launchpad.reserved_return_slots}, "
                f"capacity={launchpad.drone_capacity}"
            )
            logger.error("Simulator capacity error: %s", message)
            raise RuntimeError(f"Simulator capacity error: {message}")
        if drone.vehicle_id not in launchpad.drone_ids:
            launchpad.drone_ids.append(drone.vehicle_id)
        drone.current_launchpad_id = launchpad.launchpad_id
        launchpad.service_events += 1
        self._dock_current_battery_for_charging(drone, launchpad, batteries, current_time)
        if self.metrics_recorder is not None:
            self.metrics_recorder.log_drone_event("drone_return", current_time, drone, launchpad=launchpad)

    def _start_current_leg(
        self,
        drone: Drone,
        stop: Stop,
        orders: dict[str, Order],
        batteries: dict[str, Battery],
        start_time: int,
        weather_factor: float | None,
    ) -> None:
        origin = drone.current_location
        distance_m = self.air_router.travel_distance(origin, stop.location)
        self._validate_uav_leg_energy(
            drone,
            stop,
            orders,
            batteries,
            origin,
            distance_m,
            weather_factor,
        )
        travel_time_s = self.air_router.travel_time(origin, stop.location, weather_factor)
        path = [origin, stop.location]
        drone.current_leg_started_at = int(start_time)
        drone.current_leg_target_stop_id = stop.stop_id
        drone.current_leg_origin = origin
        drone.current_leg_path = path
        drone.current_leg_distance_m = float(distance_m)
        drone.current_leg_path_distance_m = polyline_distance_m(path)
        drone.current_leg_travel_time_s = float(travel_time_s)
        drone.current_leg_arrived = distance_m <= 1e-6
        drone.current_stop_eta = int(start_time) + max(1, int(travel_time_s))

    def _validate_uav_leg_energy(
        self,
        drone: Drone,
        stop: Stop,
        orders: dict[str, Order],
        batteries: dict[str, Battery],
        origin: Location,
        distance_m: float,
        weather_factor: float | None,
    ) -> None:
        if drone.max_range_m is not None and distance_m > float(drone.max_range_m) + 1e-6:
            raise RuntimeError(
                f"UAV {drone.vehicle_id} cannot start leg {stop.stop_id}: "
                f"distance {distance_m:.3f} m exceeds max range {float(drone.max_range_m):.3f} m"
            )
        battery = batteries.get(drone.current_battery_id or "")
        if (
            battery is None
            or battery.status != BatteryStatus.IN_USE
            or battery.current_vehicle_id != drone.vehicle_id
            or drone.battery_level is None
        ):
            raise RuntimeError(
                f"UAV {drone.vehicle_id} cannot start leg {stop.stop_id} without a valid mounted battery"
            )
        needed_Wh = self.air_router.energy_cost(
            origin,
            stop.location,
            payload_mass_kg=self._onboard_payload_kg(drone, orders),
            empty_mass_kg=self._effective_empty_mass_kg(drone, batteries),
            weather_factor=weather_factor,
        )
        if needed_Wh > float(drone.battery_level) + 1e-6:
            raise RuntimeError(
                f"UAV {drone.vehicle_id} cannot start leg {stop.stop_id}: "
                f"needs {needed_Wh:.6f} Wh, has {float(drone.battery_level):.6f} Wh"
            )

    def _update_location_along_current_leg(self, drone: Drone, current_time: int) -> None:
        if drone.current_leg_started_at is None or drone.current_stop_eta is None or not drone.current_leg_path:
            return
        if current_time <= drone.current_leg_started_at:
            drone.current_location = drone.current_leg_path[0]
            return
        if current_time >= drone.current_stop_eta:
            drone.current_location = drone.current_leg_path[-1]
            drone.current_leg_arrived = True
            return
        duration = max(1, drone.current_stop_eta - drone.current_leg_started_at)
        progress = (current_time - drone.current_leg_started_at) / duration
        drone.current_location = interpolate_along_polyline(
            drone.current_leg_path,
            max(0.0, min(1.0, progress)) * drone.current_leg_path_distance_m,
        )

    def _current_leg_matches(self, drone: Drone, stop: Stop) -> bool:
        return drone.current_leg_target_stop_id == stop.stop_id and drone.current_stop_eta is not None

    def _clear_current_leg(self, drone: Drone) -> None:
        drone.current_stop_eta = None
        drone.current_leg_started_at = None
        drone.current_leg_target_stop_id = None
        drone.current_leg_origin = None
        drone.current_leg_path = []
        drone.current_leg_distance_m = 0.0
        drone.current_leg_path_distance_m = 0.0
        drone.current_leg_travel_time_s = 0.0
        drone.current_leg_arrived = False

    def _onboard_payload_kg(self, drone: Drone, orders: dict[str, Order]) -> float:
        return sum(float(orders[oid].weight or 0.0) for oid in drone.onboard_order_ids if oid in orders)

    def _transit_status(self, stop: Stop) -> VehicleStatus:
        return VehicleStatus.FLYING_LOADED if stop.stop_type == StopType.DELIVERY else VehicleStatus.FLYING_EMPTY

    def _sync_vehicle_battery_charge(self, drone: Drone, batteries: dict[str, Battery]) -> None:
        battery = batteries.get(drone.current_battery_id or "")
        if battery is not None and drone.battery_level is not None:
            battery.current_charge_Wh = drone.battery_level

    def _dock_current_battery_for_charging(
        self,
        drone: Drone,
        launchpad: Launchpad,
        batteries: dict[str, Battery],
        current_time: int,
    ) -> None:
        battery = batteries.get(drone.current_battery_id or "")
        if battery is None or drone.battery_level is None:
            raise RuntimeError(f"UAV {drone.vehicle_id} landed without a mounted battery")
        battery.current_charge_Wh = float(drone.battery_level)
        self._start_battery_charging(battery, launchpad, current_time)
        if battery.battery_id not in launchpad.charging_battery_ids:
            launchpad.charging_battery_ids.append(battery.battery_id)
        if battery.battery_id in launchpad.charged_battery_ids:
            launchpad.charged_battery_ids.remove(battery.battery_id)
        drone.current_battery_id = None
        drone.battery_level = None

    def _start_battery_charging(self, battery: Battery, launchpad: Launchpad, current_time: int) -> None:
        battery.status = BatteryStatus.CHARGING
        battery.current_vehicle_id = None
        battery.current_launchpad_id = launchpad.launchpad_id
        battery.current_charge_Wh = max(0.0, min(battery.capacity_Wh, battery.current_charge_Wh))
        battery.charge_started_at = current_time
        battery.charge_start_Wh = battery.current_charge_Wh
        battery.charge_complete_at = self._charge_complete_time(battery, current_time)

    def _charge_complete_time(self, battery: Battery, current_time: int) -> int:
        capacity = max(1e-9, float(battery.capacity_Wh))
        deficit_ratio = max(0.0, min(1.0, (capacity - battery.current_charge_Wh) / capacity))
        if deficit_ratio <= 0.0:
            return current_time
        return current_time + max(1, int(self._full_charge_seconds(battery) * deficit_ratio))

    def _effective_empty_mass_kg(self, drone: Drone, batteries: dict[str, Battery]) -> float | None:
        """The drone's empty mass with the pack it is actually carrying.

        ``drone.empty_mass_kg`` is the airframe plus a *baseline* pack, because
        that is how the power intercept was calibrated. Swapping in a pack of a
        different mass moves the vehicle's empty mass by the difference, and the
        energy model charges that difference at the payload slope.

        Returns the drone's own figure unchanged when no pack is mounted or the
        mounted pack declares no mass, so a homogeneous instance is unaffected.
        """
        if drone.empty_mass_kg is None:
            return None
        battery = batteries.get(drone.current_battery_id or "")
        pack_mass = None if battery is None else getattr(battery, "mass_kg", None)
        if pack_mass is None:
            return float(drone.empty_mass_kg)
        return float(drone.empty_mass_kg) - self._baseline_battery_mass_kg() + float(pack_mass)

    def _baseline_battery_mass_kg(self) -> float:
        spec = (self.config.get("batteries", {}) or {}).get("heterogeneity", {}) or {}
        return float(spec.get("base_mass_kg", BASE_BATTERY_MASS_KG))

    def _full_charge_seconds(self, battery: Battery) -> int:
        """Empty-to-full time for this pack.

        A heterogeneous pack carries its own, set from its capacity at a fixed
        charger power. A pack from an instance written before heterogeneity
        existed carries None and falls back to the scenario-wide constant, which
        is what every homogeneous instance used.
        """
        own = getattr(battery, "full_charge_seconds", None)
        return max(1, int(own)) if own else self._battery_charge_time_seconds()

    def _complete_charging_batteries(
        self,
        launchpads: dict[str, Launchpad],
        batteries: dict[str, Battery],
        current_time: int,
    ) -> None:
        for launchpad in launchpads.values():
            for battery_id in list(launchpad.charging_battery_ids):
                battery = batteries.get(battery_id)
                if battery is None or battery.charge_complete_at is None:
                    continue
                self._update_charging_battery_charge(battery, current_time)
                if battery.charge_complete_at > current_time:
                    continue
                battery.status = BatteryStatus.CHARGED
                battery.current_charge_Wh = battery.capacity_Wh
                battery.charge_started_at = None
                battery.charge_start_Wh = None
                battery.charge_complete_at = None
                battery.current_vehicle_id = None
                battery.current_launchpad_id = launchpad.launchpad_id
                launchpad.charging_battery_ids.remove(battery_id)
                if battery_id not in launchpad.charged_battery_ids:
                    launchpad.charged_battery_ids.append(battery_id)

    def _update_charging_battery_charge(self, battery: Battery, current_time: int) -> float:
        if battery.status != BatteryStatus.CHARGING or battery.charge_complete_at is None:
            return battery.current_charge_Wh
        start_at = int(battery.charge_started_at or 0)
        start_charge = float(
            battery.charge_start_Wh
            if battery.charge_start_Wh is not None
            else battery.current_charge_Wh
        )
        if current_time <= start_at:
            return battery.current_charge_Wh
        if current_time >= battery.charge_complete_at:
            battery.current_charge_Wh = battery.capacity_Wh
            return battery.current_charge_Wh
        duration = max(1, battery.charge_complete_at - start_at)
        progress = (current_time - start_at) / duration
        battery.current_charge_Wh = start_charge + progress * (battery.capacity_Wh - start_charge)
        return battery.current_charge_Wh

    def _battery_charge_time_seconds(self) -> int:
        service_times = self.config.get("service_times", {})
        launchpad_service = self.config.get("launchpads", {}).get("service", {})
        return max(
            1,
            int(service_times.get("battery_charge_seconds", launchpad_service.get("battery_charge_seconds", 1800))),
        )
