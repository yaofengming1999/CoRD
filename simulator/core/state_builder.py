from __future__ import annotations

from ..model.entities import Battery, Drone, Launchpad, Order
from ..model.types import BatteryStateDict, LaunchpadStateDict, OrderStateDict, StateDict, VehicleStateDict


class StateBuilder:
    """Build the public drone-only simulator state."""

    def build(
        self,
        current_time: int,
        drones: dict[str, Drone],
        orders: dict[str, Order],
        weather_state: dict,
        step_stats: dict,
        launchpads: dict[str, Launchpad] | None = None,
        batteries: dict[str, Battery] | None = None,
    ) -> StateDict:
        return {
            "time": current_time,
            "orders": {oid: self._order_state(order) for oid, order in orders.items()},
            "drones": {
                drone_id: self._drone_state(drone, current_time)
                for drone_id, drone in drones.items()
            },
            "launchpads": {
                launchpad_id: self._launchpad_state(launchpad)
                for launchpad_id, launchpad in (launchpads or {}).items()
            },
            "batteries": {
                battery_id: self._battery_state(battery)
                for battery_id, battery in (batteries or {}).items()
            },
            "weather": weather_state,
            "stats": step_stats,
        }

    def _order_state(self, order: Order) -> OrderStateDict:
        return {
            "order_id": order.order_id,
            "created_at": order.created_at,
            "origin_launchpad_id": order.origin_launchpad_id,
            "destination": {"lat": order.destination.lat, "lon": order.destination.lon},
            "destination_endpoint_id": order.destination_endpoint_id,
            "size": order.size,
            "weight": order.weight,
            "value": order.value,
            "deadline": order.deadline,
            "status": order.status.value,
            "assigned_drone_id": order.assigned_drone_id,
            "selected_battery_id": order.selected_battery_id,
            "return_launchpad_id": order.return_launchpad_id,
            "base_revenue": order.base_revenue,
            "distance_revenue": order.distance_revenue,
            "weight_revenue": order.weight_revenue,
            "dispatch_time": order.dispatch_time,
            "delivery_time": order.delivery_time,
        }

    def _drone_state(self, drone: Drone, current_time: int) -> VehicleStateDict:
        end_location = drone.route_plan[-1].location if drone.route_plan else drone.current_location
        return {
            "vehicle_id": drone.vehicle_id,
            "vehicle_type": drone.vehicle_type.value,
            "current_location": {"lat": drone.current_location.lat, "lon": drone.current_location.lon},
            "available_at": drone.available_at,
            "capacity": drone.capacity,
            "remaining_capacity": drone.remaining_capacity,
            "speed_mps": drone.speed_mps,
            "battery_level": drone.battery_level,
            "battery_capacity": drone.battery_capacity,
            "max_range_m": drone.max_range_m,
            "empty_mass_kg": drone.empty_mass_kg,
            "max_payload_kg": drone.max_payload_kg,
            "current_battery_id": drone.current_battery_id,
            "current_launchpad_id": drone.current_launchpad_id,
            "route_plan": [self._stop_state(stop) for stop in drone.route_plan],
            "onboard_order_ids": list(drone.onboard_order_ids),
            "route_plan_end_location": {"lat": end_location.lat, "lon": end_location.lon},
            "status": drone.status.value,
            "current_stop_eta": drone.current_stop_eta,
            "current_leg_started_at": drone.current_leg_started_at,
            "current_leg_target_stop_id": drone.current_leg_target_stop_id,
            "current_leg_progress_ratio": self._leg_progress_ratio(drone, current_time),
            "current_leg_distance_m": drone.current_leg_distance_m,
            "current_leg_travel_time_s": drone.current_leg_travel_time_s,
            "total_distance_m": drone.total_distance_m,
            "total_energy": drone.total_energy,
        }

    def _leg_progress_ratio(self, drone: Drone, current_time: int) -> float:
        if drone.current_leg_arrived:
            return 1.0
        if drone.current_leg_started_at is None or drone.current_stop_eta is None:
            return 0.0
        duration = max(1, drone.current_stop_eta - drone.current_leg_started_at)
        return max(0.0, min(1.0, (current_time - drone.current_leg_started_at) / duration))

    def _launchpad_state(self, launchpad: Launchpad) -> LaunchpadStateDict:
        return {
            "launchpad_id": launchpad.launchpad_id,
            "launchpad_type": "stationary",
            "current_location": {"lat": launchpad.current_location.lat, "lon": launchpad.current_location.lon},
            "drone_capacity": launchpad.drone_capacity,
            "order_queue": list(launchpad.order_queue),
            "drone_ids": list(launchpad.drone_ids),
            "charged_battery_ids": list(launchpad.charged_battery_ids),
            "charging_battery_ids": list(launchpad.charging_battery_ids),
            "charged_batteries": len(launchpad.charged_battery_ids),
            "charging_batteries": len(launchpad.charging_battery_ids),
            "status": launchpad.status.value,
            "reserved_return_slots": launchpad.reserved_return_slots,
            "deployment_cost": launchpad.deployment_cost,
            "op_cost_per_second": launchpad.op_cost_per_second,
            "service_cost_per_event": launchpad.service_cost_per_event,
            "active_time_s": launchpad.active_time_s,
            "service_events": launchpad.service_events,
        }

    def _battery_state(self, battery: Battery) -> BatteryStateDict:
        return {
            "battery_id": battery.battery_id,
            "capacity_Wh": battery.capacity_Wh,
            "current_charge_Wh": battery.current_charge_Wh,
            "status": battery.status.value,
            "current_launchpad_id": battery.current_launchpad_id,
            "current_vehicle_id": battery.current_vehicle_id,
            "charge_started_at": battery.charge_started_at,
            "charge_start_Wh": battery.charge_start_Wh,
            "charge_complete_at": battery.charge_complete_at,
            "total_discharged_Wh": battery.total_discharged_Wh,
        }

    def _stop_state(self, stop) -> dict:
        return {
            "stop_id": stop.stop_id,
            "location": {"lat": stop.location.lat, "lon": stop.location.lon},
            "stop_type": stop.stop_type.value,
            "related_order_id": stop.related_order_id,
            "related_launchpad_id": stop.related_launchpad_id,
        }
