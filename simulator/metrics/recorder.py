from __future__ import annotations

import json
from pathlib import Path

from ..model.entities import Battery, Drone, Launchpad, Location, Order, Stop
from ..model.enums import OrderStatus, VehicleStatus
from .trajectory_points import add_interpolated_points_to_trajectories, vehicle_leg_snapshot


class MetricsRecorder:
    """Record drone-only frames, decisions, and lifecycle events."""

    def __init__(self, record_frames: bool = True) -> None:
        # Per-step frame snapshots + save-time trajectory interpolation drive
        # only the HTML replay; they cost the majority of episode wall-time and
        # are pure waste during training/eval. Off => record_frame is a no-op
        # and there is nothing to interpolate at save time.
        self.record_frames = record_frames
        self._step_log: list[dict] = []
        self._events: list[dict] = []
        self._episode_frames: list[dict] = []
        self._actions: list[dict] = []
        self._battery_dispatch_metrics: list[dict[str, float]] = []
        self._return_metrics: list[dict[str, float]] = []
        self._rebalance_metrics: list[dict[str, float]] = []

    def reset(self) -> None:
        record_frames = getattr(self, "record_frames", True)
        self.__init__(record_frames=record_frames)

    def log_step(
        self,
        current_time: int,
        drones: dict[str, Drone],
        orders: dict[str, Order],
        launchpads: dict[str, Launchpad] | None = None,
        batteries: dict[str, Battery] | None = None,
        weather_state: dict | None = None,
        interval_end: int | None = None,
    ) -> None:
        self._step_log.append(self.step_stats(current_time, drones, orders))

    def record_frame(
        self,
        current_time: int,
        drones: dict[str, Drone],
        orders: dict[str, Order],
        launchpads: dict[str, Launchpad] | None = None,
        batteries: dict[str, Battery] | None = None,
        weather_state: dict | None = None,
        stats: dict | None = None,
        frame_kind: str = "step",
    ) -> None:
        if not self.record_frames:
            return
        self._episode_frames.append({
            "frame_index": len(self._episode_frames),
            "frame_kind": frame_kind,
            "time": current_time,
            "drones": {
                drone_id: self._drone_to_dict(drone, current_time)
                for drone_id, drone in drones.items()
            },
            "orders": {order_id: self._order_to_dict(order) for order_id, order in orders.items()},
            "launchpads": {
                launchpad_id: self._launchpad_to_dict(launchpad)
                for launchpad_id, launchpad in (launchpads or {}).items()
            },
            "batteries": {
                battery_id: self._battery_to_dict(battery)
                for battery_id, battery in (batteries or {}).items()
            },
            "weather": dict(weather_state or {}),
            "stats": dict(stats or self.step_stats(current_time, drones, orders)),
        })

    def log_actions(self, current_time: int, step: int, actions: list[dict]) -> None:
        self._actions.append({
            "time": current_time,
            "step": step,
            "actions": [dict(action) for action in actions],
        })

    def log_battery_dispatch_metrics(
        self,
        *,
        selected_charge_Wh: float,
        battery_capacity_Wh: float,
        minimum_required_Wh: float,
        feasible_return_launchpads: int,
        delivery_energy_Wh: float = 0.0,
        return_energy_Wh: float = 0.0,
    ) -> None:
        """Slack of the battery chosen for one order, over each of the two flights.

        Two margins are recorded, both against the same reserve:

          roundtrip_margin = charge - E_delivery - E_return - reserve
          delivery_margin  = charge - E_delivery             - reserve

        so ``roundtrip_margin`` is what is left after flying the order out AND
        flying the drone back to a launchpad, while ``delivery_margin`` stops at the
        customer. Their difference is ``return_cost``, the share of the battery the
        flight back claims. The two are kept apart because a policy that does not
        choose the return pad is only ever pushed against the second number.

        ``delivery_margin`` is comparable across every policy: the loaded leg does
        not depend on which pad the drone goes back to, and the reserve is a fixed
        share of capacity, so it does not matter whether the caller passed the
        committed return plan or the energy-minimal one. ``roundtrip_margin`` and
        ``return_cost`` do depend on that choice, and the caller documents it.
        """
        capacity = max(1e-9, float(battery_capacity_Wh))
        charge = float(selected_charge_Wh)
        minimum = float(minimum_required_Wh)
        margin = charge - minimum
        delivery = float(delivery_energy_Wh)
        returning = float(return_energy_Wh)
        # minimum_required = E_delivery + E_return + reserve, and only the reserve
        # carries over to the delivery-only margin
        reserve = max(0.0, minimum - delivery - returning)
        delivery_margin = charge - delivery - reserve
        self._battery_dispatch_metrics.append({
            "selected_charge_Wh": charge,
            "selected_soc": charge / capacity,
            "minimum_required_Wh": minimum,
            "roundtrip_margin_Wh": margin,
            "roundtrip_margin_ratio": margin / capacity,
            "delivery_energy_Wh": delivery,
            "delivery_margin_Wh": delivery_margin,
            "delivery_margin_ratio": delivery_margin / capacity,
            "return_cost_Wh": returning,
            "return_cost_ratio": returning / capacity,
            "feasible_return_launchpads": float(feasible_return_launchpads),
        })

    def log_return_metrics(
        self,
        *,
        feasible_launchpads: int,
        distance_m: float,
        nearest_feasible_distance_m: float,
        arrival_margin_Wh: float,
        battery_capacity_Wh: float,
        returned_to_origin: bool | None,
        capacity_blocked_launchpads: int = 0,
        target_occupancy_ratio: float = 0.0,
    ) -> None:
        capacity = max(1e-9, float(battery_capacity_Wh))
        row = {
            "feasible_launchpads": float(feasible_launchpads),
            "distance_m": float(distance_m),
            "nearest_feasible_distance_m": float(nearest_feasible_distance_m),
            "extra_distance_vs_nearest_m": max(
                0.0, float(distance_m) - float(nearest_feasible_distance_m)
            ),
            "arrival_margin_Wh": float(arrival_margin_Wh),
            "arrival_margin_ratio": float(arrival_margin_Wh) / capacity,
            "capacity_blocked_launchpads": float(capacity_blocked_launchpads),
            "target_occupancy_ratio": float(target_occupancy_ratio),
        }
        if returned_to_origin is not None:
            row["returned_to_origin"] = float(returned_to_origin)
        self._return_metrics.append(row)

    def log_rebalance_metrics(
        self,
        *,
        selected_charge_Wh: float,
        battery_capacity_Wh: float,
        distance_m: float,
        energy_Wh: float,
        arrival_margin_Wh: float,
    ) -> None:
        capacity = max(1e-9, float(battery_capacity_Wh))
        charge = float(selected_charge_Wh)
        self._rebalance_metrics.append({
            "selected_charge_Wh": charge,
            "selected_soc": charge / capacity,
            "distance_m": float(distance_m),
            "energy_Wh": float(energy_Wh),
            "arrival_margin_Wh": float(arrival_margin_Wh),
            "arrival_margin_ratio": float(arrival_margin_Wh) / capacity,
        })

    def operational_summary(self) -> dict[str, float]:
        dispatch = self._battery_dispatch_metrics
        returns = self._return_metrics
        rebalances = self._rebalance_metrics
        return {
            "battery_dispatches": float(len(dispatch)),
            "battery_avg_selected_charge_Wh": self._average(dispatch, "selected_charge_Wh"),
            "battery_avg_selected_soc": self._average(dispatch, "selected_soc"),
            "battery_avg_minimum_required_Wh": self._average(dispatch, "minimum_required_Wh"),
            "battery_avg_roundtrip_margin_Wh": self._average(dispatch, "roundtrip_margin_Wh"),
            "battery_avg_roundtrip_margin_ratio": self._average(dispatch, "roundtrip_margin_ratio"),
            "battery_avg_delivery_margin_Wh": self._average(dispatch, "delivery_margin_Wh"),
            "battery_avg_delivery_margin_ratio": self._average(dispatch, "delivery_margin_ratio"),
            "battery_avg_return_cost_ratio": self._average(dispatch, "return_cost_ratio"),
            "battery_avg_feasible_return_launchpads": self._average(
                dispatch, "feasible_return_launchpads"
            ),
            "return_decisions": float(len(returns)),
            "return_avg_feasible_launchpads": self._average(returns, "feasible_launchpads"),
            "return_avg_distance_m": self._average(returns, "distance_m"),
            "return_avg_nearest_feasible_distance_m": self._average(
                returns, "nearest_feasible_distance_m"
            ),
            "return_avg_extra_distance_vs_nearest_m": self._average(
                returns, "extra_distance_vs_nearest_m"
            ),
            "return_avg_arrival_margin_Wh": self._average(returns, "arrival_margin_Wh"),
            "return_avg_arrival_margin_ratio": self._average(returns, "arrival_margin_ratio"),
            "return_origin_rate": self._average(returns, "returned_to_origin"),
            "return_avg_capacity_blocked_launchpads": self._average(
                returns, "capacity_blocked_launchpads"
            ),
            "return_avg_target_occupancy_ratio": self._average(
                returns, "target_occupancy_ratio"
            ),
            "drone_rebalances": float(len(rebalances)),
            "rebalance_total_distance_m": self._sum(rebalances, "distance_m"),
            "rebalance_total_energy_Wh": self._sum(rebalances, "energy_Wh"),
            "rebalance_avg_selected_charge_Wh": self._average(
                rebalances, "selected_charge_Wh"
            ),
            "rebalance_avg_selected_soc": self._average(rebalances, "selected_soc"),
            "rebalance_avg_arrival_margin_Wh": self._average(rebalances, "arrival_margin_Wh"),
            "rebalance_avg_arrival_margin_ratio": self._average(
                rebalances, "arrival_margin_ratio"
            ),
        }

    @staticmethod
    def _average(rows: list[dict[str, float]], key: str) -> float:
        values = [float(row[key]) for row in rows if key in row]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _sum(rows: list[dict[str, float]], key: str) -> float:
        return sum(float(row.get(key, 0.0)) for row in rows)

    def log_order_created(self, current_time: int, step: int, order: Order) -> None:
        self._events.append({
            "event_type": "order_created",
            "time": current_time,
            "step": step,
            "order_id": order.order_id,
            "order": self._order_to_dict(order),
        })

    def log_drone_event(
        self,
        event_type: str,
        current_time: int,
        drone: Drone,
        *,
        order: Order | None = None,
        launchpad: Launchpad | None = None,
    ) -> None:
        self._events.append({
            "event_type": event_type,
            "time": current_time,
            "drone_id": drone.vehicle_id,
            "order_id": order.order_id if order is not None else None,
            "launchpad_id": launchpad.launchpad_id if launchpad is not None else None,
            "location": self._location_to_dict(drone.current_location),
            "drone": self._drone_to_dict(drone, current_time),
            "order": self._order_to_dict(order) if order is not None else None,
        })

    def step_stats(self, current_time: int, drones: dict[str, Drone], orders: dict[str, Order]) -> dict:
        return {
            "time": current_time,
            "total_orders": len(orders),
            "pending": sum(order.status == OrderStatus.PENDING for order in orders.values()),
            "delivered": sum(order.status == OrderStatus.DELIVERED for order in orders.values()),
            "idle_vehicles": sum(drone.status == VehicleStatus.IDLE for drone in drones.values()),
            "total_vehicles": len(drones),
        }

    def summarize_episode(self) -> dict:
        last = self._step_log[-1] if self._step_log else {}
        total = int(last.get("total_orders", 0))
        delivered = int(last.get("delivered", 0))
        return {
            "total_orders_generated": total,
            "total_delivered": delivered,
            "completion_rate": delivered / total if total else 0.0,
            "num_steps": len(self._step_log),
        }

    def step_log(self) -> list[dict]:
        return list(self._step_log)

    def events(self) -> list[dict]:
        return list(self._events)

    def episode_frames(self) -> list[dict]:
        return list(self._episode_frames)

    def actions(self) -> list[dict]:
        return list(self._actions)

    def save_recording(self, output_dir: str | Path, metadata: dict | None = None) -> dict[str, str]:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        files = {
            "metadata": path / "metadata.json",
            "episode_frames": path / "episode_frames.jsonl",
            "actions": path / "actions.json",
            "events": path / "events.json",
            "trajectories": path / "trajectories.json",
        }
        with files["metadata"].open("w", encoding="utf-8") as handle:
            json.dump(metadata or {}, handle, ensure_ascii=False, indent=2)
        with files["episode_frames"].open("w", encoding="utf-8") as handle:
            for frame in self._episode_frames:
                handle.write(json.dumps(frame, ensure_ascii=False) + "\n")
        with files["actions"].open("w", encoding="utf-8") as handle:
            json.dump(self._actions, handle, ensure_ascii=False, indent=2)
        with files["events"].open("w", encoding="utf-8") as handle:
            json.dump(self._events, handle, ensure_ascii=False, indent=2)
        with files["trajectories"].open("w", encoding="utf-8") as handle:
            json.dump(self._build_trajectories(), handle, ensure_ascii=False, indent=2)
        return {key: str(value) for key, value in files.items()}

    def _build_trajectories(self) -> dict:
        vehicles: dict[str, dict] = {}
        for frame in self._episode_frames:
            for drone_id, drone in frame.get("drones", {}).items():
                trajectory = vehicles.setdefault(drone_id, {
                    "vehicle_id": drone_id,
                    "vehicle_type": "uav",
                    "time_points": [],
                    "event_points": [],
                    "interpolated_points": [],
                    "route_points": [],
                })
                trajectory["time_points"].append({
                    "kind": "time",
                    "time": frame["time"],
                    "frame_index": frame["frame_index"],
                    "status": drone["status"],
                    "location": drone["current_location"],
                    "leg": vehicle_leg_snapshot(drone),
                })
        for event in self._events:
            drone_id = event.get("drone_id")
            if drone_id not in vehicles or event.get("location") is None:
                continue
            vehicles[drone_id]["event_points"].append({
                "kind": "event",
                "event_type": event.get("event_type"),
                "time": event.get("time"),
                "order_id": event.get("order_id"),
                "launchpad_id": event.get("launchpad_id"),
                "location": event.get("location"),
                "leg": vehicle_leg_snapshot(event.get("drone")),
            })
        return add_interpolated_points_to_trajectories({"vehicles": vehicles})

    def _location_to_dict(self, location: Location) -> dict:
        return {"lat": location.lat, "lon": location.lon, "node_id": location.node_id}

    def _stop_to_dict(self, stop: Stop) -> dict:
        return {
            "stop_id": stop.stop_id,
            "location": self._location_to_dict(stop.location),
            "stop_type": stop.stop_type.value,
            "related_order_id": stop.related_order_id,
            "related_launchpad_id": stop.related_launchpad_id,
        }

    def _order_to_dict(self, order: Order) -> dict:
        return {
            "order_id": order.order_id,
            "created_at": order.created_at,
            "origin_launchpad_id": order.origin_launchpad_id,
            "destination": self._location_to_dict(order.destination),
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
            "dispatch_time": order.dispatch_time,
            "delivery_time": order.delivery_time,
        }

    def _drone_to_dict(self, drone: Drone, current_time: int | None = None) -> dict:
        return {
            "vehicle_id": drone.vehicle_id,
            "vehicle_type": "uav",
            "current_location": self._location_to_dict(drone.current_location),
            "available_at": drone.available_at,
            "capacity": drone.capacity,
            "remaining_capacity": drone.remaining_capacity,
            "speed_mps": drone.speed_mps,
            "battery_level": drone.battery_level,
            "battery_capacity": drone.battery_capacity,
            "max_range_m": drone.max_range_m,
            "empty_mass_kg": drone.empty_mass_kg,
            "max_payload_kg": drone.max_payload_kg,
            "current_launchpad_id": drone.current_launchpad_id,
            "current_battery_id": drone.current_battery_id,
            "route_plan": [self._stop_to_dict(stop) for stop in drone.route_plan],
            "onboard_order_ids": list(drone.onboard_order_ids),
            "status": drone.status.value,
            "current_stop_eta": drone.current_stop_eta,
            "current_leg_started_at": drone.current_leg_started_at,
            "current_leg_target_stop_id": drone.current_leg_target_stop_id,
            "current_leg_progress_ratio": self._leg_progress_ratio(drone, current_time),
            "current_leg_distance_m": drone.current_leg_distance_m,
            "current_leg_travel_time_s": drone.current_leg_travel_time_s,
            "current_leg_path": [self._location_to_dict(location) for location in drone.current_leg_path],
            "total_distance_m": drone.total_distance_m,
            "total_energy": drone.total_energy,
        }

    def _leg_progress_ratio(self, drone: Drone, current_time: int | None) -> float:
        if drone.current_leg_arrived:
            return 1.0
        if current_time is None or drone.current_leg_started_at is None or drone.current_stop_eta is None:
            return 0.0
        duration = max(1, drone.current_stop_eta - drone.current_leg_started_at)
        return max(0.0, min(1.0, (current_time - drone.current_leg_started_at) / duration))

    def _launchpad_to_dict(self, launchpad: Launchpad) -> dict:
        return {
            "launchpad_id": launchpad.launchpad_id,
            "current_location": self._location_to_dict(launchpad.current_location),
            "drone_capacity": launchpad.drone_capacity,
            "order_queue": list(launchpad.order_queue),
            "drone_ids": list(launchpad.drone_ids),
            "charged_battery_ids": list(launchpad.charged_battery_ids),
            "charging_battery_ids": list(launchpad.charging_battery_ids),
            "status": launchpad.status.value,
            "reserved_return_slots": launchpad.reserved_return_slots,
            "deployment_cost": launchpad.deployment_cost,
            "op_cost_per_second": launchpad.op_cost_per_second,
            "service_cost_per_event": launchpad.service_cost_per_event,
            "active_time_s": launchpad.active_time_s,
            "service_events": launchpad.service_events,
        }

    def _battery_to_dict(self, battery: Battery) -> dict:
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
