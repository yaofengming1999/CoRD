from __future__ import annotations

from dataclasses import dataclass

from ..model.entities import Battery, Drone, Order


@dataclass(frozen=True)
class TimeAdvanceDecision:
    next_time: int
    delta_t: int
    reason: str


def next_simulation_time(
    *,
    config: dict,
    current_time: int,
    max_horizon: int,
    default_delta_t: int,
    pending_orders_by_time: list[Order],
    next_order_index: int,
    drones: dict[str, Drone],
    orders: dict[str, Order],
    batteries: dict[str, Battery],
) -> TimeAdvanceDecision:
    sim_cfg = config.get("simulation", {})
    mode = str(sim_cfg.get("time_advancement", sim_cfg.get("mode", "fixed_step"))).lower()
    if mode in {"fixed", "fixed_step", "fixed-step"}:
        delta_t = max(0, min(int(default_delta_t), int(max_horizon) - int(current_time)))
        return TimeAdvanceDecision(current_time + delta_t, delta_t, "fixed_step")
    if mode not in {"event", "event_driven", "event-driven"}:
        raise ValueError(f"Unknown simulation time advancement mode: {mode}")

    max_delta_t = max(1, int(config.get("event_driven", {}).get("max_delta_t_seconds", default_delta_t)))
    if current_time >= max_horizon:
        return TimeAdvanceDecision(max_horizon, 0, "max_horizon")
    if (
        next_order_index < len(pending_orders_by_time)
        and pending_orders_by_time[next_order_index].created_at <= current_time
    ):
        return TimeAdvanceDecision(current_time, 0, "order_release_due")

    candidates: list[tuple[int, str]] = [(min(max_horizon, current_time + max_delta_t), "event_safety_tick")]
    if next_order_index < len(pending_orders_by_time):
        candidates.append((int(pending_orders_by_time[next_order_index].created_at), "next_order_arrival"))
    for drone in drones.values():
        if drone.current_stop_eta is not None:
            candidates.append((int(drone.current_stop_eta), "drone_stop_eta"))
        if drone.current_leg_started_at is not None:
            candidates.append((int(drone.current_leg_started_at), "drone_leg_departure"))
        candidates.append((int(drone.available_at), "drone_available"))
    for order in orders.values():
        candidates.append((int(order.deadline), "order_deadline"))
    for battery in batteries.values():
        if battery.charge_complete_at is not None:
            candidates.append((int(battery.charge_complete_at), "battery_charge_complete"))

    future = [(time, reason) for time, reason in candidates if current_time < time <= max_horizon]
    if not future:
        return TimeAdvanceDecision(max_horizon, max_horizon - current_time, "max_horizon")
    next_time, reason = min(future, key=lambda item: (item[0], item[1]))
    return TimeAdvanceDecision(next_time, next_time - current_time, reason)
