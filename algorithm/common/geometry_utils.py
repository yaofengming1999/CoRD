from __future__ import annotations

from simulator.utils.geo import haversine_distance


def distance_m(origin: dict | None, destination: dict | None) -> float:
    if not origin or not destination:
        return 0.0
    return float(
        haversine_distance(
            float(origin.get("lat", 0.0)),
            float(origin.get("lon", 0.0)),
            float(destination.get("lat", 0.0)),
            float(destination.get("lon", 0.0)),
        )
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


def urgency(order: dict, current_time: int) -> float:
    time_to_deadline = int(order.get("deadline", current_time)) - current_time
    return 1.0 / (1.0 + max(0.0, float(time_to_deadline)) / 60.0)


def relocation_settings(config: dict) -> tuple[bool, int]:
    """(enabled, decision interval seconds) for parked-drone relocation."""
    reloc = config.get("drone_decision", {}).get("rebalance", {}) or {}
    return bool(reloc.get("enabled", False)), int(reloc.get("interval_s", 120))


def drone_has_departure_battery(drone: dict, launchpad: dict) -> bool:
    return bool(
        list(launchpad.get("charged_battery_ids", []))
        + list(launchpad.get("charging_battery_ids", []))
    )


def relocation_worthwhile(state: dict, drone: dict, launchpads: list[dict]) -> bool:
    """Gate for a parked idle drone: only consider moving when its own pad has
    no work, it has a launchpad battery to evaluate, and some other pad has
    more queued work than docked drones to serve it."""
    current_pad_id = str(drone.get("current_launchpad_id"))
    current_pad = next(
        (lp for lp in launchpads if str(lp.get("launchpad_id")) == current_pad_id),
        None,
    )
    if current_pad is None:
        return False
    if current_pad.get("order_queue"):
        return False
    if not drone_has_departure_battery(drone, current_pad):
        return False
    return any(
        str(lp.get("launchpad_id")) != current_pad_id
        and lp.get("order_queue")
        and len(lp.get("order_queue", [])) > len(lp.get("drone_ids", []))
        for lp in launchpads
    )
