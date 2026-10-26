from __future__ import annotations

from ..utils.geo import haversine_distance, interpolate_location


DEFAULT_INTERPOLATED_POINTS_PER_INTERVAL = 3


def add_interpolated_points_to_trajectories(
    trajectories: dict,
    points_per_interval: int = DEFAULT_INTERPOLATED_POINTS_PER_INTERVAL,
) -> dict:
    output = dict(trajectories)
    output["interpolated_points_per_interval"] = int(points_per_interval)
    vehicles: dict[str, dict] = {}

    for vehicle_id, trajectory in output.get("vehicles", {}).items():
        next_trajectory = dict(trajectory)
        next_trajectory.setdefault("vehicle_id", vehicle_id)
        next_trajectory.setdefault("time_points", [])
        next_trajectory.setdefault("event_points", [])
        next_trajectory["interpolated_points"] = build_interpolated_points(
            next_trajectory["time_points"],
            next_trajectory["event_points"],
            points_per_interval=points_per_interval,
        )
        merged = (
            next_trajectory["time_points"]
            + next_trajectory["interpolated_points"]
            + next_trajectory["event_points"]
        )
        merged.sort(key=route_point_sort_key)
        next_trajectory["route_points"] = merged
        vehicles[vehicle_id] = next_trajectory

    output["vehicles"] = vehicles
    return output


def build_interpolated_points(
    time_points: list[dict],
    event_points: list[dict],
    points_per_interval: int = DEFAULT_INTERPOLATED_POINTS_PER_INTERVAL,
) -> list[dict]:
    points_per_interval = int(points_per_interval)
    if points_per_interval <= 0 or len(time_points) < 2:
        return []

    sorted_time_points = sorted(time_points, key=route_point_sort_key)
    anchors = sorted(
        [point for point in time_points + event_points if point.get("location") is not None],
        key=route_point_sort_key,
    )
    has_leg_data = any(extract_leg(point) is not None for point in anchors)
    interpolated: list[dict] = []

    for start, end in zip(sorted_time_points, sorted_time_points[1:]):
        start_time = numeric_time(start)
        end_time = numeric_time(end)
        if start_time is None or end_time is None or end_time <= start_time:
            continue

        for sample_index in range(1, points_per_interval + 1):
            fraction = sample_index / (points_per_interval + 1)
            sample_time = start_time + (end_time - start_time) * fraction
            location = location_at_time(
                sample_time,
                anchors,
                start,
                end,
                has_leg_data=has_leg_data,
            )
            if location is None:
                continue
            interpolated.append(
                {
                    "kind": "interpolated",
                    "time": clean_number(sample_time),
                    "frame_index": None,
                    "from_sequence": start.get("sequence"),
                    "to_sequence": end.get("sequence"),
                    "sample_index": sample_index,
                    "sample_count": points_per_interval,
                    "status": start.get("status"),
                    "location": location,
                }
            )

    return interpolated


def location_at_time(
    target_time: float,
    anchors: list[dict],
    fallback_start: dict,
    fallback_end: dict,
    has_leg_data: bool = True,
) -> dict | None:
    leg = leg_at_time(target_time, anchors)
    if leg is not None:
        return interpolate_leg_location(leg, target_time)

    if not has_leg_data:
        return linear_location_between_anchors(target_time, anchors, fallback_start, fallback_end)

    previous = previous_anchor(target_time, anchors)
    if previous is not None:
        return copy_location_dict(previous.get("location"))
    return copy_location_dict(fallback_start.get("location"))


def linear_location_between_anchors(
    target_time: float,
    anchors: list[dict],
    fallback_start: dict,
    fallback_end: dict,
) -> dict | None:
    before = None
    after = None

    for anchor in anchors:
        anchor_time = numeric_time(anchor)
        if anchor_time is None:
            continue
        if abs(anchor_time - target_time) < 1e-9:
            return copy_location_dict(anchor.get("location"))
        if anchor_time < target_time:
            before = anchor
        elif anchor_time > target_time:
            after = anchor
            break

    before = before or fallback_start
    after = after or fallback_end
    before_time = numeric_time(before)
    after_time = numeric_time(after)
    if before_time is None or after_time is None:
        return None
    if after_time <= before_time:
        return copy_location_dict(before.get("location"))
    fraction = (target_time - before_time) / (after_time - before_time)
    return interpolate_location_dict(
        before.get("location"),
        after.get("location"),
        max(0.0, min(1.0, fraction)),
    )


def previous_anchor(target_time: float, anchors: list[dict]) -> dict | None:
    previous = None
    for anchor in anchors:
        anchor_time = numeric_time(anchor)
        if anchor_time is None:
            continue
        if anchor_time <= target_time:
            previous = anchor
        else:
            break
    return previous


def leg_at_time(target_time: float, anchors: list[dict]) -> dict | None:
    candidates = []
    for anchor in anchors:
        leg = extract_leg(anchor)
        if leg is None:
            continue
        started_at = numeric_value(leg.get("started_at"))
        eta = numeric_value(leg.get("eta"))
        if started_at is None or eta is None or eta < started_at:
            continue
        if started_at <= target_time <= eta:
            candidates.append(leg)

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda leg: (
            numeric_value(leg.get("started_at")) or 0.0,
            -(numeric_value(leg.get("eta")) or 0.0),
        ),
    )


def extract_leg(point: dict) -> dict | None:
    leg = point.get("leg")
    if isinstance(leg, dict):
        path = clean_path(leg.get("path"))
        if path:
            return {
                "started_at": leg.get("started_at"),
                "eta": leg.get("eta"),
                "target_stop_id": leg.get("target_stop_id"),
                "path": path,
            }

    path = clean_path(point.get("current_leg_path"))
    if not path:
        return None
    return {
        "started_at": point.get("current_leg_started_at"),
        "eta": point.get("current_stop_eta"),
        "target_stop_id": point.get("current_leg_target_stop_id"),
        "path": path,
    }


def vehicle_leg_snapshot(vehicle: dict | None) -> dict | None:
    if not vehicle:
        return None
    path = clean_path(vehicle.get("current_leg_path"))
    if not path:
        return None
    started_at = vehicle.get("current_leg_started_at")
    eta = vehicle.get("current_stop_eta")
    if started_at is None or eta is None:
        return None
    return {
        "started_at": started_at,
        "eta": eta,
        "target_stop_id": vehicle.get("current_leg_target_stop_id"),
        "path": path,
    }


def clean_path(path: list[dict] | None) -> list[dict]:
    if not path:
        return []
    points = []
    for location in path:
        copied = copy_location_dict(location)
        if copied is not None and "lat" in copied and "lon" in copied:
            points.append(copied)
    return points


def interpolate_leg_location(leg: dict, target_time: float) -> dict | None:
    path = clean_path(leg.get("path"))
    if not path:
        return None
    started_at = numeric_value(leg.get("started_at"))
    eta = numeric_value(leg.get("eta"))
    if started_at is None or eta is None:
        return None
    if target_time <= started_at or eta <= started_at:
        return copy_location_dict(path[0])
    if target_time >= eta:
        return copy_location_dict(path[-1])

    progress = (target_time - started_at) / (eta - started_at)
    return interpolate_along_path(path, progress)


def interpolate_along_path(path: list[dict], progress: float) -> dict | None:
    if not path:
        return None
    if len(path) == 1 or progress <= 0.0:
        return copy_location_dict(path[0])
    if progress >= 1.0:
        return copy_location_dict(path[-1])

    total_distance = path_distance_m(path)
    if total_distance <= 0.0:
        return copy_location_dict(path[-1])

    remaining = total_distance * progress
    for start, end in zip(path, path[1:]):
        segment = haversine_distance(
            float(start["lat"]),
            float(start["lon"]),
            float(end["lat"]),
            float(end["lon"]),
        )
        if segment <= 0.0:
            continue
        if remaining <= segment:
            fraction = max(0.0, min(1.0, remaining / segment))
            lat, lon = interpolate_location(
                float(start["lat"]),
                float(start["lon"]),
                float(end["lat"]),
                float(end["lon"]),
                fraction,
            )
            return {"lat": lat, "lon": lon, "node_id": None}
        remaining -= segment

    return copy_location_dict(path[-1])


def path_distance_m(path: list[dict]) -> float:
    if len(path) < 2:
        return 0.0
    return sum(
        haversine_distance(
            float(start["lat"]),
            float(start["lon"]),
            float(end["lat"]),
            float(end["lon"]),
        )
        for start, end in zip(path, path[1:])
    )


def interpolate_location_dict(
    start: dict | None,
    end: dict | None,
    fraction: float,
) -> dict | None:
    if start is None or end is None:
        return None
    if "lat" not in start or "lon" not in start or "lat" not in end or "lon" not in end:
        return None
    return {
        "lat": float(start["lat"]) + (float(end["lat"]) - float(start["lat"])) * fraction,
        "lon": float(start["lon"]) + (float(end["lon"]) - float(start["lon"])) * fraction,
        "node_id": None,
    }


def copy_location_dict(location: dict | None) -> dict | None:
    if location is None:
        return None
    return dict(location)


def numeric_time(point: dict) -> float | None:
    value = point.get("time")
    return numeric_value(value)


def numeric_value(value) -> float | None:
    if value is None:
        return None
    return float(value)


def clean_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else value


def route_point_sort_key(point: dict) -> tuple[float, int]:
    kind_order = {
        "time": 0,
        "event": 1,
        "interpolated": 2,
    }
    time_value = numeric_time(point)
    return (
        time_value if time_value is not None else 0.0,
        kind_order.get(point.get("kind"), 99),
    )
