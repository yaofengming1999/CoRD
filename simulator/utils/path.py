from __future__ import annotations

from collections.abc import Sequence

from ..model.entities import Location
from .geo import haversine_distance, interpolate_location


def polyline_distance_m(points: Sequence[Location]) -> float:
    """Approximate path length over a lat/lon polyline."""
    if len(points) < 2:
        return 0.0
    return sum(
        haversine_distance(a.lat, a.lon, b.lat, b.lon)
        for a, b in zip(points, points[1:])
    )


def interpolate_along_polyline(points: Sequence[Location], distance_m: float) -> Location:
    """Return the location at distance_m along a polyline."""
    if not points:
        raise ValueError("Cannot interpolate an empty path.")
    if len(points) == 1 or distance_m <= 0.0:
        return Location(lat=points[0].lat, lon=points[0].lon, node_id=points[0].node_id)

    remaining = float(distance_m)
    for start, end in zip(points, points[1:]):
        segment = haversine_distance(start.lat, start.lon, end.lat, end.lon)
        if segment <= 0.0:
            continue
        if remaining <= segment:
            frac = max(0.0, min(1.0, remaining / segment))
            lat, lon = interpolate_location(start.lat, start.lon, end.lat, end.lon, frac)
            return Location(lat=lat, lon=lon)
        remaining -= segment

    last = points[-1]
    return Location(lat=last.lat, lon=last.lon, node_id=last.node_id)
