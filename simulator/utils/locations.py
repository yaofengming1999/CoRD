from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from ..model.entities import Location

if TYPE_CHECKING:
    from ..world.road_network import RoadNetwork


def location_from_mapping(
    data: Mapping[str, Any],
    *,
    road_network: "RoadNetwork | None" = None,
    mode: str = "drive",
    snap_to_road: bool = True,
) -> Location:
    raw_node_id = data.get("node_id")
    node_id = int(raw_node_id) if raw_node_id not in (None, "") else None
    location = Location(
        lat=float(data["lat"]),
        lon=float(data["lon"]),
        node_id=node_id,
    )
    if road_network is None or not snap_to_road:
        return location
    return snap_location_to_road(location, road_network, mode=mode)


def snap_location_to_road(
    location: Location,
    road_network: "RoadNetwork",
    *,
    mode: str = "drive",
) -> Location:
    if location.node_id is not None:
        node_location = road_network.node_location(location.node_id, mode=mode)
        if node_location is not None:
            return node_location

    node_id = road_network.nearest_node(location, mode=mode)
    if node_id is not None and node_id != -1:
        node_location = road_network.node_location(node_id, mode=mode)
        if node_location is not None:
            return node_location
    return location
