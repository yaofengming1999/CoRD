"""The frozen per-city instance spec.

``data/dataset/order_instance/city_instances.json`` holds the six properties
that define a city's instance -- time window, map extent, pad count, drone
count, battery count, and the resulting order count -- so a run names a city
and gets the frozen instance rather than reconstructing it from flags. Only two
of the six used to be frozen (the layout and the extent box); the rest
travelled on the command line, which is how two runs of "the same" city end up
incomparable.

``city_defaults(city)`` returns the ``scenario.*`` keys for that city. The
caller's own keys win, so an explicit flag is a deliberate override and a sweep
still works; nothing here forces a value.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data/dataset/order_instance/city_instances.json"

# Keys of ``shared`` that are scenario parameters (the rest are provenance:
# the sizing rule's inputs, kept so a fleet number can be traced back).
_SCENARIO_KEYS = (
    "extent",
    "pad_spacing_km",
    "min_flight_km",
    "max_flight_km",
    "window",
    "drone_placement",
    "demand_split",
    "prune_empty_pads",
    "drones_per_launchpad",
    "batteries_per_drone",
    "n_battery_types",
)


@lru_cache(maxsize=1)
def registry(path: str | None = None) -> dict:
    source = Path(path) if path else REGISTRY_PATH
    if not source.exists():
        return {}
    return json.loads(source.read_text())


def known_cities(path: str | None = None) -> tuple[str, ...]:
    return tuple(registry(path).get("cities", {}))


def city_defaults(city: str, path: str | None = None) -> dict:
    """Frozen ``scenario.*`` values for this city, or {} when it has no entry.

    A city with no entry is not an error: the scenario builder's own defaults
    still apply, so an unregistered city behaves exactly as it did before.
    """
    data = registry(path)
    entry = data.get("cities", {}).get(city)
    if entry is None:
        return {}
    # shared first, then the city's own, so a city that names a key (grubhub's
    # 4-hour window and 3 km pitch) overrides the shared value
    out = {key: data["shared"][key] for key in _SCENARIO_KEYS if key in data["shared"]}
    out.update({key: entry[key] for key in _SCENARIO_KEYS if key in entry})
    return out


def city_record(city: str, path: str | None = None) -> dict | None:
    """The whole entry, including the measured order counts and the layout hash."""
    return registry(path).get("cities", {}).get(city)
