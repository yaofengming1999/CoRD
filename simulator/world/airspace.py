from __future__ import annotations
from dataclasses import dataclass


@dataclass
class NoFlyZone:
    center_lat: float
    center_lon: float
    radius_m: float
    label: str = ""


class Airspace:
    """Airspace metadata. Stub for Stage A; no-fly zones added in Stage C."""

    def __init__(self, config: dict):
        self.config = config
        self.no_fly_zones: list[NoFlyZone] = []

    def load(self):
        pass

    def penalty_factor(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Return a multiplier >= 1.0 applied to UAV distance/time."""
        return 1.0
