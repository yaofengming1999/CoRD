from __future__ import annotations

import math
from typing import Any


class Weather:
    """Simulator-level weather parameters.

    Wind direction uses the meteorological convention by default:
    0 means wind comes from north, 90 means wind comes from east.
    """

    def __init__(self, config: dict):
        weather_cfg = config.get("weather", {})
        wind_cfg = weather_cfg.get("wind", {})
        self.enabled = bool(weather_cfg.get("enable", False))
        self.wind_speed_mps = float(weather_cfg.get("wind_speed_mps", wind_cfg.get("speed_mps", 0.0)))
        self.wind_direction_deg = float(
            weather_cfg.get("wind_direction_deg", wind_cfg.get("direction_deg", 0.0))
        ) % 360.0
        self.wind_direction_convention = str(
            weather_cfg.get("wind_direction_convention", wind_cfg.get("direction_convention", "from"))
        ).lower()
        self.uav_air_speed_mps = float(
            config.get("vehicles", {})
            .get("uavs", {})
            .get("speed_mps", 50.0 / 3.6)
        )
        self._state: dict = {}

    def reset(self):
        self._state = self._current_state()

    def step(self, current_time: int):
        self._state = self._current_state()

    def get_state(self) -> dict:
        return dict(self._state)

    def uav_speed_factor(self, origin: Any | None = None, destination: Any | None = None) -> float:
        """Ground-speed multiplier kept for compatibility.

        The paper energy model uses wind through alpha_w, not by changing
        UAV ground speed. Therefore this remains 1.0.
        """
        return 1.0

    def uav_wind_state(self, origin: Any | None = None, destination: Any | None = None) -> dict[str, float]:
        """Return route-specific wind inputs for Gao et al. Eq. (4)-(5)."""
        if not self.enabled or self.wind_speed_mps <= 0.0:
            return self._empty_wind_state()
        if origin is None or destination is None:
            return self._empty_wind_state()
        if self.uav_air_speed_mps <= 0.0:
            return self._empty_wind_state()

        origin_lat_lon = self._lat_lon(origin)
        destination_lat_lon = self._lat_lon(destination)
        if origin_lat_lon == destination_lat_lon:
            return self._empty_wind_state()

        flight_bearing = self._bearing_deg(origin_lat_lon, destination_lat_lon)
        wind_toward = self._wind_toward_deg()
        theta_deg = abs(self._angle_delta_deg(flight_bearing, wind_toward))
        return {
            "wind_speed_mps": self.wind_speed_mps,
            "wind_theta_deg": theta_deg,
            "flight_bearing_deg": flight_bearing,
            "wind_toward_deg": wind_toward,
        }

    def road_speed_factor(self) -> float:
        return 1.0

    def _current_state(self) -> dict:
        return {
            "enabled": self.enabled,
            "wind_speed_mps": self.wind_speed_mps,
            "wind_direction_deg": self.wind_direction_deg,
            "wind_direction_convention": self.wind_direction_convention,
            "wind_toward_deg": self._wind_toward_deg(),
        }

    def _empty_wind_state(self) -> dict[str, float]:
        return {
            "wind_speed_mps": 0.0,
            "wind_theta_deg": 0.0,
            "flight_bearing_deg": 0.0,
            "wind_toward_deg": self._wind_toward_deg(),
        }

    def _wind_toward_deg(self) -> float:
        if self.wind_direction_convention == "toward":
            return self.wind_direction_deg
        return (self.wind_direction_deg + 180.0) % 360.0

    def _bearing_deg(self, origin: Any, destination: Any) -> float:
        lat1, lon1 = self._lat_lon(origin)
        lat2, lon2 = self._lat_lon(destination)
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        x = math.sin(delta_lon) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lon)
        return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

    def _lat_lon(self, value: Any) -> tuple[float, float]:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]), float(value[1])
        if isinstance(value, dict):
            return float(value["lat"]), float(value["lon"])
        return float(value.lat), float(value.lon)

    def _angle_delta_deg(self, a: float, b: float) -> float:
        return (a - b + 540.0) % 360.0 - 180.0
