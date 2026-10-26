from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any

from ..model.entities import Location
from ..utils.geo import haversine_distance
from .drone_energy_model import DroneEnergyModel

if TYPE_CHECKING:
    from ..world.airspace import Airspace
    from ..world.weather import Weather


class AirRouter:
    """Simplified UAV routing: geodesic distance with optional airspace penalty."""

    def __init__(
        self,
        config: dict,
        airspace: Optional["Airspace"] = None,
        weather: Optional["Weather"] = None,
    ):
        self.config = config
        self.airspace = airspace
        self.weather = weather

        uav_cfg = config.get("vehicles", {}).get("uavs", {})
        self._speed_mps = uav_cfg.get("speed_mps", 50.0 / 3.6)
        self._max_range_m = uav_cfg.get("max_range_m", 5000.0)
        self._battery_capacity = uav_cfg.get("battery_capacity", 720.0)
        self.energy_model = DroneEnergyModel(config)
        # Dispatch-candidate energy evaluations this episode (router is rebuilt on env.reset).
        self.task_energy_calls = 0

    def travel_distance(self, origin: Any, destination: Any) -> float:
        """Straight-line air distance in metres (with optional airspace penalty)."""
        o = self._as_location(origin)
        d = self._as_location(destination)
        base = haversine_distance(o.lat, o.lon, d.lat, d.lon)
        factor = 1.0
        if self.airspace is not None:
            factor = self.airspace.penalty_factor(o.lat, o.lon, d.lat, d.lon)
        return base * factor

    def travel_time(
        self,
        origin: Any,
        destination: Any,
        weather_factor: float | None = None,
    ) -> float:
        """Travel time in seconds."""
        dist = self.travel_distance(origin, destination)
        effective_speed = self._speed_mps * (1.0 if weather_factor is None else float(weather_factor))
        return dist / effective_speed if effective_speed > 0 else float("inf")

    def energy_cost(
        self,
        origin: Any,
        destination: Any,
        payload_mass_kg: float = 0.0,
        empty_mass_kg: float | None = None,
        weather_factor: float | None = None,
    ) -> float:
        """Battery energy consumed by one UAV leg, in Wh."""
        dist = self.travel_distance(origin, destination)
        wind = self._route_wind_state(origin, destination)
        return self.energy_model.estimate_leg_energy(
            dist,
            payload_mass_kg=payload_mass_kg,
            empty_mass_kg=empty_mass_kg,
            weather_factor=1.0 if weather_factor is None else float(weather_factor),
            wind_speed_mps=wind["wind_speed_mps"],
            wind_theta_deg=wind["wind_theta_deg"],
        )["E_leg_Wh"]

    def estimate_leg_energy(
        self,
        origin: Any,
        destination: Any,
        payload_mass_kg: float = 0.0,
        empty_mass_kg: float | None = None,
        weather_factor: float | None = None,
    ) -> dict:
        """Return task-model energy details for one UAV leg."""
        dist = self.travel_distance(origin, destination)
        wind = self._route_wind_state(origin, destination)
        result = self.energy_model.estimate_leg_energy(
            dist,
            payload_mass_kg=payload_mass_kg,
            empty_mass_kg=empty_mass_kg,
            weather_factor=1.0 if weather_factor is None else float(weather_factor),
            wind_speed_mps=wind["wind_speed_mps"],
            wind_theta_deg=wind["wind_theta_deg"],
        )
        result["distance_m"] = dist
        result.update(wind)
        return result

    def estimate_task_energy(
        self,
        drone: Any,
        order: Any,
        departure_launchpad: Any,
        return_launchpad: Any,
        weather_factor: float | None = None,
    ) -> dict:
        """Estimate empty reposition + loaded delivery + empty return energy."""
        self.task_energy_calls += 1
        drone_location = self._location_from_entity(drone)
        departure_location = self._location_from_entity(departure_launchpad)
        destination_location = self._order_destination(order)
        return_location = self._location_from_entity(return_launchpad)

        drone_launchpad_id = self._field(drone, "current_launchpad_id")
        departure_id = self._field(departure_launchpad, "launchpad_id")
        if drone_launchpad_id is not None and drone_launchpad_id == departure_id:
            distance_reposition = 0.0
        else:
            distance_reposition = self.travel_distance(drone_location, departure_location)
        distance_loaded = self.travel_distance(departure_location, destination_location)
        distance_return = self.travel_distance(destination_location, return_location)
        reposition_wind = self._route_wind_state(drone_location, departure_location)
        loaded_wind = self._route_wind_state(departure_location, destination_location)
        return_wind = self._route_wind_state(destination_location, return_location)
        battery_capacity = self._field(drone, "battery_capacity", self.energy_model.default_battery_capacity_Wh)
        battery_remaining = self._field(drone, "battery_level", battery_capacity)
        result = self.energy_model.estimate_task_energy(
            distance_loaded_m=distance_loaded,
            distance_return_m=distance_return,
            distance_reposition_m=distance_reposition,
            payload_mass_kg=self._field(order, "weight", 0.0),
            battery_remaining_Wh=battery_remaining,
            battery_capacity_Wh=battery_capacity,
            empty_mass_kg=self._field(drone, "empty_mass_kg", self.energy_model.default_empty_mass_kg),
            max_payload_kg=self._field(drone, "max_payload_kg", self.energy_model.default_max_payload_kg),
            weather_factor=1.0 if weather_factor is None else float(weather_factor),
            wind_speed_reposition_mps=reposition_wind["wind_speed_mps"],
            wind_theta_reposition_deg=reposition_wind["wind_theta_deg"],
            wind_speed_loaded_mps=loaded_wind["wind_speed_mps"],
            wind_theta_loaded_deg=loaded_wind["wind_theta_deg"],
            wind_speed_return_mps=return_wind["wind_speed_mps"],
            wind_theta_return_deg=return_wind["wind_theta_deg"],
        )
        result["distance_loaded_m"] = distance_loaded
        result["distance_reposition_m"] = distance_reposition
        result["distance_return_m"] = distance_return
        result["wind_reposition"] = reposition_wind
        result["wind_loaded"] = loaded_wind
        result["wind_return"] = return_wind
        range_limit = self._field(drone, "max_range_m", self._max_range_m)
        if range_limit is None:
            range_limit = self._max_range_m
        result["range_feasible"] = (
            distance_reposition <= range_limit
            and distance_loaded <= range_limit
            and distance_return <= range_limit
        )
        result["is_feasible"] = result["is_feasible"] and result["range_feasible"]
        return result

    def reserve_energy(
        self,
        battery_capacity_Wh: float | None = None,
        safe_return_energy_Wh: float = 0.0,
    ) -> float:
        return self.energy_model.reserve_energy_Wh(battery_capacity_Wh, safe_return_energy_Wh)

    def feasible(
        self,
        origin: Any,
        destination: Any,
        battery_level: Optional[float] = None,
        payload_mass_kg: float = 0.0,
        empty_mass_kg: float | None = None,
    ) -> bool:
        """Check if UAV can reach destination given battery and range constraints."""
        dist = self.travel_distance(origin, destination)
        if dist > self._max_range_m:
            return False
        if battery_level is not None:
            wind = self._route_wind_state(origin, destination)
            needed = self.energy_model.estimate_leg_energy(
                dist,
                payload_mass_kg=payload_mass_kg,
                empty_mass_kg=empty_mass_kg,
                wind_speed_mps=wind["wind_speed_mps"],
                wind_theta_deg=wind["wind_theta_deg"],
            )["E_leg_Wh"]
            if needed > battery_level:
                return False
        return True

    def _as_location(self, value: Any) -> Location:
        if isinstance(value, Location):
            return value
        if isinstance(value, dict):
            return Location(lat=float(value["lat"]), lon=float(value["lon"]))
        return Location(lat=float(value.lat), lon=float(value.lon))

    def _location_from_entity(self, value: Any) -> Location:
        if isinstance(value, Location):
            return value
        if isinstance(value, dict):
            if "current_location" in value:
                return self._as_location(value["current_location"])
            if "location" in value:
                return self._as_location(value["location"])
            return self._as_location(value)
        if hasattr(value, "current_location"):
            return self._as_location(value.current_location)
        if hasattr(value, "location"):
            return self._as_location(value.location)
        return self._as_location(value)

    def _order_destination(self, order: Any) -> Location:
        if isinstance(order, dict):
            return self._as_location(order["destination"])
        return self._as_location(order.destination)

    def _field(self, value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    def _route_wind_state(self, origin: Any, destination: Any) -> dict[str, float]:
        try:
            origin_loc = self._as_location(origin)
            destination_loc = self._as_location(destination)
            if origin_loc.lat == destination_loc.lat and origin_loc.lon == destination_loc.lon:
                return self._empty_wind_state()
        except (AttributeError, KeyError, TypeError, ValueError):
            return self._empty_wind_state()
        if self.weather is None:
            return self._empty_wind_state()
        return self.weather.uav_wind_state(origin, destination)

    def _empty_wind_state(self) -> dict[str, float]:
        return {
            "wind_speed_mps": 0.0,
            "wind_theta_deg": 0.0,
            "flight_bearing_deg": 0.0,
            "wind_toward_deg": 0.0,
        }
