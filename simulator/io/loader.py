from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import yaml

from simulator.model.entities import Battery, Drone, Launchpad, Location, Order
from simulator.model.enums import BatteryStatus, LaunchpadStatus, OrderStatus, VehicleStatus
from simulator.utils.geo import haversine_distance
from simulator.utils.locations import location_from_mapping
from simulator.io.city_orders import day_files
from simulator.io.city_registry import REGISTRY_PATH, city_defaults
from simulator.io.launchpad_layout import parse_window
from simulator.io.scenario import (
    DEFAULT_DRONE_PLACEMENT,
    DEFAULT_EXTENT,
    DEFAULT_PRUNE_EMPTY_PADS,
    DEFAULT_MAX_FLIGHT_KM,
    DEFAULT_MIN_FLIGHT_KM,
    DEFAULT_PAD_SPACING_KM,
    DEFAULT_MAX_SLACK_SECONDS,
    DEFAULT_MIN_SLACK_SECONDS,
    DEFAULT_SLACK_FLIGHT_MULTIPLE,
    build_scenario,
)


@dataclass
class InstanceData:
    instance_path: Path
    orders: list[Order]
    drones: list[Drone]
    launchpads: list[Launchpad]
    batteries: list[Battery]
    weather: dict


class InstanceLoader:
    """Load one fully specified simulator instance from an instance/day folder."""

    REQUIRED_FILES = (
        "orders.yaml",
        "drones.yaml",
        "launchpads.yaml",
        "batteries.yaml",
        "weather.yaml",
    )

    def __init__(self, config: dict, road_network):
        self.config = config
        self.road_network = road_network

    def load(self, *, seed: int | None = None) -> InstanceData:
        if self.config.get("scenario", {}).get("enabled"):
            return self._load_scenario(seed=seed)
        instance_path = self._instance_path(seed=seed)
        self._validate_required_files(instance_path)

        launchpads = self._load_launchpads(instance_path / "launchpads.yaml")
        launchpad_by_id = {launchpad.launchpad_id: launchpad for launchpad in launchpads}
        drones = self._load_drones(instance_path / "drones.yaml", launchpad_by_id)
        drone_by_id = {drone.vehicle_id: drone for drone in drones}
        batteries = self._load_batteries(instance_path / "batteries.yaml")
        weather = self._read_yaml_mapping(instance_path / "weather.yaml")

        self._bind_launchpad_drones(launchpads, drones)
        self._bind_batteries(launchpads, drone_by_id, batteries)
        orders = self._load_orders(instance_path / "orders.yaml", launchpads)

        return InstanceData(
            instance_path=instance_path,
            orders=orders,
            drones=drones,
            launchpads=launchpads,
            batteries=batteries,
            weather=weather,
        )

    def _load_scenario(self, *, seed: int | None) -> InstanceData:
        """Build the instance from ``config["scenario"]`` instead of an instance folder.

        The fleet is constructed from four counts and only the order stream comes
        from disk, so a different launchpad count or battery mix is a parameter
        rather than a regenerated instance set. Everything downstream -- the
        state builder, the transition engine, the metrics -- sees the same
        ``InstanceData`` either way.
        """
        spec = dict(self.config.get("scenario", {}))
        if spec.get("city") is None:
            raise ValueError("scenario.city is required when scenario.enabled is true")
        # The frozen instance spec fills in whatever the caller did not say, so
        # naming a registered city is enough to reproduce it; an explicit key is
        # a deliberate override and still wins. A city with no entry gets {} and
        # behaves exactly as it did before the registry existed.
        frozen = city_defaults(str(spec["city"]))
        if frozen and spec.get("n_launchpads") is not None:
            # the caller fixed the pad count, which is the other half of a
            # mutually exclusive pair -- do not also hand them the frozen pitch
            frozen.pop("pad_spacing_km", None)
        for key, value in frozen.items():
            if spec.get(key) is None:
                spec[key] = value
        if frozen:
            # write the merged spec back, so the run's resolved_config records the
            # instance it actually played rather than just the city's name
            self.config["scenario"] = {**self.config.get("scenario", {}), **spec}
        for required in ("drones_per_launchpad", "batteries_per_drone"):
            if spec.get(required) is None:
                raise ValueError(
                    f"scenario.{required} is required when scenario.enabled is true and "
                    f"{spec['city']} has no entry in {REGISTRY_PATH.name}"
                )
        # the pad layout is driven by exactly one of a count and a pitch; giving
        # neither defaults to the pitch, which is the layout the city statistics
        # were measured for
        n_launchpads = spec.get("n_launchpads")
        pad_spacing_km = spec.get("pad_spacing_km")
        if n_launchpads is None and pad_spacing_km is None:
            pad_spacing_km = DEFAULT_PAD_SPACING_KM
        if n_launchpads is not None and pad_spacing_km is not None:
            raise ValueError(
                "scenario.n_launchpads and scenario.pad_spacing_km both set; "
                "one fixes the pad count and the other fixes the pad pitch, so give one"
            )

        day = spec.get("day")
        split = str(spec.get("split", "train"))
        root = spec.get("dataset_root")
        if day is None:
            # pick a day of the split from the episode seed, the scenario analogue
            # of instance.selection=random_saved
            days = len(day_files(str(spec["city"]), split, **({} if root is None else {"root": root})))
            day = random.Random(int(seed if seed is not None else 0)).randrange(days)

        orders, drones, launchpads, batteries, report = build_scenario(
            str(spec["city"]),
            None if n_launchpads is None else int(n_launchpads),
            int(spec["drones_per_launchpad"]),
            int(spec["batteries_per_drone"]),
            int(spec.get("n_battery_types", 1)),
            pad_spacing_km=None if pad_spacing_km is None else float(pad_spacing_km),
            extent=str(spec.get("extent", DEFAULT_EXTENT)),
            window=parse_window(spec.get("window", "11-12")),
            drone_placement=str(spec.get("drone_placement", DEFAULT_DRONE_PLACEMENT)),
            prune_empty_pads=bool(spec.get("prune_empty_pads", DEFAULT_PRUNE_EMPTY_PADS)),
            demand_split=str(spec.get("demand_split", "train")),
            min_drones_per_pad=int(spec.get("min_drones_per_pad", 0)),
            split=split,
            day=int(day),
            config=self.config,
            dataset_root=root,
            min_flight_km=float(spec.get("min_flight_km", DEFAULT_MIN_FLIGHT_KM)),
            max_flight_km=float(spec.get("max_flight_km", DEFAULT_MAX_FLIGHT_KM)),
            slack_flight_multiple=float(
                spec.get("slack_flight_multiple", DEFAULT_SLACK_FLIGHT_MULTIPLE)
            ),
            min_slack_seconds=int(spec.get("min_slack_seconds", DEFAULT_MIN_SLACK_SECONDS)),
            max_slack_seconds=int(spec.get("max_slack_seconds", DEFAULT_MAX_SLACK_SECONDS)),
            seed=int(seed if seed is not None else 42),
        )
        runtime = self.config.setdefault("_runtime", {})
        runtime["scenario_report"] = report
        return InstanceData(
            # a synthetic path: nothing reads it as a directory, but the records
            # and logs identify an episode by it, so it has to name the cell
            instance_path=Path(
                f"scenario_{spec['city']}/{split}/day{day}"
                f"/lp{report.n_launchpads}_d{spec['drones_per_launchpad']}"
                f"_b{spec['batteries_per_drone']}_t{spec.get('n_battery_types', 1)}"
                f"{'' if report.drone_placement == 'uniform' else '_' + report.drone_placement}"
            ),
            orders=orders,
            drones=drones,
            launchpads=launchpads,
            batteries=batteries,
            weather={},
        )

    def _instance_path(self, *, seed: int | None) -> Path:
        instance_cfg = self.config.get("instance", {})
        if instance_cfg.get("path"):
            raise ValueError(
                "instance.path is no longer supported. Use instance.root, "
                "instance.city, instance.split, instance.selection, and instance.fixed_id instead."
            )

        split_path = self._split_path(instance_cfg)
        selection = str(instance_cfg.get("selection", "fixed")).lower()
        if selection == "fixed":
            fixed_id = instance_cfg.get("fixed_id")
            if not fixed_id:
                raise ValueError("instance.selection=fixed requires instance.fixed_id")
            path = self._fixed_instance_path(split_path, str(fixed_id))
        elif selection == "random_saved":
            candidates = self._saved_instance_dirs(split_path)
            if not candidates:
                raise FileNotFoundError(
                    f"No saved instance folders with required YAML files found under {split_path}"
                )
            rng_seed = int(seed if seed is not None else self.config.get("simulation", {}).get("seed", 0))
            path = candidates[random.Random(rng_seed).randrange(len(candidates))]
        else:
            raise ValueError(
                "Unsupported instance.selection: "
                f"{selection!r}. Expected 'fixed' or 'random_saved'."
            )

        if not path.is_dir():
            raise FileNotFoundError(f"Instance path does not exist or is not a directory: {path}")
        return path

    def _split_path(self, instance_cfg: dict) -> Path:
        city = instance_cfg.get("city")
        if not city:
            raise ValueError("Simulator config must define instance.city")
        split = instance_cfg.get("split")
        if not split:
            raise ValueError("Simulator config must define instance.split")
        split_path_part = Path(str(split))
        if split_path_part.is_absolute() or len(split_path_part.parts) != 1:
            raise ValueError(f"instance.split must be a single folder name, got {split!r}")
        root = self._resolve_repo_path(str(instance_cfg.get("root", "data/instances")))
        split_path = root / str(city) / str(split)
        if not split_path.is_dir():
            raise FileNotFoundError(f"Instance split folder does not exist: {split_path}")
        return split_path

    def _fixed_instance_path(self, split_path: Path, fixed_id: str) -> Path:
        fixed_path = Path(fixed_id)
        if fixed_path.is_absolute() or len(fixed_path.parts) != 1:
            raise ValueError(
                "instance.fixed_id must be a single folder name under the instance split, "
                f"got {fixed_id!r}"
            )
        return split_path / fixed_id

    def _saved_instance_dirs(self, city_path: Path) -> list[Path]:
        return sorted(
            path for path in city_path.iterdir()
            if path.is_dir() and all((path / name).is_file() for name in self.REQUIRED_FILES)
        )

    def _resolve_repo_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = Path(__file__).resolve().parent.parent.parent / path
        return path

    def _validate_required_files(self, instance_path: Path) -> None:
        missing = [name for name in self.REQUIRED_FILES if not (instance_path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"Instance path {instance_path} is missing required files: {', '.join(missing)}"
            )

    def _load_orders(self, path: Path, launchpads: list[Launchpad]) -> list[Order]:
        orders: list[Order] = []
        for item in self._read_yaml_list(path, allow_empty=True):
            origin_launchpad_id = item.get("origin_launchpad_id")
            if origin_launchpad_id is None:
                pickup = self._location_from_mapping(item["pickup"])
                origin_launchpad_id = min(
                    launchpads,
                    key=lambda launchpad: haversine_distance(
                        pickup.lat,
                        pickup.lon,
                        launchpad.location.lat,
                        launchpad.location.lon,
                    ),
                ).launchpad_id
            if str(origin_launchpad_id) not in {launchpad.launchpad_id for launchpad in launchpads}:
                raise ValueError(
                    f"Order {item.get('order_id')} references unknown origin launchpad {origin_launchpad_id}"
                )
            destination_data = item.get("destination", item.get("dropoff"))
            if destination_data is None:
                raise ValueError(f"Order {item.get('order_id')} must define destination")
            destination = self._location_from_mapping(destination_data)
            created_at = int(item["created_at"])
            orders.append(
                Order(
                    order_id=str(item["order_id"]),
                    created_at=created_at,
                    origin_launchpad_id=str(origin_launchpad_id),
                    destination=destination,
                    size=float(item.get("size", 1.0)),
                    weight=float(item.get("weight", 1.0)),
                    value=float(item.get("value", 0.0)),
                    deadline=int(item["deadline"]),
                    destination_endpoint_id=(
                        str(item["destination_endpoint_id"])
                        if item.get("destination_endpoint_id") is not None
                        else None
                    ),
                    status=OrderStatus.PENDING,
                    base_revenue=float(item.get("base_revenue", item.get("value", 0.0))),
                    distance_revenue=float(item.get("distance_revenue", 0.0)),
                    weight_revenue=float(item.get("weight_revenue", 0.0)),
                )
            )
        orders.sort(key=lambda order: order.created_at)
        return orders

    def _load_drones(self, path: Path, launchpad_by_id: dict[str, Launchpad]) -> list[Drone]:
        drones: list[Drone] = []
        defaults = self.config.get("vehicles", {}).get("uavs", {})
        for item in self._read_yaml_list(path):
            launchpad_id = item.get("current_launchpad_id")
            if launchpad_id not in launchpad_by_id:
                raise ValueError(f"Drone {item.get('drone_id')} references unknown launchpad {launchpad_id}")
            launchpad = launchpad_by_id[launchpad_id]
            drones.append(
                Drone(
                    vehicle_id=str(item["drone_id"]),
                    current_location=launchpad.current_location,
                    available_at=int(item.get("available_at", 0)),
                    capacity=1.0,
                    remaining_capacity=1.0,
                    speed_mps=float(
                        item.get("speed_mps", defaults.get("speed_mps", 50.0 / 3.6))
                    ),
                    battery_level=item.get("battery_level"),
                    battery_capacity=item.get("battery_capacity"),
                    max_range_m=item.get("max_range_m", defaults.get("max_range_m")),
                    empty_mass_kg=item.get("empty_mass_kg", defaults.get("empty_mass_kg")),
                    max_payload_kg=item.get("max_payload_kg", defaults.get("max_payload_kg")),
                    current_battery_id=item.get("current_battery_id"),
                    current_launchpad_id=str(launchpad_id),
                    status=VehicleStatus(str(item.get("status", "idle"))),
                )
            )
        return drones

    def _load_launchpads(self, path: Path) -> list[Launchpad]:
        launchpads: list[Launchpad] = []
        for item in self._read_yaml_list(path):
            launchpads.append(
                Launchpad(
                    launchpad_id=str(item["launchpad_id"]),
                    current_location=self._location_from_mapping(item["current_location"]),
                    drone_capacity=(
                        int(item["drone_capacity"])
                        if item.get("drone_capacity") is not None
                        else None
                    ),
                    status=LaunchpadStatus(str(item.get("status", "stationary"))),
                    reserved_return_slots=int(item.get("reserved_return_slots", 0)),
                    deployment_cost=float(item.get("deployment_cost", 0.0)),
                    op_cost_per_second=float(item.get("op_cost_per_second", 0.0)),
                    service_cost_per_event=float(item.get("service_cost_per_event", 0.0)),
                    active_time_s=float(item.get("active_time_s", 0.0)),
                    service_events=int(item.get("service_events", 0)),
                )
            )
        return launchpads

    def _load_batteries(self, path: Path) -> list[Battery]:
        batteries: list[Battery] = []
        capacity_override = self.config.get("batteries", {}).get("capacity_override_Wh")
        if capacity_override is not None and float(capacity_override) <= 0.0:
            raise ValueError("batteries.capacity_override_Wh must be positive")
        for item in self._read_yaml_list(path):
            source_capacity = float(item["capacity_Wh"])
            if source_capacity <= 0.0:
                raise ValueError(f"Battery {item.get('battery_id')} capacity_Wh must be positive")
            source_charge = float(item["current_charge_Wh"])
            capacity = (
                source_capacity if capacity_override is None else float(capacity_override)
            )
            soc = max(0.0, min(1.0, source_charge / source_capacity))
            current_charge = capacity * soc
            source_charge_start = item.get("charge_start_Wh")
            charge_start = (
                None
                if source_charge_start is None
                else capacity
                * max(0.0, min(1.0, float(source_charge_start) / source_capacity))
            )
            batteries.append(
                Battery(
                    battery_id=str(item["battery_id"]),
                    capacity_Wh=capacity,
                    current_charge_Wh=current_charge,
                    status=BatteryStatus(str(item["status"])),
                    current_launchpad_id=item.get("current_launchpad_id"),
                    current_vehicle_id=item.get("current_vehicle_id"),
                    charge_started_at=item.get("charge_started_at"),
                    charge_start_Wh=charge_start,
                    charge_complete_at=item.get("charge_complete_at"),
                )
            )
        return batteries

    def _bind_launchpad_drones(self, launchpads: list[Launchpad], drones: list[Drone]) -> None:
        launchpad_by_id = {launchpad.launchpad_id: launchpad for launchpad in launchpads}
        for launchpad in launchpads:
            launchpad.drone_ids = []
        for drone in drones:
            launchpad = launchpad_by_id.get(drone.current_launchpad_id or "")
            if launchpad is None:
                raise ValueError(f"UAV {drone.vehicle_id} references unknown launchpad {drone.current_launchpad_id}")
            launchpad.drone_ids.append(drone.vehicle_id)

    def _bind_batteries(
        self,
        launchpads: list[Launchpad],
        drone_by_id: dict[str, Drone],
        batteries: list[Battery],
    ) -> None:
        launchpad_by_id = {launchpad.launchpad_id: launchpad for launchpad in launchpads}
        battery_by_id = {battery.battery_id: battery for battery in batteries}
        for launchpad in launchpads:
            launchpad.charged_battery_ids = []
            launchpad.charging_battery_ids = []

        for drone in drone_by_id.values():
            battery_id = drone.current_battery_id
            if battery_id:
                battery = battery_by_id.get(battery_id)
                if battery is None:
                    raise ValueError(f"UAV {drone.vehicle_id} references unknown battery {battery_id}")
                if battery.status != BatteryStatus.IN_USE or battery.current_vehicle_id != drone.vehicle_id:
                    raise ValueError(
                        f"Battery {battery.battery_id} must be in_use and bound to UAV {drone.vehicle_id}"
                    )
                self._dock_initial_drone_battery(drone, battery)
            drone.current_battery_id = None
            drone.battery_level = None

        for battery in batteries:
            if battery.status == BatteryStatus.IN_USE:
                raise ValueError(
                    f"Battery {battery.battery_id} cannot be in_use while all UAVs start docked at launchpads"
                )
            if battery.current_launchpad_id not in launchpad_by_id:
                raise ValueError(f"Battery {battery.battery_id} references unknown launchpad {battery.current_launchpad_id}")
            launchpad = launchpad_by_id[battery.current_launchpad_id]
            if battery.status == BatteryStatus.CHARGED:
                launchpad.charged_battery_ids.append(battery.battery_id)
            elif battery.status == BatteryStatus.CHARGING:
                launchpad.charging_battery_ids.append(battery.battery_id)

    def _dock_initial_drone_battery(self, drone: Drone, battery: Battery) -> None:
        launchpad_id = drone.current_launchpad_id
        if launchpad_id is None:
            raise ValueError(f"UAV {drone.vehicle_id} has an onboard battery but is not at a launchpad")
        battery.current_vehicle_id = None
        battery.current_launchpad_id = launchpad_id
        battery.current_charge_Wh = max(0.0, min(battery.capacity_Wh, float(battery.current_charge_Wh)))
        if battery.current_charge_Wh >= battery.capacity_Wh - 1e-9:
            battery.status = BatteryStatus.CHARGED
            battery.current_charge_Wh = battery.capacity_Wh
            battery.charge_started_at = None
            battery.charge_start_Wh = None
            battery.charge_complete_at = None
            return

        own = getattr(battery, "full_charge_seconds", None)
        full_charge_seconds = max(1, int(own)) if own else max(1, self._battery_charge_time_seconds())
        deficit_ratio = (battery.capacity_Wh - battery.current_charge_Wh) / max(1e-9, battery.capacity_Wh)
        battery.status = BatteryStatus.CHARGING
        battery.charge_started_at = 0
        battery.charge_start_Wh = battery.current_charge_Wh
        battery.charge_complete_at = max(1, int(full_charge_seconds * deficit_ratio))

    def _battery_charge_time_seconds(self) -> int:
        service_times = self.config.get("service_times", {})
        launchpad_service = self.config.get("launchpads", {}).get("service", {})
        return int(
            service_times.get(
                "battery_charge_seconds",
                launchpad_service.get("battery_charge_seconds", 1800),
            )
        )

    def _location_from_mapping(self, data: dict) -> Location:
        return location_from_mapping(data, road_network=self.road_network, mode="drive")

    def _read_yaml_list(self, path: Path, *, allow_empty: bool = False) -> list[dict]:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list) or (not data and not allow_empty):
            qualifier = "a YAML list" if allow_empty else "a non-empty YAML list"
            raise ValueError(f"Instance file must be {qualifier}: {path}")
        return data

    def _read_yaml_mapping(self, path: Path) -> dict:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Instance file must be a YAML mapping: {path}")
        return data
