from __future__ import annotations

import argparse
from pathlib import Path

import yaml


# Historical v2 calibration retained for reproducibility. The active simulator
# and new generator defaults are defined by drone_only_v3_3kw_720wh.
SPEED_MPS = 50.0 / 3.6
MAX_RANGE_M = 24_000.0
EMPTY_MASS_KG = 15.0
MAX_PAYLOAD_KG = 10.0
BATTERY_CAPACITY_WH = 1440.0
BATTERY_CHARGE_SECONDS = 3600
EMPTY_POWER_W = 1800.0
PAYLOAD_POWER_SLOPE_W_PER_KG = 70.0
RESERVE_RATIO = 0.20
OLD_WEIGHT_MIN_KG = 0.2
OLD_WEIGHT_MAX_KG = 2.27
NEW_WEIGHT_MIN_KG = 0.2
DATASET_VERSION = "drone_only_v2"
PHYSICAL_MODEL_VERSION = "drone_only_v2_60min_charge"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate saved drone-only demand instances to physical model v2."
    )
    parser.add_argument("--root", type=Path, default=Path("data/instances_drone"))
    return parser.parse_args()


def read_yaml(path: Path):
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def write_yaml(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=False)
    temporary.replace(path)


def migrated_weight(weight: float) -> float:
    quantile = (float(weight) - OLD_WEIGHT_MIN_KG) / (
        OLD_WEIGHT_MAX_KG - OLD_WEIGHT_MIN_KG
    )
    if quantile < -1e-6 or quantile > 1.0 + 1e-6:
        raise ValueError(f"legacy order weight {weight} is outside expected range")
    quantile = max(0.0, min(1.0, quantile))
    return NEW_WEIGHT_MIN_KG + quantile * (MAX_PAYLOAD_KG - NEW_WEIGHT_MIN_KG)


def physical_model() -> dict[str, float | str]:
    return {
        "version": PHYSICAL_MODEL_VERSION,
        "empty_vehicle_and_battery_mass_kg": EMPTY_MASS_KG,
        "max_payload_kg": MAX_PAYLOAD_KG,
        "speed_kmh": 50.0,
        "speed_mps": SPEED_MPS,
        "max_range_m": MAX_RANGE_M,
        "battery_capacity_Wh": BATTERY_CAPACITY_WH,
        "empty_power_W": EMPTY_POWER_W,
        "payload_power_slope_W_per_kg": PAYLOAD_POWER_SLOPE_W_PER_KG,
        "full_payload_power_W": (
            EMPTY_POWER_W + PAYLOAD_POWER_SLOPE_W_PER_KG * MAX_PAYLOAD_KG
        ),
        "battery_reserve_ratio": RESERVE_RATIO,
        "battery_charge_seconds": BATTERY_CHARGE_SECONDS,
        "order_weight_migration": "affine_U(0.2,2.27)_to_U(0.2,10)",
    }


def migrate_instance(path: Path) -> tuple[Path, bool]:
    metadata_path = path / "metadata.yaml"
    metadata = read_yaml(metadata_path)
    existing_physical = metadata.get("physical_model", {})
    if (
        existing_physical.get("version") == PHYSICAL_MODEL_VERSION
        and int(existing_physical.get("battery_charge_seconds", 0))
        == BATTERY_CHARGE_SECONDS
    ):
        return path, False

    already_v2 = (
        str(metadata.get("version", "")).startswith(DATASET_VERSION)
        or "_drone_only_v2_" in path.name
    )

    drones_path = path / "drones.yaml"
    drones = read_yaml(drones_path)
    for drone in drones:
        drone["speed_mps"] = SPEED_MPS
        drone["max_range_m"] = MAX_RANGE_M
        drone["empty_mass_kg"] = EMPTY_MASS_KG
        drone["max_payload_kg"] = MAX_PAYLOAD_KG
    write_yaml(drones_path, drones)

    orders_path = path / "orders.yaml"
    orders = read_yaml(orders_path) or []
    if not already_v2:
        for order in orders:
            order["weight"] = migrated_weight(float(order["weight"]))
        write_yaml(orders_path, orders)

    simulator_config_path = path / "simulator_config.yaml"
    simulator_config = read_yaml(simulator_config_path) or {}
    uav = simulator_config.setdefault("vehicles", {}).setdefault("uavs", {})
    uav.update(
        {
            "speed_mps": SPEED_MPS,
            "max_range_m": MAX_RANGE_M,
            "empty_mass_kg": EMPTY_MASS_KG,
            "max_payload_kg": MAX_PAYLOAD_KG,
            "battery_capacity": BATTERY_CAPACITY_WH,
        }
    )
    simulator_config.setdefault("service_times", {})[
        "battery_charge_seconds"
    ] = BATTERY_CHARGE_SECONDS
    simulator_config["energy_model"] = {
        "uav": {
            "mode": "paper",
            "speed_mps": SPEED_MPS,
            "mass_power_intercept_W": EMPTY_POWER_W,
            "mass_power_slope_W_per_kg": PAYLOAD_POWER_SLOPE_W_PER_KG,
            "battery_capacity_Wh": BATTERY_CAPACITY_WH,
            "max_payload_kg": MAX_PAYLOAD_KG,
            "reserve_ratio": RESERVE_RATIO,
        }
    }
    write_yaml(simulator_config_path, simulator_config)

    old_id = str(metadata.get("instance_id", path.name))
    new_id = old_id.replace("_drone_only_v1_", "_drone_only_v2_")
    metadata.setdefault("source_instance_id", old_id)
    metadata["instance_id"] = new_id
    metadata["version"] = DATASET_VERSION
    metadata["physical_model"] = physical_model()
    write_yaml(metadata_path, metadata)

    destination = path.with_name(new_id)
    if destination != path:
        if destination.exists():
            raise FileExistsError(f"migration target already exists: {destination}")
        path.rename(destination)
    return destination, True


def main() -> None:
    args = parse_args()
    required = {"metadata.yaml", "orders.yaml", "drones.yaml", "simulator_config.yaml"}
    instances = sorted(
        path
        for path in args.root.glob("*/*/*")
        if path.is_dir() and all((path / name).is_file() for name in required)
    )
    if not instances:
        raise FileNotFoundError(f"no complete instances found under {args.root}")
    migrated = 0
    skipped = 0
    for index, path in enumerate(instances, start=1):
        _, changed = migrate_instance(path)
        if changed:
            migrated += 1
        else:
            skipped += 1
        if index % 50 == 0 or index == len(instances):
            print(f"processed {index}/{len(instances)}", flush=True)
    print(f"migration complete: migrated={migrated}, skipped={skipped}")


if __name__ == "__main__":
    main()
