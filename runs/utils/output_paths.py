from __future__ import annotations

from datetime import datetime
from pathlib import Path


def slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def city_key(config: dict) -> str:
    instance_cfg = config.get("instance", {})
    configured_city = instance_cfg.get("city")
    if configured_city:
        return slug(str(configured_city))

    place_name = config.get("map", {}).get("place_name", "unknown_city")
    return slug(str(place_name).split(",")[0])


def drone_mode_key(config: dict) -> str:
    return "drone_only"


def time_advancement_key(config: dict) -> str:
    return slug(str(config.get("simulation", {}).get("time_advancement", "fixed_step")))


def run_output_dir(
    config: dict,
    *,
    split: str,
    algorithm: str,
    seed: int | None = None,
    record_dir: str | Path | None = None,
    run_stamp: str | None = None,
) -> Path:
    base_dir = Path(record_dir or config.get("output", {}).get("record_dir", "records"))
    run_folder = run_stamp or timestamp()
    if seed is not None:
        run_folder = f"{run_folder}_seed{seed}"
    return (
        base_dir
        / slug(split)
        / city_key(config)
        / drone_mode_key(config)
        / time_advancement_key(config)
        / algorithm
        / run_folder
    )


def explicit_run_output_dir(
    *,
    records_root: str | Path,
    split: str,
    city: str,
    drone_mode: str,
    time_advance_mode: str,
    algorithm: str,
    seed: int | None = None,
    run_stamp: str | None = None,
) -> Path:
    run_folder = run_stamp or timestamp()
    if seed is not None:
        run_folder = f"{run_folder}_seed{seed}"
    return (
        Path(records_root)
        / slug(split)
        / slug(city)
        / slug(drone_mode)
        / slug(time_advance_mode)
        / algorithm
        / run_folder
    )
