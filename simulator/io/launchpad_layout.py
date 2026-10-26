"""Read the frozen launchpad layout, so a run does not re-derive its own pads.

``data/dataset/order_instance/make_launchpads.py`` writes one file per
(city, extent, pitch, flight mask). It holds the pads that the train split
actually uses, their coordinates, and the train demand each carries. Reading it
rather than recomputing it means a layout change is a diff in a tracked file,
not a silent drift when the grid code or the cached city box is regenerated.

The file also carries the city frame it was built in, so a run that uses a layout
depends on that one file and not on ``city_extent.json`` as well.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .city_orders import DEFAULT_WINDOW_SECONDS, DATASET_ROOT, CityFrame

LAYOUT_DIRNAME = "launchpads"
# v2: the name and the record carry the episode's time window, because the demand
# a pad carries -- and therefore which pads survive pruning -- depends on it
LAYOUT_VERSION = 2


@dataclass(frozen=True)
class LaunchpadLayout:
    """A frozen pad set, in the frame it was measured in."""

    city: str
    extent: str
    pitch_km: tuple[float, float]
    pad_grid: tuple[int, int]
    flight_mask_km: tuple[float, float]
    demand_split: str
    demand_days: int
    window_seconds: tuple[int, int] | None
    clip_box_m: tuple[float, float, float, float] | None   # (x_lo, x_hi, y_lo, y_hi)
    n_candidate_pads: int
    frame: CityFrame
    anchor: tuple[float, float]
    pad_ids: list[str]
    centres: np.ndarray            # (n, 2) metres in `frame`
    train_orders: np.ndarray       # (n,) orders over the whole demand split
    offered_load_s: np.ndarray     # (n,) drone-seconds over the whole demand split
    source: Path

    @property
    def n_pads(self) -> int:
        return len(self.pad_ids)


def parse_window(spec) -> tuple[int, int] | None:
    """The one parser for an episode window, so a layout, a fleet size and a run
    cannot disagree about what "8-12" means.

    Accepts ``"8-12"`` (hours since the source day origin), ``[28800, 43200]``
    (seconds), and ``"all"`` / ``None`` for the whole day.
    """
    if spec is None or (isinstance(spec, str) and spec.lower() == "all"):
        return None
    if isinstance(spec, (list, tuple)):
        return (int(spec[0]), int(spec[1]))
    start, _, end = str(spec).partition("-")
    return (int(start) * 3600, int(end) * 3600)


def window_tag(window: tuple[int, int] | None) -> str:
    """``t10-12`` for a whole-hour window, ``t36000-43200s`` otherwise, ``tall``."""
    if window is None:
        return "tall"
    if window[0] % 3600 == 0 and window[1] % 3600 == 0:
        return f"t{window[0] // 3600}-{window[1] // 3600}"
    return f"t{window[0]}-{window[1]}s"


def layout_name(
    city, extent, spacing_km, min_km, max_km, *,
    window=DEFAULT_WINDOW_SECONDS, version=LAYOUT_VERSION,
) -> str:
    return (
        f"{city}__{extent}_s{spacing_km:g}km_{min_km:g}-{max_km:g}km"
        f"_{window_tag(window)}__v{version}.json"
    )


def layout_path(
    city, extent, spacing_km, min_km, max_km, *, window=DEFAULT_WINDOW_SECONDS, root=None,
) -> Path:
    return Path(root or DATASET_ROOT) / LAYOUT_DIRNAME / layout_name(
        city, extent, spacing_km, min_km, max_km, window=window
    )


def load_layout(
    city: str,
    *,
    extent: str,
    spacing_km: float,
    min_km: float,
    max_km: float,
    demand_split: str = "train",
    window: tuple[int, int] | None = DEFAULT_WINDOW_SECONDS,
    root: str | None = None,
) -> LaunchpadLayout | None:
    """The frozen layout, or None when none has been written for this combination.

    Returning None rather than raising lets a parameter sweep explore a pitch
    nobody has frozen yet; the caller says in its report which path it took.
    """
    path = layout_path(city, extent, spacing_km, min_km, max_km, window=window, root=root)
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if record.get("demand_split") != demand_split:
        raise ValueError(
            f"{path} was measured on the {record.get('demand_split')!r} split, "
            f"but demand_split={demand_split!r} was asked for; regenerate with "
            f"make_launchpads.py --split {demand_split}"
        )
    pads = record["pads"]
    if not pads:
        raise ValueError(f"{path} holds no pad")
    frame_spec = record["frame"]
    return LaunchpadLayout(
        city=record["city"],
        extent=record["extent"],
        pitch_km=tuple(record["pitch_km_realised"]),
        pad_grid=tuple(record["pad_grid"]),
        flight_mask_km=tuple(record["flight_mask_km"]),
        demand_split=record["demand_split"],
        demand_days=int(record["demand_days"]),
        window_seconds=(
            None if record.get("window_seconds") is None else tuple(record["window_seconds"])
        ),
        clip_box_m=(
            None if record.get("clip_box_m") is None else tuple(record["clip_box_m"])
        ),
        n_candidate_pads=int(record["n_candidate_pads"]),
        frame=CityFrame(
            planar=bool(frame_spec["planar"]),
            origin_x=float(frame_spec["origin_x"]),
            origin_y=float(frame_spec["origin_y"]),
            cos_lat=float(frame_spec["cos_lat"]),
        ),
        anchor=tuple(record["anchor_lat_lon"]),
        pad_ids=[str(p["id"]) for p in pads],
        centres=np.array([[float(p["east_m"]), float(p["north_m"])] for p in pads]),
        train_orders=np.array([float(p["train_orders"]) for p in pads]),
        offered_load_s=np.array([float(p["train_offered_load_s"]) for p in pads]),
        source=path,
    )


def hourly_profile(path: Path) -> tuple[list[str], np.ndarray, list[int]]:
    """Pad ids, a (pads x hours) order matrix and the hour labels, for plotting."""
    record = json.loads(Path(path).read_text())
    pads = record["pads"]
    hours = sorted({int(h) for p in pads for h in p["train_orders_by_hour"]})
    matrix = np.zeros((len(pads), len(hours)))
    position = {h: i for i, h in enumerate(hours)}
    for row, pad in enumerate(pads):
        for hour, count in pad["train_orders_by_hour"].items():
            matrix[row, position[int(hour)]] = count
    return [str(p["id"]) for p in pads], matrix, hours
