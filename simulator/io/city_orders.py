"""Read one day of orders from ``data/dataset/order_instance`` and put it in metres.

The dataset holds eight cities from five sources, and they do not share a
coordinate system. Seven are ``(longitude, latitude)`` in degrees -- real for
LaDe and Tianchi, privacy-shifted for the Meituan cities, but degrees either way
so the same projection applies. Grubhub is anonymised planar metres with the UTM
zone discarded, which the city manifest marks as ``coordinate_system:
planar_m``. Reading the manifest rather than guessing from the magnitude of the
numbers is the point of this module: Grubhub's x runs to 12281, which a
degree-based reader would silently accept and then place every order on the far
side of the planet.

Everything leaves here as metres in a local equirectangular frame centred on the
city, which is what the launchpad grid, the order binding and the 1-10 km mask
all work in. The frame is per city and never compared across cities.

Missing values are ``-1`` (a scalar) or a tuple containing ``-1``. Every LaDe
city has no pickup location at all -- 100% missing -- which is why the launchpad
binding needs a delivery-side fallback rather than treating a missing pickup as
a broken row.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

EARTH_RADIUS_M = 6371000.0
DATASET_ROOT = "data/dataset/order_instance"
EXTENT_FILE = "city_extent.json"
# The default trim level for the city box. Not "full": the raw min/max box is
# outlier-driven, and launchpads are laid over this box, so a box inflated by a
# handful of stray rows spends most of its pads on empty ground -- meituanC's raw
# box is 29.4 km across against 10.6 km trimmed, and only 40 of the 225 pads a
# 2 km pitch puts in the raw box ever see an order. "q01" clips each axis to its
# 1st-99th percentile, which keeps about 96% of the points inside (two axes, two
# tails each). Orders are never dropped by this: the box only places pads, and an
# order outside it binds to an edge pad.
#
# Consequence worth knowing: the "full" box is what
# data/dataset/visualization/plot_city_statistics.py publishes as map_width_km /
# map_height_km, so the default box is deliberately *not* the published city size.
DEFAULT_EXTENT_TRIM = "q01"

# The slice of the source day one episode plays out on, in seconds since the
# source day origin (the instances encode 08:00 as 28800). The published CSVs
# hold 08:00-12:00, but demand inside that window climbs steeply and peaks in the
# last hour: 11:00 carries roughly twice the mean hour, so a fleet sized on the
# four-hour average is swamped at the peak while the morning sits nearly idle.
# Running the peak hour alone keeps the hardest part of the day -- about half the
# orders of the 10:00-12:00 window -- at a quarter of the simulated steps of the
# original four hours, and makes the fleet sizing a statement about the regime
# the episode is actually in rather than about an average it never visits.
DEFAULT_WINDOW_SECONDS = (39600, 43200)   # 11:00 - 12:00


@dataclass(frozen=True)
class CityFrame:
    """The local metre frame for one city."""

    planar: bool
    origin_x: float          # degrees of longitude, or metres
    origin_y: float          # degrees of latitude, or metres
    cos_lat: float           # longitude foreshortening at the city's mean latitude

    def to_metres(self, x: float, y: float) -> tuple[float, float]:
        if self.planar:
            return float(x) - self.origin_x, float(y) - self.origin_y
        return (
            math.radians(float(x) - self.origin_x) * EARTH_RADIUS_M * self.cos_lat,
            math.radians(float(y) - self.origin_y) * EARTH_RADIUS_M,
        )

    def to_source(self, east_m: float, north_m: float) -> tuple[float, float]:
        """Inverse of ``to_metres``, so a launchpad can be reported in city units."""
        if self.planar:
            return east_m + self.origin_x, north_m + self.origin_y
        return (
            self.origin_x + math.degrees(east_m / (EARTH_RADIUS_M * self.cos_lat)),
            self.origin_y + math.degrees(north_m / EARTH_RADIUS_M),
        )


@dataclass
class RawOrder:
    order_id: str
    created_at: int                      # seconds on the source day axis
    deadline: int
    delivered_at: int | None
    pickup_m: tuple[float, float] | None   # None when the source has no pickup
    delivery_m: tuple[float, float]


def _in_window(text: str | None, window: tuple[int, int]) -> bool:
    """Is this row's create_time inside ``[start, end)`` on the source day axis?

    A row with no usable arrival time is out: it has no place on the clock, and
    ``load_day`` would drop it a few lines later anyway.
    """
    if text is None:
        return False
    try:
        created = int(float(text))
    except (TypeError, ValueError):
        return False
    return window[0] <= created < window[1]


def _parse_point(text: str | None) -> tuple[float, float] | None:
    """A ``"(x, y)"`` cell, or None when the source marks it missing."""
    if text is None:
        return None
    raw = text.strip()
    if raw in {"", "-1", "nan", "None"}:
        return None
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    x, y = float(value[0]), float(value[1])
    if x == -1.0 or y == -1.0:
        return None
    return x, y


def city_dir(city: str, *, root: str = DATASET_ROOT) -> Path:
    path = Path(root) / city
    if not path.is_dir():
        available = sorted(p.name for p in Path(root).iterdir() if p.is_dir()) if Path(root).is_dir() else []
        raise FileNotFoundError(f"No order instances for city {city!r} under {root}; have {available}")
    return path


def day_files(city: str, split: str, *, root: str = DATASET_ROOT) -> list[Path]:
    """The day CSVs of one split, in day order."""
    directory = city_dir(city, root=root) / split
    if not directory.is_dir():
        raise FileNotFoundError(f"City {city!r} has no split {split!r} at {directory}")
    files = sorted(directory.glob("day*.csv"), key=lambda p: int(p.stem.removeprefix("day")))
    if not files:
        raise FileNotFoundError(f"No day*.csv under {directory}")
    return files


def city_is_planar(city: str, *, root: str = DATASET_ROOT) -> bool:
    manifest = json.loads((city_dir(city, root=root) / "manifest.json").read_text())
    return manifest.get("source_metadata", {}).get("coordinate_system") == "planar_m"


def city_extent(
    city: str,
    *,
    root: str = DATASET_ROOT,
    trim: str = DEFAULT_EXTENT_TRIM,
) -> dict:
    """The city's cached bounding box, in source units.

    Produced by ``data/dataset/order_instance/make_city_extent.py`` over every day
    of every split, so it does not move when the day or the split changes.
    ``trim="full"`` is the raw min/max box, which reproduces the published
    ``map_width_km`` / ``map_height_km``; the other levels drop the outer
    percentile on each axis.
    """
    path = Path(root) / EXTENT_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"No city extent cache at {path}; run "
            "python data/dataset/order_instance/make_city_extent.py"
        )
    table = json.loads(path.read_text())
    if city not in table:
        raise KeyError(f"City {city!r} is not in {path}; have {sorted(table)}")
    record = table[city]
    if trim not in record:
        offered = sorted(k for k, v in record.items() if isinstance(v, dict))
        raise KeyError(f"Extent trim {trim!r} is not cached for {city!r}; have {offered}")
    box = dict(record[trim])
    box["planar"] = bool(record["planar"])
    return box


def city_frame(
    city: str,
    *,
    root: str = DATASET_ROOT,
    trim: str = DEFAULT_EXTENT_TRIM,
) -> tuple[CityFrame, tuple[float, float], tuple[float, float]]:
    """A metre frame centred on the city, plus that city's box in it.

    The frame is a property of the city and not of one day, which is what makes a
    launchpad grid reproducible: the same pad index lands on the same coordinate
    in train and in test. Returns the frame and the box as ``(x_lo, x_hi)``,
    ``(y_lo, y_hi)`` in metres.
    """
    box = city_extent(city, root=root, trim=trim)
    (x_lo, x_hi), (y_lo, y_hi) = box["x"], box["y"]
    mid_x, mid_y = (x_lo + x_hi) / 2.0, (y_lo + y_hi) / 2.0
    frame = CityFrame(
        planar=box["planar"],
        origin_x=mid_x,
        origin_y=mid_y,
        # taken at the box's mid latitude, the same convention the published
        # width uses, so the frame's metres and that table agree
        cos_lat=1.0 if box["planar"] else max(1e-6, math.cos(math.radians(mid_y))),
    )
    lo = frame.to_metres(x_lo, y_lo)
    hi = frame.to_metres(x_hi, y_hi)
    return frame, (lo[0], hi[0]), (lo[1], hi[1])


def load_day(
    city: str,
    split: str,
    day: int,
    *,
    root: str = DATASET_ROOT,
    frame: CityFrame | None = None,
    window: tuple[int, int] | None = DEFAULT_WINDOW_SECONDS,
) -> tuple[list[RawOrder], CityFrame]:
    """One day's orders in a local metre frame, plus that frame.

    Rows without a usable delivery location are dropped -- the delivery point is
    the one coordinate the simulator cannot do without. Rows without a pickup are
    kept with ``pickup_m=None`` for the binding step to resolve.

    ``frame`` overrides the day-centred frame, which is what a caller placing a
    city-level launchpad grid wants: the orders and the pads then share one frame
    that is the same on every day.

    ``window`` keeps only the orders created in ``[start, end)`` seconds on the
    source day axis; ``None`` keeps the whole file. The filter runs before the
    day-centred frame is measured, so the frame describes the orders an episode
    will actually see.
    """
    files = day_files(city, split, root=root)
    if not 0 <= day < len(files):
        raise IndexError(f"{city}/{split} has {len(files)} days; day={day} is out of range")
    planar = city_is_planar(city, root=root)

    rows = list(csv.DictReader(files[day].open()))
    points: list[tuple[float, float]] = []
    parsed: list[tuple[dict, tuple[float, float] | None, tuple[float, float]]] = []
    for row in rows:
        delivery = _parse_point(row.get("delivery_location"))
        if delivery is None:
            continue
        if window is not None and not _in_window(row.get("create_time"), window):
            continue
        pickup = _parse_point(row.get("pickup_location"))
        parsed.append((row, pickup, delivery))
        points.append(delivery)
        if pickup is not None:
            points.append(pickup)
    if not parsed:
        raise ValueError(
            f"{files[day]} has no row with a delivery location"
            + ("" if window is None else f" created in [{window[0]}, {window[1]}) s")
        )

    if frame is None:
        # the frame is centred on the day's own points, so x/y stay small and the
        # longitude foreshortening is taken at a latitude the city actually occupies
        mean_x = sum(p[0] for p in points) / len(points)
        mean_y = sum(p[1] for p in points) / len(points)
        frame = CityFrame(
            planar=planar,
            origin_x=mean_x,
            origin_y=mean_y,
            cos_lat=1.0 if planar else max(1e-6, math.cos(math.radians(mean_y))),
        )
    elif frame.planar != planar:
        raise ValueError(
            f"{city} is {'planar' if planar else 'degree'}-coordinated but the supplied "
            f"frame is {'planar' if frame.planar else 'degree'}-coordinated"
        )

    def as_int(value: str | None) -> int | None:
        if value is None or str(value).strip() in {"", "-1", "nan", "None"}:
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    out: list[RawOrder] = []
    for row, pickup, delivery in parsed:
        created = as_int(row.get("create_time"))
        if created is None:
            continue                      # an order with no arrival time has no place on the clock
        promised = as_int(row.get("promised_delivery_time"))
        out.append(
            RawOrder(
                order_id=str(row["order_id"]),
                created_at=created,
                # keep the source's own promise when it has one; the scenario
                # builder supplies a fallback window when it does not
                deadline=promised if promised is not None else -1,
                delivered_at=as_int(row.get("record_delivery_time")),
                pickup_m=None if pickup is None else frame.to_metres(*pickup),
                delivery_m=frame.to_metres(*delivery),
            )
        )
    out.sort(key=lambda o: o.created_at)
    return out, frame
