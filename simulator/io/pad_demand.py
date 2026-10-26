"""Per-launchpad demand weights, for placing a fleet where the orders are.

A uniform ``drones_per_launchpad`` wastes a fleet on a demand-blind grid: with a
2 km pitch, 159 of shenzhen's 432 pads never see an order, and the busiest pad
carries 58 times the offered load of the median one. Placing drones in
proportion to demand fixes that without changing the pad layout.

The weight is **offered load**, not order count: the drone-seconds a pad's orders
consume, ``sum(2 * d / v)`` over the orders that bind to it and survive the
flight mask. That is the quantity a pad needs drones for -- a pad with few but
long orders ties up as much of the fleet as one with many short ones.

Two things make this safe to use as a design decision rather than a leak:

* the weights come from the **train** split by default, so an instance built on a
  test day is not placed using that day's own demand;
* they are a property of (city, grid, mask), not of the episode, so every episode
  of a run shares one placement and a transfer result is not comparing two
  different fleets.

Computing them reads every day of the split, which is slow enough to be worth
caching: the result is memoised in process and written under
``pad_demand/`` beside the order instances. Pre-warm with

    python data/dataset/order_instance/make_pad_demand.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .city_orders import DEFAULT_WINDOW_SECONDS, DATASET_ROOT, CityFrame, day_files, load_day

# bump when the weight definition changes, so stale caches are not reused
# v2: the weights are measured over the episode's time window (10:00-12:00 by
# default), not over the whole 08:00-12:00 file
DEMAND_VERSION = 2
CACHE_DIRNAME = "pad_demand"

_MEMO: dict[tuple, np.ndarray] = {}


def _cache_path(city: str, grid_key: str, split: str, root: str) -> Path:
    name = f"{city}__{grid_key}__{split}__v{DEMAND_VERSION}.json"
    return Path(root) / CACHE_DIRNAME / name


def _nearest(points: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Index of the nearest centre to each point, in memory-bounded chunks."""
    out = np.empty(len(points), dtype=np.int32)
    step = max(1, 4_000_000 // max(1, len(centres)))
    for start in range(0, len(points), step):
        chunk = points[start : start + step]
        out[start : start + step] = np.argmin(
            ((chunk[:, None, :] - centres[None, :, :]) ** 2).sum(-1), axis=1
        )
    return out


def compute_offered_load(
    city: str,
    centres: np.ndarray,
    frame: CityFrame,
    *,
    min_km: float,
    max_km: float,
    speed_mps: float,
    split: str = "train",
    root: str | None = None,
    window: tuple[int, int] | None = DEFAULT_WINDOW_SECONDS,
) -> np.ndarray:
    """Drone-seconds per pad over a whole split. No caching; see ``offered_load``.

    ``window`` is the episode's slice of the source day: the weights describe the
    hours a run actually plays, so a pad busy only in the morning does not earn
    drones for an afternoon episode.
    """
    kwargs = {} if root is None else {"root": root}
    weights = np.zeros(len(centres))
    for day in range(len(day_files(city, split, **kwargs))):
        raw, _ = load_day(city, split, day, frame=frame, window=window, **kwargs)
        if not raw:
            continue
        delivery = np.array([o.delivery_m for o in raw], dtype=float)
        anchor = np.array(
            [o.pickup_m if o.pickup_m is not None else o.delivery_m for o in raw], dtype=float
        )
        index = _nearest(anchor, centres)
        legs_m = np.linalg.norm(centres[index] - delivery, axis=1)
        keep = (legs_m >= min_km * 1000.0) & (legs_m <= max_km * 1000.0)
        # the round trip is what occupies a drone; delivery_seconds is 0 and the
        # engine charges no swap time, so this is the whole occupancy
        np.add.at(weights, index[keep], 2.0 * legs_m[keep] / speed_mps)
    return weights


def offered_load(
    city: str,
    centres: np.ndarray,
    frame: CityFrame,
    *,
    grid_key: str,
    min_km: float,
    max_km: float,
    speed_mps: float,
    split: str = "train",
    root: str | None = None,
    window: tuple[int, int] | None = DEFAULT_WINDOW_SECONDS,
    use_cache: bool = True,
) -> np.ndarray:
    """``compute_offered_load``, memoised in process and on disk.

    ``grid_key`` must identify the pad layout the weights were measured against
    -- a cache entry read back for a different layout would be meaningless, and
    the length check below is the only thing that would catch it.
    """
    root = root or DATASET_ROOT
    memo_key = (city, grid_key, split, float(min_km), float(max_km), float(speed_mps), window, root)
    if use_cache and memo_key in _MEMO:
        return _MEMO[memo_key]

    path = _cache_path(city, grid_key, split, root)
    if use_cache and path.is_file():
        record = json.loads(path.read_text())
        weights = np.asarray(record["offered_load_seconds"], dtype=float)
        if len(weights) == len(centres):
            _MEMO[memo_key] = weights
            return weights
        # a stale entry from a different layout under the same key: recompute
    weights = compute_offered_load(
        city,
        centres,
        frame,
        min_km=min_km,
        max_km=max_km,
        speed_mps=speed_mps,
        split=split,
        root=root,
        window=window,
    )
    if use_cache:
        _write_cache(path, city, grid_key, split, min_km, max_km, speed_mps, window, weights)
        _MEMO[memo_key] = weights
    return weights


def _write_cache(path, city, grid_key, split, min_km, max_km, speed_mps, window, weights) -> None:
    """Write atomically, so two runs starting together cannot read half a file."""
    record = {
        "version": DEMAND_VERSION,
        "city": city,
        "grid_key": grid_key,
        "split": split,
        "min_flight_km": min_km,
        "max_flight_km": max_km,
        "speed_mps": speed_mps,
        "window_seconds": None if window is None else list(window),
        "n_pads": len(weights),
        "offered_load_seconds": [round(float(w), 3) for w in weights],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name, suffix=".tmp", delete=False
    )
    try:
        with handle:
            json.dump(record, handle)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def allocate(weights: np.ndarray, total: int, *, minimum: int = 0) -> np.ndarray:
    """Split ``total`` units across pads in proportion to ``weights``.

    Largest-remainder rounding, so the parts sum to ``total`` exactly rather than
    drifting with the pad count. ``minimum`` is a floor applied to the pads that
    carry any demand at all; a pad with zero demand is left at zero, because a
    drone parked where no order ever arrives cannot fly and -- with
    ``drone_decision.rebalance.enabled`` false, the default -- cannot leave.
    """
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    n = len(weights)
    active = weights > 0.0
    counts = np.zeros(n, dtype=int)
    if total == 0 or not active.any():
        return counts

    floor = min(int(minimum), total // max(1, int(active.sum())))
    counts[active] = floor
    remaining = total - int(counts.sum())
    if remaining <= 0:
        return counts

    share = weights / weights.sum() * remaining
    counts += np.floor(share).astype(int)
    short = remaining - int(np.floor(share).sum())
    if short > 0:
        # the largest fractional parts take the leftovers; ties break on pad index
        order = np.argsort(-(share - np.floor(share)), kind="stable")
        counts[order[:short]] += 1
    return counts
