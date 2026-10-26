"""Build a simulator instance from a city name and four counts, with no YAML.

    build_scenario(city, n_launchpads, drones_per_launchpad, batteries_per_drone,
                   n_battery_types)

The earlier pipeline baked a fleet into every instance folder on disk, so trying
a different launchpad count or a different battery mix meant regenerating 100
instance directories. Here the order stream is the only thing read from disk and
the fleet is constructed, so the four counts are experiment parameters.

What the counts mean, and what they do not
------------------------------------------
``n_launchpads``          pads placed on a uniform grid over the city's extent.
                          May be ``None``, in which case ``pad_spacing_km`` sets
                          the pitch and the count follows from the city's size.
``drones_per_launchpad``  drones parked at each pad. Total = pads x this.
``batteries_per_drone``   packs *per drone*, so the total is pads x drones x this.
                          They are placed at the pad where those drones start but
                          are **not bound to a drone**: every pack begins loose on
                          the pad, charged, available to any drone there. A drone
                          mounts one only when it is dispatched.
``n_battery_types``       rungs of the heterogeneous ladder (see
                          ``model/battery_types``). 1 reproduces the homogeneous
                          720 Wh fleet. Types are dealt round-robin *within* each
                          pad, so every pad holds the same mix rather than pad 0
                          getting all the big packs.

Launchpad placement
-------------------
A uniform grid over a bounding box, with the grid aspect chosen to match the box,
which is what "evenly distributed on the map" means geometrically. It is
deliberately demand-blind: a k-means placement would put pads where the orders
are and quietly couple the pad count to the demand pattern, and then a pad sweep
would be measuring two things at once.

Two things are chosen: which box, and how finely it is divided.

*Which box* is ``extent``. It is the **city** box cached by
``data/dataset/order_instance/make_city_extent.py`` over every day of every
split, so pad ``lp_7`` is the same coordinate in train and in test; without that
a transfer result would be comparing two different pad layouts.

The default is ``"q01"``, each axis clipped to its 1st-99th percentile, and not
``"full"``, the raw min/max box. The raw box is outlier-driven and pads are laid
over the box, so the raw box spends its pads on empty ground: meituanC measures
29.4 km across raw against 10.6 km trimmed, and of the 225 pads a 2 km pitch puts
in the raw box only 40 ever see an order. Trimming drops no orders -- the box
places pads, and an order outside it binds to an edge pad -- it only stops the
grid from covering ground no order uses. ``"q005"`` and ``"q025"`` are the other
cached levels, and ``extent="day"`` restores the day-local box.

Note that ``"full"``, not the default, is what
``plot_city_statistics`` publishes as ``map_width_km`` / ``map_height_km``, so the
box a run uses is deliberately smaller than the city size that table reports.

*How finely* is exactly one of two parameters. ``pad_spacing_km`` fixes the pitch
and lets the count follow: one pad every ``pad_spacing_km`` along each axis,
rounded to a whole number of pads, so ``n = cols * rows`` and the realised pitch
is the box divided by that. ``n_launchpads`` fixes the count and lets the pitch
follow; when the grid cannot be exactly ``n`` cells the trailing cells of the last
row are dropped, so the last row may be short.

The pitch interacts with the flight mask, and the direction depends on whether
the city has pickup locations at all. Where it does, the pad sits near the pickup
and the flown leg is about the real order distance, so retention barely moves
between a 1 km and a 3 km pitch. Where it does not -- every LaDe city -- the pad
is the nearest one to the *delivery*, so the flown leg *is* the distance to the
nearest pad: a fine grid puts every delivery within 1 km of a pad and the floor
masks the city away. At a 2 km pitch that costs the LaDe cities 73-82% of their
orders, and they retain best at a pitch near 10 km. See the ``pad_spacing_km``
discussion in the project notes for the measured table.

Order binding
-------------
Each order's pickup is replaced by the launchpad it is bound to -- the pad *is*
the new pickup, because a drone departs from a pad and not from a restaurant. The
pad is the nearest one to the order's pickup location; when the source has no
pickup location at all (every LaDe city: 100% missing) the nearest pad to the
*delivery* location is used instead, which keeps those cities usable rather than
discarding them.

The flight that matters is then pad -> delivery. It sets the deadline too: the
window an order gets is ``slack_flight_multiple`` of its own one-way flight time,
clipped to ``[min_slack_seconds, max_slack_seconds]``, and the source's promised
time is not used at all -- see the constants for the measurements behind that.

Orders whose flight is
shorter than ``min_flight_km`` or longer than ``max_flight_km`` are masked out:
below the floor a drone is the wrong vehicle, above the ceiling it cannot make
the round trip. The mask is applied last, after binding, so it filters the leg
the simulator will actually fly.

Coordinates
-----------
``city_orders`` returns metres in a city-local frame. Entities are emitted back
in degrees, anchored at the city's own mean latitude where it has one, so the
simulator's haversine distances reproduce the frame's metres. Grubhub is
anonymised planar metres with no geography, so it is anchored at
``PLANAR_ANCHOR`` and its absolute coordinates are meaningless -- only the
distances are, which is all the simulator uses them for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..model.battery_types import BatteryType, from_config as battery_types_from_config
from ..model.entities import Battery, Drone, Launchpad, Location, Order
from ..model.enums import BatteryStatus, LaunchpadStatus, OrderStatus, VehicleStatus
from .launchpad_layout import load_layout, window_tag
from .pad_demand import allocate, offered_load
from .city_orders import (
    DEFAULT_EXTENT_TRIM,
    DEFAULT_WINDOW_SECONDS,
    EARTH_RADIUS_M,
    RawOrder,
    city_frame,
    load_day,
)

# the five-city shared revenue model; the simulator already depends on this
# package for SCENARIO_DEFAULT, and a second copy of the formula would drift
from data.instance_generator_synthetic.scenario import order_revenue

# Grubhub has no geography; this only has to be a plausible latitude so that the
# degree <-> metre conversion is well conditioned.
PLANAR_ANCHOR = (22.3, 114.17)

# --- the deadline window
#
# The window is a multiple of the order's own one-way flight time, clipped to a
# floor and a ceiling. The source's promised time is deliberately NOT used.
#
# Measured on the five cities, the source promise is a flat ~30 minutes that
# carries no per-order information: grubhub and shenzhen have no promised time at
# all (100% of orders fell back to the old flat 1800 s), and in meituanA/B the
# 1st percentile of the promise is 1801 s with 75% of orders inside 1808-1889 s,
# i.e. "order time + 30 min" applied regardless of distance. Using it bought
# realism in name only while leaving the five cities on two different conventions.
#
# It also left the window 15-18x the one-way flight time -- a 2 minute flight
# inside a 30 minute promise -- so an order could sit undispatched for 26 minutes
# and still be on time, and a 1 km order and a 9 km order had the same deadline.
# Scaling with flight time puts distance back into the on-time constraint.
DEFAULT_SLACK_FLIGHT_MULTIPLE = 6.0   # window = this many one-way flight times
DEFAULT_MIN_SLACK_SECONDS = 600       # floor, for orders at the 1 km mask edge
DEFAULT_MAX_SLACK_SECONDS = 1800      # ceiling, for orders at the 10 km mask edge
# Order weight and value are unchanged from the previous simulator (only the
# deadline convention changed): weight is
# U(0.2, max_payload) kg and value comes from the shared `order_revenue` model
# (base U(5, 15) + 2.0/km + 0.5/kg), measured over the launchpad-to-delivery leg.
ORDER_WEIGHT_MIN_KG = 0.2
ORDER_SIZE = 1.0
# Launchpads are laid over the *city* box by default, not the day's own box, so
# the same pad index is the same coordinate in every day and split. "day"
# restores the day-local box.
DEFAULT_EXTENT = DEFAULT_EXTENT_TRIM
# the slice of the source day an episode plays; see city_orders for why 10:00-12:00
DEFAULT_WINDOW = DEFAULT_WINDOW_SECONDS
DEFAULT_PAD_SPACING_KM = 2.0
# How the fleet is spread over the pads. "uniform" gives every pad
# ``drones_per_launchpad``, which is the control condition. "demand" keeps the
# same total and splits it in proportion to each pad's offered load, measured on
# the train split -- see ``io/pad_demand``.
DEFAULT_DRONE_PLACEMENT = "demand"
DRONE_PLACEMENTS = ("uniform", "demand")
# Drop the pads the demand split never used, before any drone is placed. On a
# demand-blind grid most pads can be dead ground -- at a 2 km pitch 164 of
# shenzhen's 432 pads carry no train order at all -- and a drone parked on one
# can never fly.
DEFAULT_PRUNE_EMPTY_PADS = True
# The floor is the leg below which a drone is the wrong vehicle: the pad has
# replaced the restaurant, so this is the whole trip the drone makes, and a 1 km
# hop is a walk. It is a large cut -- at a 2 km pitch it removes 63% of
# meituanC's orders and 64% of shenzhen's -- because meal delivery is a
# short-distance business (both cities' median order is 1.5 km end to end).
# Everything downstream is therefore a statement about the drone-eligible subset
# of a city, not about its delivery problem as a whole; nothing in this simulator
# models the courier who would take the orders this drops.
# The ceiling is where the round trip stops fitting in a pack.
DEFAULT_MIN_FLIGHT_KM = 2.0
DEFAULT_MAX_FLIGHT_KM = 10.0


@dataclass
class ScenarioReport:
    """What the build did, for the caller to print or assert on."""

    city: str
    split: str
    day: int
    n_launchpads: int
    n_candidate_pads: int             # grid cells before pruning
    pads_pruned: int
    layout_source: str | None         # the frozen layout used, or None if derived
    drone_placement: str
    drones_per_pad: tuple[int, int, int]   # (min, median, max) over pads
    pads_without_drones: int
    n_drones: int
    n_batteries: int
    battery_types: list[BatteryType]
    orders_in_source: int
    orders_kept: int
    masked_too_near: int
    masked_too_far: int
    outside_clip_box: int             # orders a core layout's box excludes
    bound_via_pickup: int
    bound_via_delivery: int
    slack_seconds: tuple[int, int, int]   # (min, median, max) of the deadline window
    slacks_at_floor: int
    slacks_at_cap: int
    extent: str                       # which box the pads were laid over
    extent_km: tuple[float, float]
    window_seconds: tuple[int, int] | None   # the slice of the source day played
    pad_grid: tuple[int, int]         # (cols, rows)
    pad_spacing_km: tuple[float, float]
    horizon_seconds: int

    def summary(self) -> str:
        kept = 100.0 * self.orders_kept / max(1, self.orders_in_source)
        caps = "/".join(f"{t.capacity_Wh:.0f}" for t in self.battery_types)
        return (
            f"{self.city}/{self.split}/day{self.day}: "
            f"{self.n_launchpads}/{self.n_candidate_pads} pads {self.pad_grid[0]}x{self.pad_grid[1]} "
            f"({self.pad_spacing_km[0]:.2f}x{self.pad_spacing_km[1]:.2f} km apart) "
            f"over the {self.extent} box, {self.extent_km[0]:.1f}x{self.extent_km[1]:.1f} km, "
            f"{'whole day' if self.window_seconds is None else '%s-%s s' % self.window_seconds}, "
            f"{'frozen' if self.layout_source else 'derived'}, "
            f"{self.n_drones} drones {self.drone_placement} "
            f"({self.drones_per_pad[0]}/{self.drones_per_pad[1]}/{self.drones_per_pad[2]} per pad "
            f"min/med/max, {self.pads_without_drones} pads empty), "
            f"{self.n_batteries} batteries [{caps} Wh]; "
            f"orders {self.orders_kept}/{self.orders_in_source} ({kept:.1f}% kept, "
            f"{self.masked_too_near} under, {self.masked_too_far} over"
            + (f", {self.outside_clip_box} outside the box" if self.outside_clip_box else "")
            + "), "
            f"bound {self.bound_via_pickup} by pickup / {self.bound_via_delivery} by delivery; "
            f"window {self.slack_seconds[0]}/{self.slack_seconds[1]}/{self.slack_seconds[2]}s "
            f"min/med/max ({self.slacks_at_floor} at floor, {self.slacks_at_cap} at cap); "
            f"horizon {self.horizon_seconds}s"
        )


def grid_shape(
    width_m: float, height_m: float, *, n: int | None = None, spacing_m: float | None = None
) -> tuple[int, int]:
    """Column and row count for a near-isotropic grid over a ``width x height`` box.

    Exactly one of ``n`` and ``spacing_m`` drives it:

    ``spacing_m``  the pitch is given and the count follows -- one pad every
                   ``spacing_m`` along each axis, rounded to the nearest whole
                   number of pads so the realised pitch is as near the requested
                   one as the box allows. The grid is then a full rectangle and
                   the pad count is ``cols * rows``.
    ``n``          the count is given and the pitch follows. The column count is
                   chosen so the cells are as near square as the box allows,
                   which keeps the spacing isotropic instead of making pads dense
                   in one axis.
    """
    if (n is None) == (spacing_m is None):
        raise ValueError("give exactly one of n_launchpads and pad_spacing_km")
    width = max(float(width_m), 1.0)
    height = max(float(height_m), 1.0)
    if spacing_m is not None:
        if spacing_m <= 0.0:
            raise ValueError(f"pad_spacing_km must be > 0, got {spacing_m / 1000.0}")
        return max(1, round(width / spacing_m)), max(1, round(height / spacing_m))
    if n < 1:
        raise ValueError(f"n_launchpads must be >= 1, got {n}")
    cols = min(max(1, int(round(math.sqrt(n * width / height)))), n)
    return cols, math.ceil(n / cols)


def grid_positions(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    *,
    n: int | None = None,
    spacing_m: float | None = None,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float], tuple[int, int]]:
    """Cell centres of a uniform grid over an explicit box, with its size and pitch.

    The grid is stretched to fit the box rather than laid at an exact pitch from a
    corner, so no pad sits outside the city and the realised pitch is the box
    divided by the pad count. Returns the centres, the box size in metres and the
    realised pitch in metres.

    When ``n`` is given and the grid cannot be exactly ``n`` cells, the trailing
    cells of the last row are dropped, so the last row may be short.
    """
    lo = np.array([float(x_range[0]), float(y_range[0])])
    hi = np.array([float(x_range[1]), float(y_range[1])])
    span = np.maximum(hi - lo, 1.0)          # a degenerate axis still needs a width
    cols, rows = grid_shape(span[0], span[1], n=n, spacing_m=spacing_m)
    total = cols * rows if n is None else min(int(n), cols * rows)
    xs = lo[0] + (np.arange(cols) + 0.5) * span[0] / cols
    ys = lo[1] + (np.arange(rows) + 0.5) * span[1] / rows
    centres = np.array([[x, y] for y in ys for x in xs], dtype=float)[:total]
    return (
        centres,
        (float(span[0]), float(span[1])),
        (float(span[0] / cols), float(span[1] / rows)),
        (cols, rows),
    )


def box_of(points_m: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """The bounding box of a point cloud, as ``(x_lo, x_hi), (y_lo, y_hi)``."""
    lo = points_m.min(axis=0)
    hi = points_m.max(axis=0)
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def _metres_to_location(east_m: float, north_m: float, anchor: tuple[float, float]) -> Location:
    lat0, lon0 = anchor
    cos_lat = max(1e-6, math.cos(math.radians(lat0)))
    return Location(
        lat=lat0 + math.degrees(north_m / EARTH_RADIUS_M),
        lon=lon0 + math.degrees(east_m / (EARTH_RADIUS_M * cos_lat)),
    )


def build_scenario(
    city: str,
    n_launchpads: int | None,
    drones_per_launchpad: int,
    batteries_per_drone: int,
    n_battery_types: int = 1,
    *,
    pad_spacing_km: float | None = None,
    extent: str = DEFAULT_EXTENT,
    window: tuple[int, int] | None = DEFAULT_WINDOW,
    drone_placement: str = DEFAULT_DRONE_PLACEMENT,
    prune_empty_pads: bool = DEFAULT_PRUNE_EMPTY_PADS,
    demand_split: str = "train",
    min_drones_per_pad: int = 0,
    split: str = "train",
    day: int = 0,
    config: dict | None = None,
    dataset_root: str | None = None,
    min_flight_km: float = DEFAULT_MIN_FLIGHT_KM,
    max_flight_km: float = DEFAULT_MAX_FLIGHT_KM,
    slack_flight_multiple: float = DEFAULT_SLACK_FLIGHT_MULTIPLE,
    min_slack_seconds: int = DEFAULT_MIN_SLACK_SECONDS,
    max_slack_seconds: int = DEFAULT_MAX_SLACK_SECONDS,
    order_weight_min: float = ORDER_WEIGHT_MIN_KG,
    order_size: float = ORDER_SIZE,
    seed: int = 42,
) -> tuple[list[Order], list[Drone], list[Launchpad], list[Battery], ScenarioReport]:
    """One instance. See the module docstring for what the counts mean."""
    if drones_per_launchpad < 1:
        raise ValueError(f"drones_per_launchpad must be >= 1, got {drones_per_launchpad}")
    if batteries_per_drone < 1:
        raise ValueError(f"batteries_per_drone must be >= 1, got {batteries_per_drone}")
    if not 0.0 <= min_flight_km < max_flight_km:
        raise ValueError(f"need 0 <= min_flight_km < max_flight_km, got {min_flight_km}, {max_flight_km}")

    config = config or {}
    kwargs = {} if dataset_root is None else {"root": dataset_root}

    # --- the pads: the frozen layout when one exists for this combination
    # A frozen layout already encodes the grid, the pruning and the train demand,
    # so taking it skips re-deriving all three and, more importantly, pins them:
    # the same file gives the same pads however the grid code later changes.
    layout = None
    if extent != "day" and pad_spacing_km is not None and n_launchpads is None:
        layout = load_layout(
            city,
            extent=extent,
            spacing_km=pad_spacing_km,
            min_km=min_flight_km,
            max_km=max_flight_km,
            demand_split=demand_split,
            window=window,
            root=dataset_root,
        )

    # --- the map the pads are laid over, and the frame everything is measured in
    if layout is not None:
        frame = layout.frame
        raw, frame = load_day(city, split, day, frame=frame, window=window, **kwargs)
    elif extent == "day":
        raw, frame = load_day(city, split, day, window=window, **kwargs)
        points = [o.delivery_m for o in raw] + [o.pickup_m for o in raw if o.pickup_m is not None]
        x_range, y_range = box_of(np.asarray(points, dtype=float))
    else:
        frame, x_range, y_range = city_frame(city, trim=extent, **kwargs)
        raw, frame = load_day(city, split, day, frame=frame, window=window, **kwargs)

    if layout is not None:
        candidates = layout.centres
        extent_m = (layout.pitch_km[0] * layout.pad_grid[0] * 1000.0,
                    layout.pitch_km[1] * layout.pad_grid[1] * 1000.0)
        cell_m = (layout.pitch_km[0] * 1000.0, layout.pitch_km[1] * 1000.0)
        cols, rows = layout.pad_grid
    else:
        candidates, extent_m, cell_m, (cols, rows) = grid_positions(
            x_range,
            y_range,
            n=n_launchpads,
            spacing_m=None if pad_spacing_km is None else pad_spacing_km * 1000.0,
        )
    anchor = (
        tuple(layout.anchor)
        if layout is not None
        else (PLANAR_ANCHOR if frame.planar else (frame.origin_y, frame.origin_x))
    )
    if drone_placement not in DRONE_PLACEMENTS:
        raise ValueError(
            f"drone_placement must be one of {DRONE_PLACEMENTS}, got {drone_placement!r}"
        )

    # --- what the demand split says about each candidate pad
    # Both the pruning and the fleet split are decided here, on `demand_split`
    # and never on the day being built, so the layout and the fleet are frozen
    # design decisions: an episode on a test day gets the pads and the drones
    # that the train days chose. The weight is offered load -- drone-seconds, not
    # order count -- because a pad with few long orders ties up as much of the
    # fleet as one with many short ones.
    needs_demand = layout is None and (prune_empty_pads or drone_placement == "demand")
    if needs_demand and extent == "day":
        raise ValueError(
            "extent='day' cannot be combined with pad pruning or demand placement: "
            "the day-local frame moves between days, so a weight measured on the "
            f"{demand_split} split does not describe this day's grid. "
            "Use a city extent, or pass prune_empty_pads=False and "
            "drone_placement='uniform'."
        )
    if needs_demand:
        weights = offered_load(
            city,
            candidates,
            frame,
            # the weights only describe the layout they were measured on
            grid_key=(
                f"{extent}_{cols}x{rows}_{min_flight_km:g}_{max_flight_km:g}"
                f"_{window_tag(window)}"
            ),
            min_km=min_flight_km,
            max_km=max_flight_km,
            speed_mps=float(
                (config.get("vehicles", {}) or {}).get("uavs", {}).get("speed_mps", 50.0 / 3.6)
            ),
            split=demand_split,
            window=window,
            root=dataset_root,
        )
    elif layout is not None:
        weights = layout.offered_load_s
    else:
        weights = np.ones(len(candidates))

    # --- drop the pads the demand split never used
    # A pad no train order ever bound to cannot be served by the fleet either:
    # with `drone_decision.rebalance.enabled` false, the default, a drone parked
    # there never gets an order, so it never flies and never leaves. Pruning on a
    # zero weight leaves the surviving pads' weights untouched, because the pads
    # removed had no orders to re-bind.
    # a frozen layout is pruned already; its own pads are all that exist
    keep = (
        np.ones(len(candidates), dtype=bool)
        if layout is not None or not prune_empty_pads
        else weights > 0.0
    )
    if not keep.any():
        raise ValueError(
            f"{city}: no pad of the {cols}x{rows} grid over the {extent} box carries any "
            f"{demand_split} demand inside [{min_flight_km:g}, {max_flight_km:g}] km"
        )
    grid_index = np.flatnonzero(keep)
    centres = candidates[keep]
    weights = weights[keep]
    n_launchpads = len(centres)

    pad_ids = layout.pad_ids if layout is not None else [f"lp_{i}" for i in range(n_launchpads)]
    launchpads = [
        Launchpad(
            launchpad_id=pad_ids[index],
            current_location=_metres_to_location(float(x), float(y), anchor),
            drone_capacity=None,          # the runtime knob sets this, not the instance
            status=LaunchpadStatus.STATIONARY,
        )
        for index, (x, y) in enumerate(centres)
    ]

    # --- how many drones each surviving pad starts with
    # `drones_per_launchpad` is the mean over the *surviving* pads, so pruning
    # shrinks the fleet with the layout instead of piling the removed pads'
    # drones onto the rest.
    total_drones = n_launchpads * drones_per_launchpad
    if drone_placement == "uniform":
        drones_at = np.full(n_launchpads, drones_per_launchpad, dtype=int)
    else:
        drones_at = allocate(weights, total_drones, minimum=min_drones_per_pad)

    # --- bind each order to a pad, mask by the leg that will actually be flown
    # Weight and revenue keep the previous simulator's conventions exactly: the
    # same two rng streams and their seeds, and the same shared `order_revenue`
    # model rather than a second copy of the formula. The revenue distance is the
    # launchpad-to-delivery leg, which is what the earlier generator used too
    # (it measured from the bound launchpad, not from the source pickup).
    weight_rng = np.random.default_rng(seed ^ 0x0A11CE)
    revenue_rng = np.random.default_rng(seed ^ 0xBA5E01)
    max_payload = float(
        (config.get("vehicles", {}) or {}).get("uavs", {}).get("max_payload_kg", 10.0)
    )
    min_m, max_m = min_flight_km * 1000.0, max_flight_km * 1000.0

    # A core layout serves one box of a larger city. Nearest-pad binding has no
    # range limit, so without this an order from the far side of the city would
    # bind to an edge pad of the core and count as demand the core never has.
    clip = None if layout is None else layout.clip_box_m

    kept: list[tuple[RawOrder, int, float]] = []
    outside = 0
    near = far = via_pickup = via_delivery = 0
    at_floor = at_cap = 0
    slacks: list[int] = []
    speed_mps = float(
        (config.get("vehicles", {}) or {}).get("uavs", {}).get("speed_mps", 50.0 / 3.6)
    )
    if slack_flight_multiple <= 0.0:
        raise ValueError(f"slack_flight_multiple must be > 0, got {slack_flight_multiple}")
    if not 0 < min_slack_seconds <= max_slack_seconds:
        raise ValueError(
            f"need 0 < min_slack_seconds <= max_slack_seconds, "
            f"got {min_slack_seconds}, {max_slack_seconds}"
        )
    for order in raw:
        anchor_point = order.pickup_m if order.pickup_m is not None else order.delivery_m
        if clip is not None and not (
            clip[0] <= anchor_point[0] <= clip[1] and clip[2] <= anchor_point[1] <= clip[3]
        ):
            outside += 1
            continue
        deltas = centres - np.asarray(anchor_point, dtype=float)
        pad_index = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
        if order.pickup_m is not None:
            via_pickup += 1
        else:
            via_delivery += 1
        flight_m = float(
            math.dist(centres[pad_index], order.delivery_m)
        )
        if flight_m < min_m:
            near += 1
            continue
        if flight_m > max_m:
            far += 1
            continue
        kept.append((order, pad_index, flight_m))

    if not kept:
        raise ValueError(
            f"{city}/{split}/day{day}: every order was masked out "
            f"({near} under {min_flight_km} km, {far} over {max_flight_km} km) with "
            f"{n_launchpads} pads {cols}x{rows}, {cell_m[0]/1000:.2f}x{cell_m[1]/1000:.2f} km apart. "
            "Pads too far from the demand mask orders as over-range; pads too close to it "
            "mask them as under-range -- move pad_spacing_km the other way, or widen the "
            "flight window."
        )

    # the clock starts at the first surviving order, as the earlier pipeline did
    start = min(order.created_at for order, _, _ in kept)
    # draw in a canonical order so the rng stream does not depend on CSV row order
    kept.sort(key=lambda item: (item[0].created_at, str(item[0].order_id)))
    orders: list[Order] = []
    for order, pad_index, flight_m in kept:
        weight = float(weight_rng.uniform(order_weight_min, max_payload))
        revenue = order_revenue(revenue_rng, flight_m / 1000.0, weight)
        # the window scales with the flight this order actually needs; the
        # source's promised time is not consulted (see the constants above)
        slack = int(round(slack_flight_multiple * flight_m / speed_mps))
        if slack <= min_slack_seconds:
            slack = min_slack_seconds
            at_floor += 1
        elif slack >= max_slack_seconds:
            slack = max_slack_seconds
            at_cap += 1
        slacks.append(slack)
        orders.append(
            Order(
                order_id=str(order.order_id),
                created_at=int(order.created_at - start),
                origin_launchpad_id=launchpads[pad_index].launchpad_id,
                destination=_metres_to_location(*order.delivery_m, anchor),
                size=order_size,
                weight=weight,
                value=revenue["value"],
                deadline=int(order.created_at - start + max(1, slack)),
                destination_endpoint_id=None,
                status=OrderStatus.PENDING,
                base_revenue=revenue["base_revenue"],
                distance_revenue=revenue["distance_revenue"],
                weight_revenue=revenue["weight_revenue"],
            )
        )
    orders.sort(key=lambda o: o.created_at)

    # --- the fleet
    uav = (config.get("vehicles", {}) or {}).get("uavs", {}) or {}
    drones: list[Drone] = []
    for launchpad, n_here in zip(launchpads, drones_at):
        for _ in range(int(n_here)):
            drone_id = f"drone_{len(drones)}"
            drones.append(
                Drone(
                    vehicle_id=drone_id,
                    current_location=launchpad.current_location,
                    available_at=0,
                    capacity=1.0,
                    remaining_capacity=1.0,
                    speed_mps=float(uav.get("speed_mps", 50.0 / 3.6)),
                    battery_level=None,
                    battery_capacity=None,
                    max_range_m=uav.get("max_range_m"),
                    empty_mass_kg=uav.get("empty_mass_kg"),
                    max_payload_kg=uav.get("max_payload_kg"),
                    current_battery_id=None,     # nothing is mounted at reset
                    current_launchpad_id=launchpad.launchpad_id,
                    status=VehicleStatus.IDLE,
                )
            )
            launchpad.drone_ids.append(drone_id)

    ladder = battery_types_from_config(config, n_battery_types)
    batteries: list[Battery] = []
    # packs follow the drones: they live at the pad those drones start on, but
    # are not bound to any one of them
    for launchpad, n_here in zip(launchpads, drones_at):
        for slot in range(int(n_here) * batteries_per_drone):
            # round-robin within the pad, so every pad holds the same mix
            spec = ladder[slot % len(ladder)]
            battery_id = f"battery_{len(batteries)}"
            batteries.append(
                Battery(
                    battery_id=battery_id,
                    capacity_Wh=spec.capacity_Wh,
                    current_charge_Wh=spec.capacity_Wh,
                    status=BatteryStatus.CHARGED,
                    current_launchpad_id=launchpad.launchpad_id,
                    current_vehicle_id=None,
                    mass_kg=spec.mass_kg,
                    full_charge_seconds=spec.full_charge_seconds,
                    type_id=spec.type_id,
                )
            )
            launchpad.charged_battery_ids.append(battery_id)

    report = ScenarioReport(
        city=city,
        split=split,
        day=day,
        n_launchpads=len(launchpads),
        n_candidate_pads=layout.n_candidate_pads if layout is not None else len(candidates),
        pads_pruned=(
            layout.n_candidate_pads - layout.n_pads
            if layout is not None
            else len(candidates) - len(launchpads)
        ),
        layout_source=None if layout is None else str(layout.source),
        window_seconds=None if window is None else tuple(window),
        drone_placement=drone_placement,
        drones_per_pad=(
            int(drones_at.min()),
            int(np.median(drones_at)),
            int(drones_at.max()),
        ),
        pads_without_drones=int((drones_at == 0).sum()),
        n_drones=len(drones),
        n_batteries=len(batteries),
        battery_types=ladder,
        orders_in_source=len(raw),
        orders_kept=len(orders),
        masked_too_near=near,
        masked_too_far=far,
        outside_clip_box=outside,
        bound_via_pickup=via_pickup,
        bound_via_delivery=via_delivery,
        slack_seconds=(
            int(min(slacks)),
            int(sorted(slacks)[len(slacks) // 2]),
            int(max(slacks)),
        ),
        slacks_at_floor=at_floor,
        slacks_at_cap=at_cap,
        extent=extent,
        extent_km=(extent_m[0] / 1000.0, extent_m[1] / 1000.0),
        pad_grid=(cols, rows),
        pad_spacing_km=(cell_m[0] / 1000.0, cell_m[1] / 1000.0),
        horizon_seconds=max(o.deadline for o in orders),
    )
    return orders, drones, launchpads, batteries, report
