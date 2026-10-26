"""Candidate feature layouts shared by the learned methods.

Raw vs. consumed blocks. ``MenuBuilder._order_features`` produces 7 raw order
quantities [value, urgency, slack, age, weight, delivery distance, queue rank]
and ``MenuBuilder.target_features`` 12 raw target quantities [return distance,
backlog, docked, reserved, incoming, outgoing, projected, gap, charged,
charging, home, occupancy ratio]. The F2 family (F2/F3/F1/AS*) consumes
ORDER_KEEP = [value, slack, age, weight, delivery distance] and
TARGET_KEEP = [backlog, docked, reserved, incoming, outgoing, gap, charged,
charging, home]; of the four route quantities [return distance, return time,
task energy, return energy] it consumes return distance and task energy
(removed 2026-09-21: urgency = monotone transform of slack, queue rank = FIFO
bookkeeping, projected occupancy = linear combination of docked/reserved/
incoming/outgoing, occupancy ratio = 0 under unlimited capacity, return time =
return distance / constant airspeed, return energy = function of return
distance alone). The raw blocks stay in ``FactorMenu.ol_raw`` because the
L2/L3 tokens are built from them.

``flat`` is the expanded triplet row (F3):

    [kind(3) | order(5) | battery(5) | route/matching(3) | target(1+9)]   = 26

``olb`` is the (o,l)|b row the row-based critic reads (and the separable actor
can read; on the default path it reads the factors instead):

    [kind(3) | order(5)+dest_xy(2) | battery(6) | cross(3) | target(9)+target_xy(2)]  = 30

  * dedup: flat route feat 'distance loaded' == order 'delivery distance' and
    flat target feat 0 == route 'distance return', so each is kept once;
  * +4 raw coordinates in a city-centred local-metre frame (/5000, like the
    distance feats): the order's destination on the order side, the return
    launchpad on the target side, so a bilinear <u,v> can rebuild dest x target
    geometry;
  * cross / route block = E_task/C, energy margin (q_b - E_task - reserve)/C and
    return distance: the order x launchpad (x battery) quantities.
"""

from __future__ import annotations

import math

import numpy as np

ORDER_RAW_DIM = 7
ORDER_KEEP = [0, 2, 3, 4, 5]        # value, slack, age, weight, delivery distance
ORDER_DIM = len(ORDER_KEEP)         # 5
BATTERY_DIM = 5
TARGET_RAW_DIM = 11                  # raw target features 1..11 (feature 0 = return distance lives in the route block)
TARGET_KEEP = [0, 1, 2, 3, 4, 6, 7, 8, 9]   # within the raw 11: backlog, docked, reserved, incoming, outgoing, gap, charged, charging, home
TARGET_DIM = len(TARGET_KEEP)       # 9
ROUTE_DIM = 3                        # flat route block: e_task/C, margin, d_ret

FLAT_DIM = 3 + ORDER_DIM + BATTERY_DIM + ROUTE_DIM + (TARGET_DIM + 1)     # 26
OLB_DIM = 3 + ORDER_DIM + 2 + BATTERY_DIM + ROUTE_DIM + TARGET_DIM + 2     # 29

# flat slices
F_KIND = slice(0, 3)
F_ORDER = slice(3, 3 + ORDER_DIM)                                    # 3:8
F_BATTERY = slice(F_ORDER.stop, F_ORDER.stop + BATTERY_DIM)          # 8:13
F_ROUTE = slice(F_BATTERY.stop, F_BATTERY.stop + ROUTE_DIM)          # 13:16  e_task/C, margin, d_ret
F_TARGET = slice(F_ROUTE.stop, FLAT_DIM)                             # 16:26  target(1+9)
# olb slices
KIND = slice(0, 3)
ORDER = slice(3, 3 + ORDER_DIM + 2)                                  # 3:10   order5 + dest_xy
BATTERY = slice(ORDER.stop, ORDER.stop + BATTERY_DIM)                # 10:15
CROSS = slice(BATTERY.stop, BATTERY.stop + ROUTE_DIM)                # 15:18  e_task/C, margin, d_ret
TARGET = slice(CROSS.stop, OLB_DIM)                                  # 18:29  target9 + target_xy
_FLAT_CROSS_IDX = list(range(F_ROUTE.start, F_ROUTE.stop))            # the flat route block is the olb cross block

CANDIDATE_DIMS = {"flat": FLAT_DIM, "olb": OLB_DIM}

# --------------------------------------------------------------------------- #
# The other way to factor one dispatch: (order, battery) | return pad.
#
# Which side of the split the return pad falls on is not a free choice. The
# energy margin is
#     charge_b - 0.2C - E_deliver(o) - E_return(o, l),
# and E_return straddles o and l. Grouping (o,l) puts the whole of E_task into
# one block and leaves charge in the other, so the margin is a row effect plus a
# column effect and an additive score represents it exactly -- which is why
# rank 0 suffices there (table 4). Grouping (o,b) instead puts charge and
# E_deliver together, but then nothing can hold E_return(o,l): the pad block
# knows only the pad. The coupling has to come from the bilinear term, and since
# both blocks carry coordinates, squared Euclidean distance is
#     ||x_dest||^2 + ||x_pad||^2 - 2<x_dest, x_pad>,
# whose cross term needs rank 2 in the plane. Road distance is not bilinear at
# all, so rank 2 approximates it and no rank recovers it.
#
# The row counts also differ: N_o*N_l + N_b against N_o*N_b + N_l. The return
# pads of a departure pad are shared by all its orders, so the pad block is one
# row per pad whichever way the split goes.
OB_DIM = ORDER_DIM + 2 + BATTERY_DIM      # 12: order5 | dest_xy2 | battery5
L_DIM = TARGET_DIM + 2                    # 11: target9 | target_xy2
CANDIDATE_DIMS["obl"] = OLB_DIM           # the row path is unchanged; only the actor's view differs

Frame = tuple[float, float, float, float]  # ref_lat, ref_lon, sx, sy


def city_frame(state: dict) -> Frame | None:
    """Local-metre frame centred on the launchpad centroid (scaled by /5000)."""
    lats, lons = [], []
    for lp in state.get("launchpads", {}).values():
        loc = lp.get("current_location")
        if loc:
            lats.append(float(loc.get("lat", 0.0)))
            lons.append(float(loc.get("lon", 0.0)))
    if not lats:
        return None
    ref_lat, ref_lon = float(np.mean(lats)), float(np.mean(lons))
    return (ref_lat, ref_lon,
            111_320.0 * math.cos(math.radians(ref_lat)) / 5000.0,
            110_540.0 / 5000.0)


def frame_xy(loc: dict | None, frame: Frame | None) -> tuple[float, float]:
    if not loc or frame is None:
        return (0.0, 0.0)
    ref_lat, ref_lon, sx, sy = frame
    return ((float(loc.get("lon", 0.0)) - ref_lon) * sx,
            (float(loc.get("lat", 0.0)) - ref_lat) * sy)


def reform_flat_to_olb(
    f_flat: np.ndarray, dest_xy: tuple[float, float], target_xy: tuple[float, float]
) -> np.ndarray:
    f_olb = np.concatenate([
        f_flat[F_KIND],
        f_flat[F_ORDER], np.asarray(dest_xy, dtype=np.float32),
        f_flat[F_BATTERY],
        f_flat[_FLAT_CROSS_IDX],
        f_flat[F_TARGET.start + 1 : F_TARGET.stop], np.asarray(target_xy, dtype=np.float32),
    ]).astype(np.float32)
    if f_olb.shape != (OLB_DIM,):
        raise RuntimeError(f"olb candidate must have shape ({OLB_DIM},), got {f_olb.shape}")
    return f_olb


# --------------------------------------------------------------------------- #
# Factor menu: the (o,l) | b representation of one dispatch decision.
#
# Instead of one 30-dim row per feasible (order, return pad, battery) triplet,
# a decision is described by
#   ol  [N_ol, 16]  one row per (order, return pad) pair (+ one row per order
#                   for its DEFER entry, + one all-zero row for STOP),
#   b   [N_b, 6]    one row per battery (+ one all-zero row for non-dispatch),
#   and per flat menu entry (in the same order the triplet loop emits them)
#   kind [N,3], ol_idx [N], b_idx [N].
# The separable actor scores ol and b once each and gathers the grid; the
# 30-dim rows (needed only by row-based critics) are rebuilt on demand from the
# raw factor blocks with the same float64 arithmetic as the triplet builder.
# --------------------------------------------------------------------------- #
OL_DIM = ORDER_DIM + TARGET_DIM + 2   # 16: order5 | target9 | d_ret | E_task/1500
B_DIM = BATTERY_DIM                   # 5
# raw per-(o,l) block: order7(raw) | dest_xy2 | target11 | target_xy2 | E_task | E_return | dist_return/5000 | time_return/horizon
OL_RAW_DIM = ORDER_RAW_DIM + 2 + TARGET_RAW_DIM + 2 + 4   # 26
_RAW_XY = slice(ORDER_RAW_DIM, ORDER_RAW_DIM + 2)                      # 7:9
_RAW_TARGET = slice(_RAW_XY.stop, _RAW_XY.stop + TARGET_RAW_DIM)       # 9:20
_RAW_TARGET_KEEP = [_RAW_TARGET.start + k for k in TARGET_KEEP]        # consumed target columns
_RAW_TXY = slice(_RAW_TARGET.stop, _RAW_TARGET.stop + 2)               # 20:22
_RAW_E_TASK, _RAW_E_RET, _RAW_D_RET, _RAW_T_RET = 22, 23, 24, 25
# raw per-battery block: battery5 | charge | cap_env(max(1,C)) | cap_model(C or 1) | reserve
B_RAW_DIM = 9


class FactorMenu:
    __slots__ = ("ol_raw", "b_raw", "ol", "b", "kind", "ol_idx", "b_idx", "n_ol", "n_b", "_rows")

    def __init__(self, ol_raw: np.ndarray, b_raw: np.ndarray, ol: np.ndarray, b: np.ndarray,
                 kind: np.ndarray, ol_idx: np.ndarray, b_idx: np.ndarray, n_ol: int, n_b: int):
        self.ol_raw = ol_raw      # [N_ol_total, 26] float64  (dispatch (o,l) rows, defer rows, zero stop row)
        self.b_raw = b_raw        # [N_b + 1, 10]    float64  (batteries, zero row for defer/stop)
        self.ol = ol              # [N_ol_total, OL_DIM=16] float32  actor input
        self.b = b                # [N_b + 1, B_DIM=5] float32  actor input
        self.kind = kind          # [N, 3] float32
        self.ol_idx = ol_idx      # [N] int64
        self.b_idx = b_idx        # [N] int64
        self.n_ol = int(n_ol)     # real (o,l) dispatch rows (for the feature_rows counter)
        self.n_b = int(n_b)       # real batteries
        self._rows = None

    def __len__(self) -> int:
        return int(len(self.ol_idx))

    @property
    def factor_rows(self) -> int:
        """Rows the separable actor actually scores: (o,l) pairs + batteries."""
        return self.n_ol + self.n_b

    def subset(self, keep: list[int]) -> "FactorMenu":
        """Menu restricted to the given flat entries (factor blocks are shared)."""
        idx = np.asarray(keep, dtype=np.int64)
        return FactorMenu(self.ol_raw, self.b_raw, self.ol, self.b,
                          self.kind[idx], self.ol_idx[idx], self.b_idx[idx], self.n_ol, self.n_b)

    def rows(self) -> np.ndarray:
        """Rebuild the olb rows (float64 ops, then float32) -- bit-identical to the triplet builder."""
        if self._rows is None:
            self._rows = rows_from_factors(self.ol_raw, self.b_raw, self.kind.astype(np.float64), self.ol_idx, self.b_idx)
        return self._rows


def rows_from_factors(ol_raw, b_raw, kind, ol_idx, b_idx) -> np.ndarray:
    o = ol_raw[ol_idx]            # [N, 26]
    bb = b_raw[b_idx]             # [N, 10]
    dispatch = kind[:, 0:1]       # [N, 1]
    e_task = o[:, _RAW_E_TASK:_RAW_E_TASK + 1]
    d_ret = o[:, _RAW_D_RET:_RAW_D_RET + 1]
    _c = BATTERY_DIM     # the raw tail starts where the feature block ends
    charge = bb[:, _c:_c + 1]; cap_env = bb[:, _c + 1:_c + 2]
    cap_model = bb[:, _c + 2:_c + 3]; reserve = bb[:, _c + 3:_c + 4]
    cross = np.concatenate([
        e_task / cap_env,
        ((charge - e_task) - reserve) / cap_model,
        d_ret,
    ], axis=1) * dispatch
    rows = np.concatenate([
        kind,                 # 3
        o[:, ORDER_KEEP],     # order 5
        o[:, _RAW_XY],        # dest_xy 2
        bb[:, 0:BATTERY_DIM],  # battery 6
        cross,                # 5
        o[:, _RAW_TARGET_KEEP],  # target 9
        o[:, _RAW_TXY],       # target_xy 2
    ], axis=1)
    return rows.astype(np.float32)


def ol_from_raw(ol_raw: np.ndarray, cap_env_ref: float, cap_over_1500_ref: float) -> np.ndarray:
    """Actor-side (o,l) input, matching ``OLBSeparableActor.inputs`` on rows bit for bit:
    order5 | target9 | dist_return | E_task/1500, where the energy term is formed
    exactly as the row path does it, f32(E/C) * f32(C/1500)."""
    n = len(ol_raw)
    e_task = (ol_raw[:, _RAW_E_TASK] / cap_env_ref).astype(np.float32) * np.float32(cap_over_1500_ref)
    out = np.empty((n, OL_DIM), dtype=np.float32)
    out[:, 0:ORDER_DIM] = ol_raw[:, ORDER_KEEP]
    out[:, ORDER_DIM:ORDER_DIM + TARGET_DIM] = ol_raw[:, _RAW_TARGET_KEEP]
    out[:, ORDER_DIM + TARGET_DIM] = ol_raw[:, _RAW_D_RET]
    out[:, ORDER_DIM + TARGET_DIM + 1] = e_task
    return out


def obl_from_factor_menu(menu: "FactorMenu", candidates: list) -> "FactorMenu":
    """Regroup an (o,l) | b menu into (o,b) | l, over the same candidates.

    This is a regrouping, not a new menu: the entries, their feasibility and the
    energy calls behind them are exactly those the (o,l) | b builder produced, so
    the two factorisations face the same action set and a difference between them
    is the split and nothing else.

    What moves. The first block becomes one row per (order, battery) pair, holding
    order5 | dest_xy2 | battery5; the second becomes one row per return pad,
    holding target9 | target_xy2. Shared pads are one row whichever way the split
    goes, since the return pads of a departure pad are the same for all its orders.

    What is lost, and why that is the experiment. The margin is
    ``charge_b - 0.2C - E_deliver(o) - E_return(o, l)``. Grouping (o,l) puts all of
    E_task into the first block, so the margin is a row effect plus a column effect
    and an additive score represents it exactly -- rank 0 suffices. Grouping (o,b)
    puts charge and E_deliver together, but nothing can then hold E_return(o, l):
    the pad block knows only the pad. The dropped columns are exactly E_task,
    E_return, the return distance and the return time -- every quantity that needs
    both o and l. Both blocks keep coordinates, so with drones flying straight
    lines the bilinear term can reach the geometry: squared distance is
    ``||x_dest||^2 + ||x_pad||^2 - 2<x_dest, x_pad>``, whose cross term is rank 2
    in the plane. The energy follows distance rather than its square, and a square
    root of a bilinear form is not of the form ``g + h + <u, v>``, so rank 2
    approximates the return leg and no finite rank recovers it.
    """
    n = len(menu.ol_idx)
    if n != len(candidates):
        raise ValueError(f"(o,b)|l regrouping needs one candidate per entry, got {n} and {len(candidates)}")

    ob_rows: list[np.ndarray] = []
    l_rows: list[np.ndarray] = []
    ob_of: dict[tuple, int] = {}
    l_of: dict[str, int] = {}
    kind = np.asarray(menu.kind, dtype=np.float32)
    ob_idx = np.empty(n, dtype=np.int64)
    l_idx = np.empty(n, dtype=np.int64)

    # one all-zero pad row, for the entries that name no pad (defer, stop)
    l_rows.append(np.zeros(L_DIM, dtype=np.float64))
    zero_l = 0

    for entry in range(n):
        candidate = candidates[entry]
        source = menu.ol_raw[int(menu.ol_idx[entry])]
        order_part = np.concatenate([source[ORDER_KEEP], source[_RAW_XY]])      # order5 | dest_xy2

        if candidate.kind == "dispatch":
            key = (str(candidate.order_id), str(candidate.battery_id))
            if key not in ob_of:
                battery = menu.b_raw[int(menu.b_idx[entry]), 0:BATTERY_DIM]
                ob_of[key] = len(ob_rows)
                ob_rows.append(np.concatenate([order_part, battery]))
            ob_idx[entry] = ob_of[key]

            pad = str(candidate.return_launchpad_id)
            if pad not in l_of:
                l_of[pad] = len(l_rows)
                l_rows.append(np.concatenate([source[_RAW_TARGET_KEEP], source[_RAW_TXY]]))
            l_idx[entry] = l_of[pad]
        else:
            # defer keeps its order block and no battery; stop keeps nothing. Both
            # point at the zero pad row, as they do in the (o,l) | b menu.
            ob_idx[entry] = len(ob_rows)
            ob_rows.append(np.concatenate([
                order_part if candidate.kind == "defer" else np.zeros(ORDER_DIM + 2),
                np.zeros(BATTERY_DIM),
            ]))
            l_idx[entry] = zero_l

    ob = np.stack(ob_rows) if ob_rows else np.zeros((0, OB_DIM))
    ell = np.stack(l_rows)
    if ob.shape[1] != OB_DIM or ell.shape[1] != L_DIM:
        raise RuntimeError(f"(o,b)|l blocks must be {OB_DIM} and {L_DIM} wide, "
                           f"got {ob.shape[1]} and {ell.shape[1]}")
    return FactorMenu(
        ol_raw=ob, b_raw=ell,
        ol=ob.astype(np.float32), b=ell.astype(np.float32),
        kind=kind, ol_idx=ob_idx, b_idx=l_idx,
        n_ol=len(ob_of), n_b=len(l_of),
    )


def stack_factor_menus(menus: list[FactorMenu]) -> dict[str, np.ndarray]:
    """Pad FactorMenus into batched numpy factor arrays (+ flat mask)."""
    R = len(menus)
    n_ol = max(m.ol.shape[0] for m in menus)
    n_b = max(m.b.shape[0] for m in menus)
    n_flat = max(len(m) for m in menus)
    # the block widths come from the menus, not from the (o,l) | b constants: the
    # (o,b) | l split produces the same structure with blocks 12 and 11 wide
    a_dim = menus[0].ol.shape[1]
    b_dim = menus[0].b.shape[1]
    if any(m.ol.shape[1] != a_dim or m.b.shape[1] != b_dim for m in menus):
        raise ValueError("a batch of menus must share one block layout")
    ol = np.zeros((R, n_ol, a_dim), dtype=np.float32)
    b = np.zeros((R, n_b, b_dim), dtype=np.float32)
    kind = np.zeros((R, n_flat, 3), dtype=np.float32)
    ol_idx = np.zeros((R, n_flat), dtype=np.int64)
    b_idx = np.zeros((R, n_flat), dtype=np.int64)
    mask = np.zeros((R, n_flat), dtype=bool)
    ol_mask = np.zeros((R, n_ol), dtype=bool)   # real (o,l) rows of each menu (incl. defer rows and the stop row)
    b_mask = np.zeros((R, n_b), dtype=bool)     # real battery rows (incl. the zero row)
    for r, m in enumerate(menus):
        ol[r, : m.ol.shape[0]] = m.ol; ol_mask[r, : m.ol.shape[0]] = True
        b[r, : m.b.shape[0]] = m.b; b_mask[r, : m.b.shape[0]] = True
        n = len(m)
        kind[r, :n] = m.kind; ol_idx[r, :n] = m.ol_idx; b_idx[r, :n] = m.b_idx; mask[r, :n] = True
    return {"factor_ol": ol, "factor_b": b, "factor_kind": kind,
            "factor_ol_idx": ol_idx, "factor_b_idx": b_idx, "mask": mask,
            "factor_ol_mask": ol_mask, "factor_b_mask": b_mask}


def stack_factor_rows(menus: list[FactorMenu], n_flat: int) -> np.ndarray:
    """Expanded olb rows for row-based critics, rebuilt from the factors."""
    rows = np.zeros((len(menus), n_flat, OLB_DIM), dtype=np.float32)
    for r, m in enumerate(menus):
        rows[r, : len(m)] = m.rows()
    return rows


def empty_factor_menu() -> FactorMenu:
    """A one-entry STOP menu (used to pad absent agents in joint batches)."""
    return FactorMenu(
        ol_raw=np.zeros((1, OL_RAW_DIM)), b_raw=np.array([[0.0] * 5 + [0.0, 1.0, 1.0, 0.0]]),
        ol=np.zeros((1, OL_DIM), dtype=np.float32), b=np.zeros((1, B_DIM), dtype=np.float32),
        kind=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
        ol_idx=np.zeros(1, dtype=np.int64), b_idx=np.zeros(1, dtype=np.int64), n_ol=0, n_b=0,
    )
