"""The three-block menu: order rows, return-pad rows, battery rows.

One decision becomes

    o   [N_o, O_DIM]   one row per order        order5 | dest_xy2
    l   [N_l, L_DIM]   one row per return pad   target9 | target_xy2
    b   [N_b, B_DIM]   one row per battery      battery5

plus an index grid over the flat menu entries, (kind, o_idx, l_idx, b_idx), in
the order the candidate list already has. A dispatch entry points into all three
blocks; defer points at its order and at the zero pad and zero battery rows; stop
points at zero rows throughout.

Why the blocks hold what they hold. Everything that depends on one of the three
alone goes into its block. The quantities that depend on a pair -- the task
energy, the return energy, the return distance and time -- belong to no block and
are dropped; recovering them is the interaction term's job, which is the point of
the experiment.
"""

from __future__ import annotations

import numpy as np

from algorithm.F2 import featmask
from algorithm.F2.menu import (
    BATTERY_DIM,
    ORDER_DIM,
    ORDER_KEEP,
    TARGET_DIM,
    _RAW_TARGET_KEEP,
    _RAW_TXY,
    _RAW_XY,
)

# The unmasked widths. What a block is actually built at is the feature mask's
# o_dim / l_dim / b_dim, which equal these when nothing is dropped.
O_DIM = ORDER_DIM + 2        # 7:  value, slack, age, weight, delivery distance | dest_xy
L_DIM = TARGET_DIM + 2       # 11: backlog, docked, reserved, incoming, outgoing, gap,
                             #     charged, charging, home | pad_xy
B_DIM = BATTERY_DIM          # 5:  soc, charge/1500, capacity/1500, charging, time to full


class TriMenu:
    """Three blocks and the index grid that gathers them into the flat menu."""

    __slots__ = ("o", "l", "b", "kind", "o_idx", "l_idx", "b_idx", "n_o", "n_l", "n_b")

    def __init__(self, o, l, b, kind, o_idx, l_idx, b_idx, n_o, n_l, n_b):
        self.o = o              # [N_o + 1, O_DIM] float32, last row zero (stop)
        self.l = l              # [N_l + 1, L_DIM] float32, last row zero (defer, stop)
        self.b = b              # [N_b + 1, B_DIM] float32, last row zero (defer, stop)
        self.kind = kind        # [N, 3] float32  one-hot: dispatch, defer, stop
        self.o_idx = o_idx      # [N] int64
        self.l_idx = l_idx      # [N] int64
        self.b_idx = b_idx      # [N] int64
        self.n_o = int(n_o)     # real orders  (for the row-count metric)
        self.n_l = int(n_l)     # real pads
        self.n_b = int(n_b)     # real batteries

    def __len__(self) -> int:
        return int(len(self.o_idx))

    @property
    def scored_rows(self) -> int:
        """Rows the actor scores: the three blocks once each, not their product."""
        return self.n_o + self.n_l + self.n_b


def tri_from_candidates(factor_menu, candidates: list) -> TriMenu:
    """Build a TriMenu from the F2 menu's candidates and raw blocks.

    The entries, their order and their feasibility are the F2 builder's, so both
    methods offer the same actions. Only the grouping of the features is new: the
    order half and the pad half of each (o,l) raw row are separated, and the pair
    columns that sit between them -- E_task, E_return, the return distance and
    time -- are dropped, since no single block can hold them.
    """
    mask = featmask.active()
    n = len(factor_menu.ol_idx)
    if n != len(candidates):
        raise ValueError(f"TriMenu needs one candidate per entry, got {n} and {len(candidates)}")

    o_rows: list[np.ndarray] = []
    l_rows: list[np.ndarray] = []
    b_rows: list[np.ndarray] = []
    o_of: dict[str, int] = {}
    l_of: dict[str, int] = {}
    b_of: dict[str, int] = {}
    o_idx = np.empty(n, dtype=np.int64)
    l_idx = np.empty(n, dtype=np.int64)
    b_idx = np.empty(n, dtype=np.int64)

    for entry in range(n):
        candidate = candidates[entry]
        source = factor_menu.ol_raw[int(factor_menu.ol_idx[entry])]

        if candidate.order_id is not None:
            key = str(candidate.order_id)
            if key not in o_of:
                o_of[key] = len(o_rows)
                o_rows.append(np.concatenate([mask.order(source[ORDER_KEEP]), source[_RAW_XY]]))
            o_idx[entry] = o_of[key]
        else:
            o_idx[entry] = -1          # stop: patched to the zero row below

        if candidate.return_launchpad_id is not None:
            key = str(candidate.return_launchpad_id)
            if key not in l_of:
                l_of[key] = len(l_rows)
                l_rows.append(np.concatenate(
                    [mask.target(source[_RAW_TARGET_KEEP]), source[_RAW_TXY]]))
            l_idx[entry] = l_of[key]
        else:
            l_idx[entry] = -1

        if candidate.battery_id is not None:
            key = str(candidate.battery_id)
            if key not in b_of:
                b_of[key] = len(b_rows)
                b_rows.append(mask.b(np.asarray(
                    factor_menu.b_raw[int(factor_menu.b_idx[entry]), 0:BATTERY_DIM], dtype=np.float64)))
            b_idx[entry] = b_of[key]
        else:
            b_idx[entry] = -1

    # one zero row per block, appended last, for the entries that name nothing.
    # The index is only known once the real rows are counted, which is why the
    # placeholders above are -1 rather than a position: pads and batteries are
    # discovered as the entries are walked, so len() mid-walk is not the final count.
    def finish(rows, width, idx):
        block = np.zeros((len(rows) + 1, width), dtype=np.float64)
        if rows:
            block[: len(rows)] = np.stack(rows)
        return block, np.where(idx < 0, len(rows), idx)

    o_block, o_idx = finish(o_rows, mask.o_dim, o_idx)
    l_block, l_idx = finish(l_rows, mask.l_dim, l_idx)
    b_block, b_idx = finish(b_rows, mask.b_dim, b_idx)

    return TriMenu(
        o=o_block.astype(np.float32), l=l_block.astype(np.float32), b=b_block.astype(np.float32),
        kind=np.asarray(factor_menu.kind, dtype=np.float32),
        o_idx=o_idx, l_idx=l_idx, b_idx=b_idx,
        n_o=len(o_of), n_l=len(l_of), n_b=len(b_of),
    )


def stack_tri_menus(menus: list[TriMenu]) -> dict[str, np.ndarray]:
    """Pad a minibatch of TriMenus, with a mask over the flat entries."""
    R = len(menus)
    n_o = max(m.o.shape[0] for m in menus)
    n_l = max(m.l.shape[0] for m in menus)
    n_b = max(m.b.shape[0] for m in menus)
    n_flat = max(len(m) for m in menus)
    o = np.zeros((R, n_o, menus[0].o.shape[-1]), dtype=np.float32)
    l = np.zeros((R, n_l, menus[0].l.shape[-1]), dtype=np.float32)
    b = np.zeros((R, n_b, menus[0].b.shape[-1]), dtype=np.float32)
    kind = np.zeros((R, n_flat, 3), dtype=np.float32)
    o_idx = np.zeros((R, n_flat), dtype=np.int64)
    l_idx = np.zeros((R, n_flat), dtype=np.int64)
    b_idx = np.zeros((R, n_flat), dtype=np.int64)
    mask = np.zeros((R, n_flat), dtype=bool)
    o_mask = np.zeros((R, n_o), dtype=bool)
    l_mask = np.zeros((R, n_l), dtype=bool)
    b_mask = np.zeros((R, n_b), dtype=bool)
    for r, m in enumerate(menus):
        o[r, : m.o.shape[0]] = m.o; o_mask[r, : m.o.shape[0]] = True
        l[r, : m.l.shape[0]] = m.l; l_mask[r, : m.l.shape[0]] = True
        b[r, : m.b.shape[0]] = m.b; b_mask[r, : m.b.shape[0]] = True
        k = len(m)
        kind[r, :k] = m.kind
        o_idx[r, :k] = m.o_idx; l_idx[r, :k] = m.l_idx; b_idx[r, :k] = m.b_idx
        mask[r, :k] = True
    return {"o": o, "l": l, "b": b, "kind": kind, "o_idx": o_idx, "l_idx": l_idx,
            "b_idx": b_idx, "mask": mask, "o_mask": o_mask, "l_mask": l_mask, "b_mask": b_mask}
