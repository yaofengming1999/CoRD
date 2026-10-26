"""Which input columns reach the networks, and nothing else about them.

The menu builders keep producing every column they always produced: the raw
tails, the cross block and the row path are untouched, because the energy
algebra and the (o,b)|l regrouping are built from them. This module decides
only which columns are handed to the actor and the critics, by name, so an
ablation is a configuration rather than an edit, and a checkpoint trained under
one setting states the setting it needs through its layer widths.

Names are the ids of the feature tables in ``docs/supplement.tex``, so what a
run dropped can be read back without consulting the code.

``full`` returns the arrays untouched -- the same objects, not copies -- so a
run with no ablation is bit-identical to one from before this module existed.

The two measured presets:

``lossless``  the 14 columns that carry nothing, certified by block rank on
  hongkong and meituan-asn (F2 and F2pro, two instances each): eight constants
  (C10 = C11 = 1/|K|, C16 = 1, C20 = C21 = 0, C23 = 1, T3 = 0 under unlimited
  pad capacity, B3 = C/1500 = 0.48 with one battery type), two exact duplicates
  (C5 == C3, every docked drone is idle; C17 == C4, since a drone in flight is
  FLYING_*, never MOVING, so "moving" counts this step's dispatches and nothing
  else), and four halves of affine relations (C6 from C17; C8 from C7 and C3;
  C9 and C12 differ by a constant; B1 = (1500/720) B2). Block rank is unchanged
  at float32 tolerance: 12 + 15 + 3 = 30 of 44 columns.

``aggressive``  the lossless set plus nine columns that do carry information but
  very little of it: C13/C15/C19 and T4/T5/T10 are zero in 82-99% of the states
  visited, C19 takes three or four distinct values in a whole episode, and
  C14/C18/T2 are nearly as thin. Unlike the lossless set this one is a question,
  not a proof, and the answer is whatever the training arm says.
"""

from __future__ import annotations

import numpy as np

CONTEXT_NAMES: tuple[str, ...] = tuple(f"C{i}" for i in range(1, 24))
ORDER_NAMES: tuple[str, ...] = ("O1", "O3", "O4", "O5", "O6")
TARGET_NAMES: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T7", "T8", "T9", "T10")
ROUTE_NAMES: tuple[str, ...] = ("R1", "R3")
BATTERY_NAMES: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5")
OL_NAMES: tuple[str, ...] = ORDER_NAMES + TARGET_NAMES + ROUTE_NAMES

ALL_NAMES = frozenset(CONTEXT_NAMES + OL_NAMES + BATTERY_NAMES)

_LOSSLESS = ("C5", "C6", "C8", "C9", "C10", "C11", "C16", "C17", "C20", "C21", "C23",
             "T3", "B1", "B3")
_THIN = ("C13", "C14", "C15", "C18", "C19", "T2", "T4", "T5", "T10")

PRESETS: dict[str, tuple[str, ...]] = {
    "full": (),
    "lossless": _LOSSLESS,
    "aggressive": _LOSSLESS + _THIN,
}


def _keep(names: tuple[str, ...], drop: frozenset[str]) -> np.ndarray:
    return np.asarray([i for i, n in enumerate(names) if n not in drop], dtype=np.int64)


class FeatureMask:
    """The kept column indices of each block, plus the widths they imply."""

    __slots__ = ("drop", "context_idx", "ol_idx", "b_idx", "order_idx", "target_idx",
                 "context_dim", "ol_dim", "b_dim", "o_dim", "l_dim", "_identity")

    def __init__(self, drop: object = ()) -> None:
        names = frozenset(str(n).strip() for n in (drop or ()) if str(n).strip())
        unknown = sorted(names - ALL_NAMES)
        if unknown:
            raise ValueError(f"unknown feature id(s) {unknown}; known ids are "
                             f"{sorted(ALL_NAMES)}")
        self.drop = names
        self.context_idx = _keep(CONTEXT_NAMES, names)
        self.ol_idx = _keep(OL_NAMES, names)
        self.b_idx = _keep(BATTERY_NAMES, names)
        self.order_idx = _keep(ORDER_NAMES, names)
        self.target_idx = _keep(TARGET_NAMES, names)
        self.context_dim = int(len(self.context_idx))
        self.ol_dim = int(len(self.ol_idx))
        self.b_dim = int(len(self.b_idx))
        # F2pro's blocks are an order / target block plus the two coordinates
        self.o_dim = int(len(self.order_idx)) + 2
        self.l_dim = int(len(self.target_idx)) + 2
        self._identity = not names

    # -- the three actor-facing projections; identity means the same object back
    def context(self, c: np.ndarray) -> np.ndarray:
        return c if self._identity else c[..., self.context_idx]

    def ol(self, x: np.ndarray) -> np.ndarray:
        return x if self._identity else x[..., self.ol_idx]

    def b(self, x: np.ndarray) -> np.ndarray:
        return x if self._identity else x[..., self.b_idx]

    def order(self, x: np.ndarray) -> np.ndarray:
        return x if self._identity else x[..., self.order_idx]

    def target(self, x: np.ndarray) -> np.ndarray:
        return x if self._identity else x[..., self.target_idx]

    def __repr__(self) -> str:
        if self._identity:
            return "FeatureMask(full: 23 + 16 + 5 = 44 columns)"
        return (f"FeatureMask(dropped {len(self.drop)}: {', '.join(sorted(self.drop))}"
                f" -> {self.context_dim} + {self.ol_dim} + {self.b_dim}"
                f" = {self.context_dim + self.ol_dim + self.b_dim} columns)")


_ACTIVE = FeatureMask()
_SPEC: tuple[str, tuple[str, ...]] | None = None


def resolve(preset: str | None = None, drop: object = None) -> tuple[str, tuple[str, ...]]:
    """(preset name, dropped ids) from a preset name and/or an explicit list."""
    name = (preset or "full").strip().lower()
    if name not in PRESETS:
        raise ValueError(f"unknown feature preset {name!r}; known: {sorted(PRESETS)}")
    ids = tuple(PRESETS[name])
    if drop:
        extra = tuple(str(n).strip() for n in drop if str(n).strip())
        ids = tuple(dict.fromkeys(ids + extra))
    return name, ids


def configure(preset: str | None = None, drop: object = None) -> FeatureMask:
    """Set the process-wide mask. One run is one setting, so a second call with
    a different setting is a bug (a network already sized for the first one)."""
    global _ACTIVE, _SPEC
    spec = resolve(preset, drop)
    if _SPEC is not None and _SPEC != spec:
        raise RuntimeError(
            f"feature mask already configured as {_SPEC[0]} {sorted(_SPEC[1])}; "
            f"cannot switch to {spec[0]} {sorted(spec[1])} in the same process"
        )
    _SPEC = spec
    _ACTIVE = FeatureMask(spec[1])
    return _ACTIVE


def active() -> FeatureMask:
    return _ACTIVE


def spec() -> tuple[str, tuple[str, ...]]:
    return _SPEC if _SPEC is not None else ("full", ())


def reset() -> None:
    """Tests only: forget the configured mask."""
    global _ACTIVE, _SPEC
    _ACTIVE, _SPEC = FeatureMask(), None
