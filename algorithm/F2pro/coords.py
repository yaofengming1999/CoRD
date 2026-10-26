"""Coordinate features for the (order, pad) interaction.

The interaction term has to carry the return leg, which is a distance. With the
raw planar coordinate p the bilinear form reaches the SQUARED distance exactly,

    ||p - q||^2 = ||p||^2 + ||q||^2 - 2<p, q>,

the two norms inside the additive parts and the cross term at rank 2. The energy
follows the distance itself, and a square root of a bilinear form is not of the
form g + h + <u, v>, so with raw coordinates no finite rank reproduces it.

Fourier features remove that obstruction. For one frequency,

    cos(w . (p - q)) = cos(w.p) cos(w.q) + sin(w.p) sin(w.q),

which is a rank-two inner product in [cos(w.p), sin(w.p)]. With m frequencies the
classical random-features result says a bilinear form of rank 2m approximates any
shift-invariant kernel of p - q, the distance among them, to an accuracy that
improves with m. So the rank the interaction needs is no longer unbounded: it is
2m, set by how many frequencies the features carry.

    phi(p) = [ p , ||p||^2 , sin(Omega p) , cos(Omega p) ]      2 + 1 + 2m wide

The frequencies are drawn once from a fixed seed and never trained, so every run
of a configuration sees the same feature map and two runs differ only by the
policy. They are scaled to the coordinate frame: one unit is 5 km and a city
spans roughly 1 to 3 units (||p|| median 0.4 to 2.1, max about 3), so unit-scale
frequencies give about one cycle across a city and the band covers the range of
pad separations that matter.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def phi_dim(frequencies: int) -> int:
    """Width of phi(p) for a planar p."""
    return 2 + 1 + 2 * int(frequencies)


class CoordFeatures(nn.Module):
    """Expand the trailing (x, y) columns of a block into phi(p).

    The coordinates are the last two columns of each block -- dest_xy for the
    order rows, pad_xy for the pad rows -- so the expansion replaces those two
    columns and leaves the rest of the row untouched.
    """

    def __init__(self, frequencies: int = 8, scale: float = 2.0, seed: int = 0):
        super().__init__()
        self.frequencies = int(frequencies)
        if self.frequencies < 0:
            raise ValueError("frequencies must be >= 0")
        if self.frequencies:
            # fixed, not learned: a configuration's feature map must not drift
            # between runs, or two seeds would differ by more than the policy
            rng = np.random.default_rng(seed)
            omega = rng.normal(0.0, float(scale), size=(2, self.frequencies))
            self.register_buffer("omega", torch.as_tensor(omega, dtype=torch.float32))

    def extra_width(self) -> int:
        """How much wider a block becomes: phi(p) minus the two columns it replaces."""
        return phi_dim(self.frequencies) - 2

    def forward(self, block: torch.Tensor) -> torch.Tensor:
        """[..., D] -> [..., D - 2 + phi_dim]; the last two columns are the point."""
        head, point = block[..., :-2], block[..., -2:]
        parts = [head, point, (point * point).sum(-1, keepdim=True)]
        if self.frequencies:
            projected = point @ self.omega                     # [..., m]
            parts += [torch.sin(projected), torch.cos(projected)]
        return torch.cat(parts, dim=-1)
