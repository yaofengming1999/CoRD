"""The three-block separable actor: additive in o, l and b, coupled only on (o, l).

    z_n = k_kappa(s)
        + 1[kappa != stop]    * g_o(x_o)
        + 1[kappa == dispatch] * ( g_l(x_l) + h_b(x_b) + <u(x_o), v(x_l)> )

The coupling sits on the order-pad pair and nowhere else, because that is where
the problem puts it: the return energy E_return(o, l) is a property of the pair,
while the battery enters the margin additively (charge_b - 0.2C - E_task), which
table 4 confirms by finding rank 0 enough for the (o,l) | b split.

The gates follow the F2 actor's convention. A defer entry names an order but no
pad and no battery, so it takes the order term only; stop takes the kind bias
alone. Writing the gates as products with the kind one-hot keeps the whole menu
in one tensor instead of three branches.

Rows scored per decision: N_o + N_l + N_b. The blocks are encoded once each and
gathered, so the product is never built -- on the cities measured the worst case
of a decision falls from about 2600 rows to under 170.
"""

from __future__ import annotations

import torch
from torch import nn

from algorithm.F2pro.coords import CoordFeatures
from algorithm.F2 import featmask


class TriSeparableActor(nn.Module):
    """Logits over a TriMenu."""

    def __init__(self, context_dim: int, hidden_dim: int, rank: int = 2,
                 coord_frequencies: int = 0, coord_scale: float = 2.0, coord_seed: int = 0):
        """``coord_frequencies`` > 0 replaces each block's trailing (x, y) with

            phi(p) = [p, ||p||^2, sin(Omega p), cos(Omega p)].

        With raw coordinates the bilinear term reaches the squared return distance
        exactly at rank 2 but never the distance itself, which is its square root.
        With m frequencies, cos(w.(p-q)) splits into a rank-two inner product per
        frequency, so a rank-2m form approximates the distance -- the rank the
        interaction needs becomes 2m instead of unbounded. m=4 is the setting
        matched to rank 8.
        """
        super().__init__()
        if int(rank) < 0:
            raise ValueError("rank must be >= 0")
        self.rank = int(rank)
        self.coord_frequencies = int(coord_frequencies)
        self.coords = (CoordFeatures(self.coord_frequencies, coord_scale, coord_seed)
                       if self.coord_frequencies else None)
        extra = self.coords.extra_width() if self.coords is not None else 0

        def mlp(width: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.context_net = mlp(context_dim)
        mask = featmask.active()
        self.o_net = mlp(mask.o_dim + extra)
        self.l_net = mlp(mask.l_dim + extra)
        self.b_net = mlp(mask.b_dim)
        self.kind_score = nn.Linear(hidden_dim, 3)      # one bias per kind, from the state
        self.o_score = nn.Linear(2 * hidden_dim, 1)
        self.l_score = nn.Linear(2 * hidden_dim, 1)
        self.b_score = nn.Linear(2 * hidden_dim, 1)
        if self.rank > 0:
            self.o_factor = nn.Linear(2 * hidden_dim, self.rank)
            self.l_factor = nn.Linear(2 * hidden_dim, self.rank)

    def forward(
        self,
        o: torch.Tensor,        # [R, N_o, O_DIM]
        l: torch.Tensor,        # [R, N_l, L_DIM]
        b: torch.Tensor,        # [R, N_b, B_DIM]
        kind: torch.Tensor,     # [R, N, 3]
        o_idx: torch.Tensor,    # [R, N]
        l_idx: torch.Tensor,    # [R, N]
        b_idx: torch.Tensor,    # [R, N]
        mask: torch.Tensor,     # [R, N] bool
        context: torch.Tensor,  # [R, context_dim]
    ) -> torch.Tensor:
        c = self.context_net(context)                                    # [R, H]
        if self.coords is not None:
            o = self.coords(o)                                           # the order's destination
            l = self.coords(l)                                           # the pad's position
        with_context = lambda block, net: torch.cat(
            [net(block), c.unsqueeze(1).expand(-1, block.shape[1], -1)], dim=-1)
        oc = with_context(o, self.o_net)                                 # [R, N_o, 2H]
        lc = with_context(l, self.l_net)
        bc = with_context(b, self.b_net)

        g_o = self.o_score(oc).squeeze(-1)                               # [R, N_o]
        g_l = self.l_score(lc).squeeze(-1)
        h_b = self.b_score(bc).squeeze(-1)

        dispatch = kind[..., 0]
        order_active = (kind[..., 0] + kind[..., 1]).clamp(0.0, 1.0)      # dispatch or defer
        logit = ((kind * self.kind_score(c).unsqueeze(1)).sum(-1)
                 + order_active * torch.gather(g_o, 1, o_idx)
                 + dispatch * torch.gather(g_l, 1, l_idx)
                 + dispatch * torch.gather(h_b, 1, b_idx))
        if self.rank > 0:
            u = self.o_factor(oc)                                        # [R, N_o, rank]
            v = self.l_factor(lc)                                        # [R, N_l, rank]
            u_flat = torch.gather(u, 1, o_idx.unsqueeze(-1).expand(-1, -1, self.rank))
            v_flat = torch.gather(v, 1, l_idx.unsqueeze(-1).expand(-1, -1, self.rank))
            logit = logit + dispatch * (u_flat * v_flat).sum(-1)
        return logit.masked_fill(~mask, torch.finfo(logit.dtype).min)
