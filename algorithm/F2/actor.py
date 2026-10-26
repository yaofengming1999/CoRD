"""The proposed (o,l)|b separable actor.

``OLBSeparableActor`` scores an olb candidate (see
``algorithm/F2/menu.py``) as

    logit = k_kind(s) + g(s, o, l) + h(s, b) + <u(s, o, l), v(s, b)>

with a rank-``rank`` bilinear coupling between the (order, return launchpad)
side and the battery side. It keeps the same selection end as every other
actor (softmax over the flat menu); ``forward_factors`` scores a FactorMenu
without expanding the (o,l) x b grid into rows.
"""

from __future__ import annotations

import torch
from torch import nn

from algorithm.F2 import featmask
from algorithm.F2.menu import BATTERY, B_DIM, CROSS, F_ORDER, KIND, OL_DIM, OLB_DIM, ORDER, ORDER_DIM, TARGET, TARGET_DIM


class OLBSeparableActor(nn.Module):
    """(o,l)|b factorization:  logit = k_kind(s) + g(s,o,l) + h(s,b) + <u(s,o,l), v(s,b)>,
    where k_kind(s) is a context-dependent bias per entry kind (dispatch / defer / stop).

    Mirrors the offline OLBStudent exactly (the one that reached 0.69 held-out /
    0.79 in-sample vs D1):
      g,u <- order(5) + target(9) + dist_return + E_task/1500                                (16)
      h,v <- battery(6)                                                                  (6)
    E_*(Wh) are recovered battery-free from ratio x capacity (cross e/C times battery C/1500).
    The order x launchpad route interaction lives INSIDE g; the battery couples to
    it only through the rank-`rank` bilinear term (offline: saturates at rank 8).
    """

    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int, rank: int = 8,
                 a_dim: int | None = None, b_dim: int | None = None):
        """``a_dim``/``b_dim`` are the two blocks' widths.

        They default to the (o,l) | b split this actor was written for. The other
        grouping, (o,b) | l, reuses every line of the scoring below and differs
        only in what the two blocks hold, so it passes its own widths rather than
        duplicating the forward pass.
        """
        super().__init__()
        if candidate_dim != OLB_DIM:
            raise ValueError(f"separable actor expects candidate_dim={OLB_DIM}, got {candidate_dim}")
        if int(rank) < 0:
            raise ValueError("rank must be >= 0")
        self.rank = int(rank)
        # the default widths are what the feature mask leaves of the two blocks
        mask = featmask.active()
        self.a_dim = mask.ol_dim if a_dim is None else int(a_dim)
        self.b_dim = mask.b_dim if b_dim is None else int(b_dim)

        def mlp(w: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(w, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.context_net = mlp(context_dim)
        self.ol_net = mlp(self.a_dim)
        self.b_net = mlp(self.b_dim)
        self.kind_score = nn.Linear(hidden_dim, 3)   # context-dependent bias per kind (dispatch, defer, stop)
        self.ol_score = nn.Linear(2 * hidden_dim, 1)
        self.b_score = nn.Linear(2 * hidden_dim, 1)
        if self.rank > 0:
            self.ol_factor = nn.Linear(2 * hidden_dim, self.rank)
            self.b_factor = nn.Linear(2 * hidden_dim, self.rank)

    @staticmethod
    def inputs(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(o,l) and battery inputs from an expanded olb row (row path / row critic parity)."""
        cap = f[..., BATTERY.start + 2 : BATTERY.start + 3]            # capacity/1500 (0 for defer/stop)
        e_task = f[..., CROSS.start : CROSS.start + 1] * cap           # E_task/1500, battery-free
        d_ret = f[..., CROSS.start + 2 : CROSS.start + 3]
        ol = torch.cat([
            f[..., ORDER.start : ORDER.start + ORDER_DIM],             # order (without dest_xy)
            f[..., TARGET.start : TARGET.start + TARGET_DIM],          # target (without target_xy)
            d_ret, e_task,
        ], dim=-1)
        return ol, f[..., BATTERY]

    def forward_factors(
        self,
        ol: torch.Tensor,       # [R, N_ol, OL_DIM]
        b: torch.Tensor,        # [R, N_b, B_DIM]
        kind: torch.Tensor,     # [R, N, 3]   per flat menu entry
        ol_idx: torch.Tensor,   # [R, N]      entry -> (o,l) row
        b_idx: torch.Tensor,    # [R, N]      entry -> battery row
        mask: torch.Tensor,     # [R, N]
        context: torch.Tensor,  # [R, C]
    ) -> torch.Tensor:
        """Same logits as ``forward`` on the expanded rows, but g/u are evaluated
        once per (o,l) row and h/v once per battery; the menu is a gather over
        the (o,l) x b grid instead of an expanded row per triplet."""
        c = self.context_net(context)                                     # [R, H]
        olc = torch.cat([self.ol_net(ol), c.unsqueeze(1).expand(-1, ol.shape[1], -1)], -1)  # [R, N_ol, 2H]
        bc = torch.cat([self.b_net(b), c.unsqueeze(1).expand(-1, b.shape[1], -1)], -1)        # [R, N_b, 2H]
        g = self.ol_score(olc).squeeze(-1)                                 # [R, N_ol]
        h = self.b_score(bc).squeeze(-1)                                   # [R, N_b]
        dispatch = kind[..., 0]
        order_active = (kind[..., 0] + kind[..., 1]).clamp(0.0, 1.0)
        kind_bias = self.kind_score(c)                                     # [R, 3]  one bias per kind
        logit = ((kind * kind_bias.unsqueeze(1)).sum(-1)                   # [R, N]  pick the entry's kind
                 + order_active * torch.gather(g, 1, ol_idx)
                 + dispatch * torch.gather(h, 1, b_idx))
        if self.rank > 0:
            u = self.ol_factor(olc)                                        # [R, N_ol, rank]
            v = self.b_factor(bc)                                          # [R, N_b, rank]
            u_flat = torch.gather(u, 1, ol_idx.unsqueeze(-1).expand(-1, -1, self.rank))
            v_flat = torch.gather(v, 1, b_idx.unsqueeze(-1).expand(-1, -1, self.rank))
            logit = logit + dispatch * (u_flat * v_flat).sum(-1)
        return logit.masked_fill(~mask, torch.finfo(logit.dtype).min)

    def forward(self, f: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        kind = f[..., KIND]
        dispatch = kind[..., 0]
        order_active = (kind[..., 0] + kind[..., 1]).clamp(0.0, 1.0)
        ol_in, b_in = self.inputs(f)
        c = self.context_net(context).unsqueeze(1).expand(-1, f.shape[1], -1)
        olc = torch.cat([self.ol_net(ol_in), c], -1)
        bc = torch.cat([self.b_net(b_in), c], -1)
        kind_bias = self.kind_score(self.context_net(context)).unsqueeze(1)   # [R, 1, 3]
        logit = ((kind * kind_bias).sum(-1)
                 + order_active * self.ol_score(olc).squeeze(-1)
                 + dispatch * self.b_score(bc).squeeze(-1))
        if self.rank > 0:
            logit = logit + dispatch * (self.ol_factor(olc) * self.b_factor(bc)).sum(-1)
        return logit.masked_fill(~mask, torch.finfo(logit.dtype).min)
