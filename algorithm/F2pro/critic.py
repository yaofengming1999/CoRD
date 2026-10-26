"""V(s) from the three blocks: the actor's encoders' shape, its own parameters.

The value of a state is read off the menu rather than off a candidate, so each
block is pooled over its rows -- mean and max, with padded rows excluded -- and
the three summaries are concatenated with the state context. Max pooling needs
the pad rows pushed out of the way, mean pooling needs them excluded from the
denominator; both are done with the block's own mask.
"""

from __future__ import annotations

import torch
from torch import nn

from algorithm.F2 import featmask


class TriVCritic(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()

        def mlp(width: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(width, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        self.context_net = mlp(context_dim)
        mask = featmask.active()
        self.o_net = mlp(mask.o_dim)
        self.l_net = mlp(mask.l_dim)
        self.b_net = mlp(mask.b_dim)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim), nn.ReLU(),   # context + 3 blocks x (mean, max)
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pool(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1).float()
        mean = (encoded * valid).sum(dim=1, keepdim=True) / count
        maximum = encoded.masked_fill(~valid, -1e9).max(dim=1, keepdim=True).values
        return torch.cat([mean.squeeze(1), maximum.squeeze(1)], dim=-1)

    def forward(self, o, l, b, o_mask, l_mask, b_mask, context) -> torch.Tensor:
        c = self.context_net(context)
        parts = [c,
                 self._pool(self.o_net(o), o_mask),
                 self._pool(self.l_net(l), l_mask),
                 self._pool(self.b_net(b), b_mask)]
        return self.value_net(torch.cat(parts, dim=-1)).squeeze(-1)
