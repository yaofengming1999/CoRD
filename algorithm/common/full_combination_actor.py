"""Full-capacity actor over the flat (order, return, battery) triplet (flat
layout): the FullCombination action expansion (F3). No factorisation -- every
candidate row is scored by MLPs on the whole triplet vector (block slices come
from ``algorithm/F2/menu.py`` so the actor follows the layout).
"""

from __future__ import annotations

import torch
from torch import nn

from algorithm.F2.menu import F_BATTERY, F_KIND, F_ORDER, FLAT_DIM


class FullCombinationActor(nn.Module):
    """Flat triplet policy with order/battery/return score decomposition."""

    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        if candidate_dim != FLAT_DIM:
            raise ValueError(
                f"full-combination actor expects candidate_dim={FLAT_DIM}, got {candidate_dim}"
            )

        def projection(width: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(width, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.kind_projection = projection(F_KIND.stop - F_KIND.start)
        self.order_projection = projection(F_ORDER.stop - F_ORDER.start)
        self.battery_projection = projection(F_BATTERY.stop - F_BATTERY.start)
        self.return_projection = projection(FLAT_DIM - F_BATTERY.stop)   # route + target blocks
        self.context_projection = projection(context_dim)
        self.order_score = projection(hidden_dim * 3)
        self.battery_score = projection(hidden_dim * 4)
        self.return_score = projection(hidden_dim * 5)
        self.interaction_score = projection(hidden_dim * 5)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        candidate_features: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        kind = self.kind_projection(candidate_features[..., F_KIND])
        order = self.order_projection(candidate_features[..., F_ORDER])
        battery = self.battery_projection(candidate_features[..., F_BATTERY])
        return_entity = self.return_projection(candidate_features[..., F_BATTERY.stop:])
        encoded_context = self.context_projection(context).unsqueeze(1).expand(
            -1,
            candidate_features.shape[1],
            -1,
        )
        order_input = torch.cat([kind, order, encoded_context], dim=-1)
        battery_input = torch.cat([kind, order, battery, encoded_context], dim=-1)
        return_input = torch.cat(
            [kind, order, battery, return_entity, encoded_context], dim=-1
        )
        interaction_input = return_input
        logits = (
            self.output(self.order_score(order_input)).squeeze(-1)
            + self.output(self.battery_score(battery_input)).squeeze(-1)
            + self.output(self.return_score(return_input)).squeeze(-1)
            + self.output(self.interaction_score(interaction_input)).squeeze(-1)
        )
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
