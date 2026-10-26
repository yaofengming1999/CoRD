"""Value baselines.

  FactorVCritic          F2 default: V(s) from the factored menu (same encoders as the actor,
                         (o,l) rows and battery rows pooled), independent of the policy
  FactorQCritic          candidate Q in the same (o,l)|b factored form as the actor (V = E_pi[Q])
  CandidateQCritic       row-based candidate Q over expanded [N, D] rows (F3; F2 ablation)
  GlobalStateAgentCritic V(global pad-set state, own context) for the per-agent structures (F2 ablation)
  GlobalStateCritic      V(global pad-set state) for the fleet-level structures (AS1 MAPPO, AS3 GlobalDronesPPO)

Global state = the set of per-launchpad contexts of the pads that currently have
an available drone (one 23-dim row per pad: time, fleet-wide pending / idle /
moving / battery ratios, the pad's queue, drones, batteries, supply-demand gap,
...). No candidate-level (order, return pad, battery) information enters those
two critics; the pads are pooled order-invariantly (mean / max)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algorithm.F2 import featmask
from algorithm.F2.menu import B_DIM, OL_DIM


class FactorQCritic(nn.Module):
    """Candidate-conditioned Q critic in the same (o,l)|b factored form as the actor:

        Q(o,l,b) = q_k(kind) + q_ol(s,o,l) + q_b(s,b) + <u_q(s,o,l), v_q(s,b)>

    Scores the (o,l) rows and the battery rows once each and gathers the menu,
    so a value never needs the expanded [N, 35] rows (O(N_ol + N_b) like the actor).
    Own parameters; only the structure is shared with ``OLBSeparableActor``.
    """

    def __init__(self, context_dim: int, hidden_dim: int, rank: int = 8):
        super().__init__()
        self.rank = int(rank)

        def mlp(w: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(w, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        mask = featmask.active()
        self.context_net = mlp(context_dim)
        self.ol_net = mlp(mask.ol_dim)
        self.b_net = mlp(mask.b_dim)
        self.kind_score = nn.Linear(hidden_dim, 3)   # context-dependent bias per kind
        self.ol_score = nn.Linear(2 * hidden_dim, 1)
        self.b_score = nn.Linear(2 * hidden_dim, 1)
        if self.rank > 0:
            self.ol_factor = nn.Linear(2 * hidden_dim, self.rank)
            self.b_factor = nn.Linear(2 * hidden_dim, self.rank)

    def q_from_factors(self, ol, b, kind, ol_idx, b_idx, mask, context) -> torch.Tensor:
        """Same shapes as ``OLBSeparableActor.forward_factors``; masked entries -> 0."""
        c = self.context_net(context)                                     # [R, H]
        olc = torch.cat([self.ol_net(ol), c.unsqueeze(1).expand(-1, ol.shape[1], -1)], -1)
        bc = torch.cat([self.b_net(b), c.unsqueeze(1).expand(-1, b.shape[1], -1)], -1)
        g = self.ol_score(olc).squeeze(-1)                                 # [R, N_ol]
        h = self.b_score(bc).squeeze(-1)                                   # [R, N_b]
        dispatch = kind[..., 0]
        order_active = (kind[..., 0] + kind[..., 1]).clamp(0.0, 1.0)
        q = ((kind * self.kind_score(c).unsqueeze(1)).sum(-1)
             + order_active * torch.gather(g, 1, ol_idx)
             + dispatch * torch.gather(h, 1, b_idx))
        if self.rank > 0:
            u = self.ol_factor(olc)
            v = self.b_factor(bc)
            u_flat = torch.gather(u, 1, ol_idx.unsqueeze(-1).expand(-1, -1, self.rank))
            v_flat = torch.gather(v, 1, b_idx.unsqueeze(-1).expand(-1, -1, self.rank))
            q = q + dispatch * (u_flat * v_flat).sum(-1)
        return q.masked_fill(~mask, 0.0)


class FactorVCritic(nn.Module):
    """State value from the factored menu, without going through the policy:

        V(s) = f( c, pool(ol rows | c), pool(battery rows | c) )

    Same three encoders as the actor (own parameters); the (o,l) rows and the
    battery rows are pooled order-invariantly (masked mean and max), so a
    decision's value sees what is on the menu (order values, slack, routes,
    batteries) but not how the actor weights it. Cost O(N_ol + N_b), like the actor.
    """

    def __init__(self, context_dim: int, hidden_dim: int,
                 a_dim: int | None = None, b_dim: int | None = None):
        """``a_dim``/``b_dim`` are the two blocks' widths, defaulting to the
        (o,l) | b split. The (o,b) | l split produces the same kind of menu with
        different block contents, so it passes its own widths."""
        super().__init__()

        def mlp(w: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(w, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU())

        mask = featmask.active()
        self.context_net = mlp(context_dim)
        self.ol_net = mlp(mask.ol_dim if a_dim is None else int(a_dim))
        self.b_net = mlp(mask.b_dim if b_dim is None else int(b_dim))
        self.value_net = nn.Sequential(
            nn.Linear(5 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _pool(h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_f = mask.unsqueeze(-1).float()
        mean = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        maximum = h.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        maximum = torch.where(mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return mean, maximum

    def value_from_factors(self, ol, b, ol_mask, b_mask, context) -> torch.Tensor:
        """ol [R, N_ol, OL_DIM], b [R, N_b, B_DIM], row masks [R, N_ol] / [R, N_b], context [R, C] -> V [R]."""
        # The default F2 value head (critic_type=factor_v): V(s) straight from the pooled
        # menu, with no dependence on the actor's distribution. The alternatives are
        # FactorQCritic / CandidateQCritic, which score every entry and take V = E_pi[Q].
        c = self.context_net(context)                                     # [R, H]
        h_ol = self.ol_net(ol)                                            # [R, N_ol, H]
        h_b = self.b_net(b)                                               # [R, N_b, H]
        ol_mean, ol_max = self._pool(h_ol, ol_mask)
        b_mean, b_max = self._pool(h_b, b_mask)
        return self.value_net(torch.cat([c, ol_mean, ol_max, b_mean, b_max], dim=-1)).squeeze(-1)


class CandidateQCritic(nn.Module):
    """Row-based candidate Q critic (F3, and the F2 ``candidate_q`` ablation): scores every
    expanded [N, D] candidate row given the context."""

    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.candidate_net = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def q_values(
        self,
        candidate_features: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self.candidate_net(candidate_features)
        encoded_context = self.context_net(context).unsqueeze(1)
        q_values = self.q_head(
            torch.cat(
                [candidates, encoded_context.expand(-1, candidates.shape[1], -1)],
                dim=-1,
            )
        ).squeeze(-1)
        return q_values.masked_fill(~mask, 0.0)


class GlobalStateCritic(nn.Module):
    """V(s) from the set of active-pad contexts.

    pad_contexts [B, P, C], pad_mask [B, P]  ->  value [B]
    """

    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()
        self.pad_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pad_contexts: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.pad_net(pad_contexts)                              # [B, P, H]
        mask_f = pad_mask.unsqueeze(-1).float()
        mean = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        maximum = encoded.masked_fill(~pad_mask.unsqueeze(-1), -1e9).max(dim=1).values
        maximum = torch.where(pad_mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.value_net(
            torch.cat([mean, maximum, mean * maximum, mean - maximum], dim=-1)
        ).squeeze(-1)


def global_state_batch(contexts_list: list[np.ndarray], context_dim: int, device):
    """Batch T global states ([P_t, C] each) into pad_contexts [T, Pmax, C], pad_mask [T, Pmax]."""
    T = len(contexts_list)
    p_max = max((len(c) for c in contexts_list), default=1) or 1
    ctx = np.zeros((T, p_max, context_dim), dtype=np.float32)
    pmask = np.zeros((T, p_max), dtype=bool)
    for t, contexts in enumerate(contexts_list):
        n = len(contexts)
        if n:
            ctx[t, :n] = contexts; pmask[t, :n] = True
    return (torch.as_tensor(ctx, dtype=torch.float32, device=device),
            torch.as_tensor(pmask, dtype=torch.bool, device=device))


class GlobalStateAgentCritic(nn.Module):
    """V(s, agent) for the per-agent structures (SerializedMARL, SingleDronePPO).

    Same pooled global state as ``GlobalStateCritic`` plus the acting drone's
    own context, so the value knows *whose* return it is estimating:

    pad_contexts [B, P, C], pad_mask [B, P], own_context [B, C]  ->  value [B]
    """

    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()
        self.pad_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.own_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, pad_contexts: torch.Tensor, pad_mask: torch.Tensor, own_context: torch.Tensor
    ) -> torch.Tensor:
        encoded = self.pad_net(pad_contexts)                              # [B, P, H]
        mask_f = pad_mask.unsqueeze(-1).float()
        mean = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        maximum = encoded.masked_fill(~pad_mask.unsqueeze(-1), -1e9).max(dim=1).values
        maximum = torch.where(pad_mask.any(dim=1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.value_net(
            torch.cat([mean, maximum, mean * maximum, mean - maximum, self.own_net(own_context)], dim=-1)
        ).squeeze(-1)
