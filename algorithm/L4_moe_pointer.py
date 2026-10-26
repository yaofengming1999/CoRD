"""L4: MoE-Pointer (KDD'26)+GlobalDronesPPO -- aligned to the original single-agent design.

Reference: https://github.com/luiluizi/MoE-Pointer.git (Multi-Vehicle Dynamic
Pickup-Delivery Problem). The original is a SINGLE-agent encoder-decoder that,
once per decision frame, autoregressively decodes a *sequence* of pointer
choices across three phases -- request->courier, courier->node, drone->node --
with a Mixture-of-Experts decoder (a shared expert plus one phase-specialized
expert, sigmoid-gated), a pointer head (query . entity-reps), autoregressive
conditioning (SOS + previous-action embedding), and a single value head.

We map it to the drone problem as the paper's authors described:
  * choose courier for a request  ~  choose battery for an order      (phase 1)
  * choose node for the courier    ~  choose return launchpad for the drone (phase 2)
  * the chosen (order, battery) pair auto-links to one idle drone at the
    launchpad, so drone assignment is NOT a decision (the original's third
    phase is dropped).
Thus our version SELECTS TWICE, not three times. Order *sequence* is not a
learned decision in the original (requests are consumed in first-decodable /
FIFO order), so we keep FIFO order sequencing here too.

This is therefore a single-agent policy: per epoch, one encoder pass + one
autoregressive decode over all dispatchable orders (FIFO), two pointer picks
per dispatched order (battery, then return launchpad), one value, and one
joint log-prob = sum of every pick's log-prob. Credit is joint-per-epoch
(one V(s_t), one team reward per physical time), matching the original's
one-value-per-frame MAT training -- NOT the per-drone trajectories used by F1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from algorithm.F2 import featmask
from algorithm.common.torch_learner_base import BaseTorchMAPPOLearner
from algorithm.common.geometry_utils import distance_m, safe_ratio
from algorithm.F2.policy import _JointPlanDecision
from algorithm.AS3_global_drones_ppo import GlobalDronesPPOStructure
from simulator.model.enums import BatteryStatus, OrderStatus
from simulator.model.types import StateDict, TaskLevelActionDict

# Shared candidate feature width. Layout: [is_battery, is_launchpad, is_noassign]
# (3) + a 15-wide body reused by both stages (battery fills the first 7, return
# launchpad fills all 15). The 3-way kind prefix also identifies the MoE phase.
CANDIDATE_DIM = 18
# the unmasked joint-context width; the learner is built at the feature
# mask's context_dim, which equals this when nothing is dropped
CONTEXT_DIM = 23  # matches base _joint_context
STAGE_BATTERY = 0
STAGE_LAUNCHPAD = 1


# --------------------------------------------------------------------------- #
# MoE building blocks (the paper's core: shared + phase-specialized experts,
# sigmoid-gated; pointer decoder with cross-attention over encoded entities).
# --------------------------------------------------------------------------- #
class _ExpertFFN(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class _MoEFeedForward(nn.Module):
    """Shared expert + one specialized expert per decode phase, sigmoid-gated.

    Directly follows the upstream MoEFFN: y = (1-g)*shared + g*specialized,
    where the specialized expert is chosen by the current phase (here: battery
    stage vs return-launchpad stage), and g is a learned scalar gate.
    """

    def __init__(self, hidden_dim: int, num_stages: int = 2):
        super().__init__()
        self.shared = _ExpertFFN(hidden_dim)
        self.experts = nn.ModuleList(_ExpertFFN(hidden_dim) for _ in range(num_stages))
        self.gate = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.gate.bias, -3.0)
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(self, values: torch.Tensor, stage: int) -> torch.Tensor:
        shared = self.shared(values)
        specialized = self.experts[stage](values)
        gate = torch.sigmoid(self.gate(values))
        return (1.0 - gate) * shared + gate * specialized


class _MoEEncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, num_stages: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.moe = _MoEFeedForward(hidden_dim, num_stages)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, values, mask, stage):
        update, _ = self.attention(
            values, values, values, key_padding_mask=~mask, need_weights=False
        )
        values = self.norm1(values + update)
        values = self.norm2(values + self.moe(values, stage))
        return values.masked_fill(~mask.unsqueeze(-1), 0.0)


class _MoEDecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, num_stages: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.moe = _MoEFeedForward(hidden_dim, num_stages)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, tokens, token_mask, memory, memory_mask, causal_mask, stage):
        update, _ = self.self_attention(
            tokens, tokens, tokens,
            attn_mask=causal_mask, key_padding_mask=~token_mask, need_weights=False,
        )
        tokens = self.norm1(tokens + update)
        update, _ = self.cross_attention(
            tokens, memory, memory, key_padding_mask=~memory_mask, need_weights=False,
        )
        tokens = self.norm2(tokens + update)
        tokens = self.norm3(tokens + self.moe(tokens, stage))
        return tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)


class _MoEPointer(nn.Module):
    """MoE encoder over the current stage's candidates + causal MoE-pointer
    decoder. `pointer_logits` returns a score per candidate for one decode
    step, conditioned on the autoregressive history of prior picks.
    """

    def __init__(self, candidate_dim, context_dim, hidden_dim, heads, layers, num_stages):
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("MoE pointer hidden_dim must be divisible by heads")
        self.candidate_in = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.context_in = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU())
        self.action_in = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.sos = nn.Parameter(torch.randn(hidden_dim))
        self.enc_blocks = nn.ModuleList(
            _MoEEncoderBlock(hidden_dim, heads, num_stages) for _ in range(layers)
        )
        self.dec_blocks = nn.ModuleList(
            _MoEDecoderBlock(hidden_dim, heads, num_stages) for _ in range(layers)
        )
        self.dec_norm = nn.LayerNorm(hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = hidden_dim ** -0.5

    def pointer_logits(self, cand_feats, cand_mask, context, hist_feats, hist_mask, stage):
        values = self.candidate_in(cand_feats)
        for block in self.enc_blocks:
            values = block(values, cand_mask, stage)
        batch = cand_feats.shape[0]
        history = self.action_in(hist_feats)
        sos = self.sos.view(1, 1, -1).expand(batch, -1, -1)
        current = self.context_in(context).unsqueeze(1)
        tokens = torch.cat([sos, history, current], dim=1)
        ones = torch.ones((batch, 1), dtype=torch.bool, device=values.device)
        token_mask = torch.cat([ones, hist_mask, ones], dim=1)
        length = tokens.shape[1]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=values.device), diagonal=1
        )
        decoded = tokens
        for block in self.dec_blocks:
            decoded = block(decoded, token_mask, values, cand_mask, causal, stage)
        query = self.dec_norm(decoded)[:, -1:].contiguous()
        logits = (self.key(values) * query).sum(dim=-1) * self.scale
        return logits.masked_fill(~cand_mask, torch.finfo(logits.dtype).min)


class _ContextCritic(nn.Module):
    """Single state value V(s_t) from the epoch context (one per frame)."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context):
        return self.net(context).squeeze(-1)


# --------------------------------------------------------------------------- #
# Per-order decode records.
# --------------------------------------------------------------------------- #
@dataclass
class _Pending:
    context: np.ndarray
    battery_feats: np.ndarray
    battery_hist: np.ndarray
    battery_action: int
    battery_log_prob: float
    value: float


@dataclass
class _MoEPointerSample:
    context: np.ndarray
    value: float
    log_prob: float
    action: int  # unused placeholder to satisfy base API expectations
    battery_feats: np.ndarray
    battery_hist: np.ndarray
    battery_action: int
    launchpad_feats: np.ndarray
    launchpad_hist: np.ndarray
    launchpad_action: int
    has_launchpad: bool


# --------------------------------------------------------------------------- #
# Learner: single-agent, joint-per-epoch PPO over the MoE-pointer sequence.
# --------------------------------------------------------------------------- #
class TorchMoEPointerPPOLearner(BaseTorchMAPPOLearner):
    """One shared MoE-pointer network; one value and one summed log-prob per
    physical epoch; GAE over epochs (single-agent, like the original MAT
    training with a joint action per frame).
    """

    candidate_empty_error = "MoE-pointer requires a non-empty candidate stage"
    actor_context_label = "MoE-pointer context"

    def __init__(self, *, candidate_dim, context_dim, attention_heads=4,
                 attention_layers=2, moe_experts=2, use_ar=True,
                 update_attention_budget=250_000, **kwargs):
        self.context_dim = int(context_dim)
        self.num_stages = 2
        # bound on the padded attention work per pointer call in the PPO update
        # (sum over the chunk of (history+2)^2 + candidates^2); memory, not maths
        self.update_attention_budget = int(update_attention_budget)
        super().__init__(candidate_dim=candidate_dim, actor_context_dim=self.context_dim, **kwargs)
        self.attention_heads = int(attention_heads)
        self.attention_layers = int(attention_layers)
        self.moe_experts = int(moe_experts)
        self.use_ar = bool(use_ar)
        self.actor = _MoEPointer(
            self.candidate_dim, self.context_dim, self.hidden_dim,
            self.attention_heads, self.attention_layers, self.num_stages,
        ).to(self.device)
        self.critic = _ContextCritic(self.context_dim, self.hidden_dim).to(self.device)
        self._initialize_optimizers()
        self._transitions: list[dict] = []

    # ---- online decode ---------------------------------------------------- #
    def _pick(self, cand_feats, context, hist_feats, stage, training):
        feats = self._np2d(cand_feats)
        hist = self._np2d(hist_feats)
        feats_t = self._t(feats[None])
        mask_t = torch.ones((1, len(feats)), dtype=torch.bool, device=self.device)
        ctx_t = self._t(np.asarray(context, dtype=np.float32)[None])
        hist_t = self._t(hist[None])
        hist_mask_t = torch.ones((1, len(hist)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits = self.actor.pointer_logits(feats_t, mask_t, ctx_t, hist_t, hist_mask_t, stage)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample() if training or self.evaluation_action_mode == "stochastic" else torch.argmax(logits, dim=-1)
            log_prob = float(dist.log_prob(action).item())
        return int(action.item()), log_prob

    def begin_battery(self, battery_feats, context, hist_feats, *, training):
        context = self._as_vector(context, self.context_dim, self.actor_context_label)
        with torch.no_grad():
            value = float(self.critic(self._t(context[None])).item())
        index, log_prob = self._pick(battery_feats, context, hist_feats, STAGE_BATTERY, training)
        return index, _Pending(
            context=context,
            battery_feats=self._np2d(battery_feats),
            battery_hist=self._np2d(hist_feats),
            battery_action=index,
            battery_log_prob=log_prob,
            value=value,
        )

    def complete_launchpad(self, pending, launchpad_feats, hist_feats, *, training):
        index, log_prob = self._pick(launchpad_feats, pending.context, hist_feats, STAGE_LAUNCHPAD, training)
        sample = _MoEPointerSample(
            context=pending.context, value=pending.value,
            log_prob=pending.battery_log_prob + log_prob, action=pending.battery_action,
            battery_feats=pending.battery_feats, battery_hist=pending.battery_hist,
            battery_action=pending.battery_action,
            launchpad_feats=self._np2d(launchpad_feats), launchpad_hist=self._np2d(hist_feats),
            launchpad_action=index, has_launchpad=True,
        )
        return index, sample

    def finish_no_assign(self, pending):
        empty = np.zeros((1, self.candidate_dim), dtype=np.float32)
        return _MoEPointerSample(
            context=pending.context, value=pending.value,
            log_prob=pending.battery_log_prob, action=pending.battery_action,
            battery_feats=pending.battery_feats, battery_hist=pending.battery_hist,
            battery_action=pending.battery_action,
            launchpad_feats=empty, launchpad_hist=np.zeros((0, self.candidate_dim), dtype=np.float32),
            launchpad_action=0, has_launchpad=False,
        )

    # ---- storage: group samples by epoch, GAE over epochs ----------------- #
    def store_episode(self, samples, rewards, decision_times, episode_end_time, transition_groups=None):
        # The AS3 branch passes per-step groups; this learner regroups by
        # decision time itself (identical partition), so the hint is unused.
        del transition_groups
        if not samples:
            return
        if not (len(samples) == len(rewards) == len(decision_times)):
            raise ValueError(self.sample_length_error)
        groups: dict[int, list[int]] = {}
        for idx, time in enumerate(decision_times):
            groups.setdefault(int(time), []).append(idx)
        times = sorted(groups)
        ordered = [groups[t] for t in times]
        values = np.asarray([samples[g[0]].value for g in ordered], dtype=np.float64)
        epoch_rewards = np.asarray(
            [sum(float(rewards[i]) for i in g) for g in ordered], dtype=np.float64
        ) / self.reward_scale
        advantages = np.zeros(len(ordered), dtype=np.float64)
        gae = 0.0
        for k in range(len(ordered) - 1, -1, -1):
            terminal = k == len(ordered) - 1
            next_time = episode_end_time if terminal else times[k + 1]
            dt = max(0.0, float(next_time - times[k]))
            disc = self.gamma ** (dt / self.time_unit_s)
            trace = self.gae_lambda ** (dt / self.time_unit_s)
            next_v = 0.0 if terminal else values[k + 1]
            delta = epoch_rewards[k] + disc * next_v - values[k]
            gae = delta + (0.0 if terminal else disc * trace * gae)
            advantages[k] = gae
        returns = advantages + values
        for k, g in enumerate(ordered):
            rows = [samples[i] for i in g]
            self._transitions.append({
                "rows": rows,
                "context": rows[0].context,
                "old_log_prob": float(sum(r.log_prob for r in rows)),
                "advantage": float(advantages[k]),
                "return": float(returns[k]),
            })

    def clear_rollout(self):
        self._transitions = []

    def update(self, epochs=None):
        if not self._transitions:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "rollout_size": 0.0}
        n = len(self._transitions)
        adv = np.asarray([t["advantage"] for t in self._transitions], dtype=np.float32)
        if len(adv) > 1 and float(adv.std()) > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.ppo_epochs if epochs is None else max(1, int(epochs))):
            order = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                ids = order[start:start + self.batch_size]
                trs = [self._transitions[int(i)] for i in ids]
                joint_old_t = self._t(np.asarray([tr["old_log_prob"] for tr in trs], dtype=np.float32))
                adv_t = self._t(adv[ids])
                # Two-pass PPO step with bounded memory. A transition's picks can run
                # into the hundreds (FIFO loop over a backlog) with a long
                # autoregressive history, so all picks of a minibatch do not fit
                # through the attention stack at once. Pass 1 (no grad) scores every
                # pick in chunks and gives the joint log-prob per transition; the
                # clipped-surrogate weight dL/dlogpi is then a per-transition scalar,
                # so pass 2 backpropagates chunk by chunk with exactly that weight.
                picks = self._picks(trs)
                with torch.no_grad():
                    joint_new, ent_per_tr = self._joint_log_probs(picks, len(trs))
                ratio = torch.exp(torch.clamp(joint_new - joint_old_t, -20.0, 20.0))
                unclipped = ratio * adv_t
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
                # d(-min(unclipped, clipped))/d(joint) = -ratio*adv on the unclipped
                # branch, 0 on the clipped (constant) branch; mean over transitions
                weight = torch.where(unclipped <= clipped, -unclipped, torch.zeros_like(unclipped)) / len(trs)
                idx_all = torch.as_tensor([p[0] for p in picks], dtype=torch.long, device=self.device)
                n_picks_per_tr = torch.zeros(len(trs), device=self.device).index_add(
                    0, idx_all, torch.ones(len(picks), device=self.device))
                self.actor_optimizer.zero_grad()
                for chunk in self._chunks(picks):
                    lp, ent, idx = self._pick_log_probs(chunk)
                    # entropy: per-pick mean within a transition, mean over transitions
                    ent_w = self.entropy_coefficient / (n_picks_per_tr[idx] * len(trs))
                    (weight[idx] * lp - ent_w * ent).sum().backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                ent_t = ent_per_tr.mean()
                actor_loss = -torch.minimum(unclipped, clipped).mean() - self.entropy_coefficient * ent_t
                contexts = [tr["context"] for tr in trs]
                returns = [float(tr["return"]) for tr in trs]

                value = self.critic(self._t(np.stack(contexts).astype(np.float32)))
                critic_loss = self.value_coefficient * nn.functional.mse_loss(
                    value, self._t(np.asarray(returns, dtype=np.float32))
                )
                self.critic_optimizer.zero_grad(); critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                entropies.append(float(ent_t.detach().cpu()))
        self._transitions = []
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "rollout_size": float(n),
        }

    # ---- batched pointer evaluation ------------------------------------- #
    @staticmethod
    def _picks(transitions):
        """Flatten a minibatch into (transition index, stage, candidates, history, context, action)."""
        picks = []
        for ti, tr in enumerate(transitions):
            for row in tr["rows"]:
                picks.append((ti, STAGE_BATTERY, row.battery_feats, row.battery_hist, row.context, row.battery_action))
                if row.has_launchpad:
                    picks.append((ti, STAGE_LAUNCHPAD, row.launchpad_feats, row.launchpad_hist, row.context, row.launchpad_action))
        return picks

    def _chunks(self, picks):
        """Split picks into chunks whose padded attention work stays bounded
        (sum of (history+2)^2 + candidates^2 per chunk <= update_attention_budget)."""
        budget = float(self.update_attention_budget)
        chunk, load = [], 0.0
        for p in picks:
            cost = float(len(p[3]) + 2) ** 2 + float(len(p[2])) ** 2
            if chunk and load + cost > budget:
                yield chunk
                chunk, load = [], 0.0
            chunk.append(p); load += cost
        if chunk:
            yield chunk

    def _pick_log_probs(self, picks):
        """log pi(action) and entropy of each pick: one padded pointer call per
        stage. Padding is masked at every attention and at the logits, so this
        matches per-pick evaluation up to float rounding. Returns
        (log_prob [K], entropy [K], transition index [K]) in the picks' order."""
        K = len(picks); D = self.candidate_dim
        lp_all = torch.zeros(K, device=self.device); ent_all = torch.zeros(K, device=self.device)
        for stage in (STAGE_BATTERY, STAGE_LAUNCHPAD):
            pos = [k for k, p in enumerate(picks) if p[1] == stage]
            if not pos:
                continue
            sel = [picks[k] for k in pos]
            feats = [self._np2d(p[2]) for p in sel]; hists = [self._np2d(p[3]) for p in sel]
            n_max = max(len(f) for f in feats); h_max = max(len(h) for h in hists)
            cand = np.zeros((len(sel), n_max, D), dtype=np.float32); cand_mask = np.zeros((len(sel), n_max), dtype=bool)
            hist = np.zeros((len(sel), h_max, D), dtype=np.float32); hist_mask = np.zeros((len(sel), h_max), dtype=bool)
            for k, (f, h) in enumerate(zip(feats, hists)):
                cand[k, :len(f)] = f; cand_mask[k, :len(f)] = True
                if len(h):
                    hist[k, :len(h)] = h; hist_mask[k, :len(h)] = True
            ctx_t = self._t(np.stack([np.asarray(p[4], dtype=np.float32) for p in sel]))
            logits = self.actor.pointer_logits(
                self._t(cand), torch.as_tensor(cand_mask, device=self.device), ctx_t,
                self._t(hist), torch.as_tensor(hist_mask, device=self.device), stage,
            )
            dist = torch.distributions.Categorical(logits=logits)
            actions_t = torch.as_tensor([int(p[5]) for p in sel], dtype=torch.long, device=self.device)
            pos_t = torch.as_tensor(pos, dtype=torch.long, device=self.device)
            lp_all = lp_all.index_add(0, pos_t, dist.log_prob(actions_t))
            ent_all = ent_all.index_add(0, pos_t, dist.entropy())
        idx = torch.as_tensor([p[0] for p in picks], dtype=torch.long, device=self.device)
        return lp_all, ent_all, idx

    def _joint_log_probs(self, picks, n_transitions):
        """Joint log-prob (sum over its picks) and per-pick mean entropy of every
        transition, evaluated chunk by chunk."""
        joint = torch.zeros(n_transitions, device=self.device)
        ent_sum = torch.zeros(n_transitions, device=self.device)
        n_picks = torch.zeros(n_transitions, device=self.device)
        for chunk in self._chunks(picks):
            lp, ent, idx = self._pick_log_probs(chunk)
            joint = joint.index_add(0, idx, lp)
            ent_sum = ent_sum.index_add(0, idx, ent)
            n_picks = n_picks.index_add(0, idx, torch.ones(len(chunk), device=self.device))
        # entropy regularised per pick, so entropy_coefficient is comparable across methods
        return joint, ent_sum / n_picks.clamp_min(1.0)

    # ---- checkpoint ------------------------------------------------------- #
    def _checkpoint_metadata(self):
        return {
            "feature_schema": self.feature_schema,
            "candidate_dim": self.candidate_dim,
            "context_dim": self.context_dim,
            "attention_heads": self.attention_heads,
            "attention_layers": self.attention_layers,
            "moe_experts": self.moe_experts,
            "transition_mode": "moe_pointer_single_agent_two_stage",
        }

    def _validate_checkpoint(self, payload):
        if payload.get("transition_mode") != "moe_pointer_single_agent_two_stage":
            raise ValueError("checkpoint is not an aligned MoE-Pointer checkpoint")

    # ---- base abstract hooks (unused: this learner overrides store/update) - #
    def _sample_rollout_fields(self, sample):
        return {}

    def _actor_forward(self, batch):
        raise NotImplementedError

    def _critic_forward(self, batch):
        raise NotImplementedError

    def _rollout_array_fields(self):
        return {}

    # ---- tensor helpers --------------------------------------------------- #
    def _t(self, value):
        return torch.as_tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32, device=self.device)

    def _np2d(self, value):
        return np.nan_to_num(
            np.asarray(value, dtype=np.float32).reshape(-1, self.candidate_dim),
            nan=0.0, posinf=1e6, neginf=-1e6,
        )


# --------------------------------------------------------------------------- #
# Policy: single-agent, FIFO orders, two pointer picks, drone auto-linked.
# --------------------------------------------------------------------------- #
class MoEPointerOrderBatteryPolicy(GlobalDronesPPOStructure):
    """L4: aligned MoE-Pointer baseline.

    Reuses the joint-plan base only for candidate feasibility / pruning /
    context helpers; overrides ``plan`` to a single-agent FIFO sequence with
    two learned pointer picks (battery, return launchpad) per dispatched order.
    """

    dispatch_action_prefix = "moe_pointer_dispatch"
    joint_plan_config_key = "order_battery_moe_pointer_ppo"
    joint_plan_feature_schema = "moe_pointer_single_agent_two_stage_v1"
    learner_class = TorchMoEPointerPPOLearner
    candidate_dim = CANDIDATE_DIM

    def __init__(self, config: dict, *, seed: int = 0):
        stage_cfg = config.setdefault("drone_decision", {}).setdefault(self.joint_plan_config_key, {})
        stage_cfg.setdefault("max_candidates_per_launchpad", 4096)
        super().__init__(config, seed=seed)

    def _learner_kwargs(self, cfg: dict, *, seed: int) -> dict:
        return {
            "candidate_dim": CANDIDATE_DIM,
            "context_dim": featmask.active().context_dim,
            "max_candidates": self.max_candidates,
            "feature_schema": self.joint_plan_feature_schema,
            "hidden_dim": int(cfg.get("hidden_dim", 128)),
            "actor_lr": float(cfg.get("actor_lr", 3e-4)),
            "critic_lr": float(cfg.get("critic_lr", 1e-3)),
            "gamma": float(cfg.get("gamma", 0.99)),
            "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
            "clip_ratio": float(cfg.get("clip_ratio", 0.2)),
            "value_coefficient": float(cfg.get("value_coefficient", 0.5)),
            "entropy_coefficient": float(cfg.get("entropy_coefficient", 0.01)),
            "max_grad_norm": float(cfg.get("max_grad_norm", 10.0)),
            "batch_size": int(cfg.get("batch_size", 16)),
            "ppo_epochs": self.gradient_steps_per_episode,
            "time_unit_s": float(cfg.get("time_unit_s", 60.0)),
            "reward_scale": float(cfg.get("reward_scale", 25.0)),
            "device": str(cfg.get("device", "cuda")),
            "seed": int(cfg.get("seed", seed)),
            "attention_heads": int(cfg.get("attention_heads", 4)),
            "attention_layers": int(cfg.get("attention_layers", 2)),
            "moe_experts": int(cfg.get("moe_experts", 2)),
            "use_ar": bool(cfg.get("use_ar", True)),
        }

    def plan(self, state: StateDict, used_orders: set[str], used_batteries: set[str]):
        current_time = int(state.get("time", 0))
        actions: list[TaskLevelActionDict] = []
        planned_targets: dict[str, int] = {}
        planned_departures: dict[str, int] = {}
        history: list[np.ndarray] = []          # epoch autoregressive history
        epoch_samples: list[_MoEPointerSample] = []
        orders = state.get("orders", {})
        launchpads = state.get("launchpads", {})

        # FIFO, single global sequence: launchpads in id order, each queue FIFO.
        for launchpad_id in sorted(launchpads, key=str):
            launchpad = launchpads[launchpad_id]
            launchpad_id = str(launchpad_id)
            self.menu.clear_terminal_lock(state, launchpad_id, launchpad)
            remaining_drones = self.menu.available_drone_count(state, launchpad)
            active_order_id = self.menu.active_order_by_launchpad.get(launchpad_id)
            queue = [
                str(oid) for oid in launchpad.get("order_queue", [])
                if str(oid) not in used_orders and str(oid) in orders
                and orders[str(oid)].get("status") == OrderStatus.PENDING.value
                and (active_order_id is None or str(oid) == active_order_id)
            ]
            queue = self.menu.prune_order_ids(state, launchpad, queue, current_time, planned_departures)
            for order_id in queue:
                if remaining_drones <= 0:
                    break
                order = orders[order_id]
                battery_candidates = self._battery_candidates(
                    state, launchpad, order_id, used_batteries, planned_targets, planned_departures
                )
                if not battery_candidates:
                    continue  # nothing feasible to point at; leave order pending

                context = self.menu.context(
                    state, launchpad, used_orders, used_batteries, remaining_drones,
                    planned_targets, planned_departures,
                    agent_index=len(epoch_samples), agent_count=max(1, len(orders)),
                )
                # Stage A: pointer over feasible batteries + a no-assign token.
                bat_feats = np.stack([c["feat"] for c in battery_candidates])
                b_index, pending = self.learner.begin_battery(
                    bat_feats, context, self._hist(history), training=self.training
                )
                chosen_b = battery_candidates[b_index]
                history.append(bat_feats[b_index].copy())
                if chosen_b["battery_id"] is None:  # no-assign -> defer this order
                    sample = self.learner.finish_no_assign(pending)
                    if self.training:
                        epoch_samples.append(sample)
                    continue
                battery_id = chosen_b["battery_id"]

                # Stage B: pointer over feasible return launchpads for (order, battery).
                launchpad_candidates = self._launchpad_candidates(
                    state, launchpad, order, battery_id, current_time, planned_targets, planned_departures
                )
                if not launchpad_candidates:
                    sample = self.learner.finish_no_assign(pending)
                    if self.training:
                        epoch_samples.append(sample)
                    continue
                lp_feats = np.stack([c["feat"] for c in launchpad_candidates])
                l_index, sample = self.learner.complete_launchpad(
                    pending, lp_feats, self._hist(history), training=self.training
                )
                chosen_l = launchpad_candidates[l_index]
                history.append(lp_feats[l_index].copy())
                if self.training:
                    epoch_samples.append(sample)

                target_launchpad_id = chosen_l["launchpad_id"]
                actions.append({
                    "action_id": (
                        f"{self.dispatch_action_prefix}_{current_time}_{order_id}_"
                        f"{battery_id}_{target_launchpad_id}"
                    ),
                    "action_type": "order_battery_dispatch",
                    "order_id": order_id,
                    "battery_id": battery_id,
                    "target_launchpad_id": target_launchpad_id,
                    "feasibility": True,
                })
                used_orders.add(order_id)
                used_batteries.add(battery_id)
                remaining_drones -= 1
                planned_departures[launchpad_id] = planned_departures.get(launchpad_id, 0) + 1
                planned_targets[target_launchpad_id] = planned_targets.get(target_launchpad_id, 0) + 1
                if self.menu.active_order_by_launchpad.get(launchpad_id) == order_id:
                    self.menu.active_order_by_launchpad.pop(launchpad_id, None)

        if self.training and epoch_samples:
            for sample in epoch_samples:
                self._episode_decisions.append(
                    _JointPlanDecision(current_time, sample, "dispatch", "system", "system", None)
                )
        return actions

    # ---- candidate builders ---------------------------------------------- #
    def _local_battery_ids(self, launchpad, batteries, used_batteries):
        return [
            str(bid) for bid in (
                list(launchpad.get("charged_battery_ids", []))
                + list(launchpad.get("charging_battery_ids", []))
            )
            if str(bid) not in used_batteries and str(bid) in batteries
        ]

    def _battery_candidates(self, state, launchpad, order_id, used_batteries,
                            planned_targets, planned_departures):
        order = state["orders"][order_id]
        batteries = state.get("batteries", {})
        departure_id = str(launchpad.get("launchpad_id"))
        candidates: list[dict] = []
        for battery_id in self._local_battery_ids(launchpad, batteries, used_batteries):
            battery = batteries[battery_id]
            details = self.candidate_details_fn(order, battery) if self.candidate_details_fn else {}
            if not bool(details.get("feasible", False)):
                continue
            # at least one return launchpad must have projected capacity
            feasible_targets = [
                tid for tid, route in details.get("return_plans", {}).items()
                if bool(route.get("feasible", False))
                and state.get("launchpads", {}).get(str(tid)) is not None
                and self.menu.target_has_projected_capacity(
                    departure_id, state["launchpads"][str(tid)], planned_targets, planned_departures
                )
            ]
            if not feasible_targets:
                continue
            candidates.append({
                "battery_id": battery_id,
                "feat": self._battery_feature(battery, details),
            })
        if candidates:
            candidates.append({"battery_id": None, "feat": self._no_assign_feature()})
        return candidates

    def _launchpad_candidates(self, state, launchpad, order, battery_id, current_time,
                              planned_targets, planned_departures):
        batteries = state.get("batteries", {})
        launchpads = state.get("launchpads", {})
        departure_id = str(launchpad.get("launchpad_id"))
        details = self.candidate_details_fn(order, batteries[battery_id]) if self.candidate_details_fn else {}
        return_plans = self.menu.prune_return_plans(
            order, launchpad, launchpads, details.get("return_plans", {})
        )
        candidates: list[dict] = []
        for target_id, route in sorted(return_plans.items()):
            target = launchpads.get(str(target_id))
            if target is None or not bool(route.get("feasible", False)):
                continue
            if not self.menu.target_has_projected_capacity(
                departure_id, target, planned_targets, planned_departures
            ):
                continue
            candidates.append({
                "launchpad_id": str(target_id),
                "feat": self._launchpad_feature(order, launchpad, target, route,
                                                planned_targets, planned_departures),
            })
        return candidates

    # ---- feature vectors (shared 18-wide layout) -------------------------- #
    def _horizon(self):
        return max(1.0, float(self.config.get("simulation", {}).get("horizon_seconds", 1)))

    def _battery_feature(self, battery, details):
        capacity = float(battery.get("capacity_Wh", 0.0) or 0.0)
        charge = float(details.get("available_charge_Wh", battery.get("current_charge_Wh", 0.0)) or 0.0)
        completion = battery.get("charge_complete_at")
        body = [
            safe_ratio(charge, capacity),
            charge / 1500.0,
            capacity / 1500.0,
            float(battery.get("status") == BatteryStatus.CHARGING.value),
            0.0 if completion is None else max(0.0, float(completion) - 0.0) / self._horizon(),
            float(details.get("task_energy_ratio", 0.0) or 0.0),
            float(details.get("reserve_after_ratio", 0.0) or 0.0),
        ]
        return self._pack(STAGE_BATTERY, body)

    def _launchpad_feature(self, order, departure, target, route, planned_targets, planned_departures):
        horizon = self._horizon()
        departure_id = str(departure.get("launchpad_id"))
        target_id = str(target.get("launchpad_id"))
        projected = self.menu.projected_target_occupancy(departure_id, target, planned_targets, planned_departures)
        backlog = float(len(target.get("order_queue", [])))
        body = [
            float(route.get("task_energy_ratio", 0.0) or 0.0),
            float(route.get("return_energy_ratio", 0.0) or 0.0),
            float(route.get("reserve_after_ratio", 0.0) or 0.0),
            float(route.get("distance_loaded_m", 0.0) or 0.0) / 5000.0,
            float(route.get("distance_return_m", 0.0) or 0.0) / 5000.0,
            float(route.get("time_return_s", 0.0) or 0.0) / horizon,
            distance_m(order.get("destination"), target.get("current_location")) / 5000.0,
            backlog / 20.0,
            len(target.get("drone_ids", [])) / 20.0,
            float(target.get("reserved_return_slots", 0) or 0) / 20.0,
            int(planned_targets.get(target_id, 0)) / 20.0,
            int(planned_departures.get(target_id, 0)) / 20.0,
            projected / 20.0,
            len(target.get("charged_battery_ids", [])) / 20.0,
            len(target.get("charging_battery_ids", [])) / 20.0,
        ]
        return self._pack(STAGE_LAUNCHPAD, body)

    def _no_assign_feature(self):
        feat = np.zeros((CANDIDATE_DIM,), dtype=np.float32)
        feat[2] = 1.0  # is_no_assign
        return feat

    @staticmethod
    def _pack(stage, body):
        feat = np.zeros((CANDIDATE_DIM,), dtype=np.float32)
        feat[stage] = 1.0  # is_battery / is_launchpad kind flag
        b = np.asarray(body, dtype=np.float32)
        if b.shape[0] > CANDIDATE_DIM - 3:
            raise RuntimeError(f"MoE-pointer body width {b.shape[0]} exceeds {CANDIDATE_DIM - 3}")
        feat[3:3 + b.shape[0]] = b
        return feat

    def _hist(self, history):
        if not history:
            return np.zeros((0, CANDIDATE_DIM), dtype=np.float32)
        return np.stack(history)
