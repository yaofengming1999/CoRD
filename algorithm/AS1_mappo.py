"""AS1: <OL+B+LowRank>+MAPPO -- synchronous MAPPO with one joint action per physical time."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithm.F2.menu import FactorMenu, empty_factor_menu, stack_factor_menus
from algorithm.common.torch_learner_base import BaseTorchMAPPOLearner
from algorithm.F2.critic import GlobalStateCritic, global_state_batch
from algorithm.common.full_combination_actor import FullCombinationActor
from algorithm.F2.actor import OLBSeparableActor
from algorithm.F2.policy import F2Policy, _JointPlanDecision
from algorithm.common.mappo_utils import mean_unique_agents_per_time, temporal_team_rewards
from algorithm.dispatch.presets import MAPPO


@dataclass(frozen=True)
class SynchronousDroneMAPPOActionSample:
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    agent_contexts: np.ndarray
    agent_mask: np.ndarray
    actions: np.ndarray
    agent_log_probs: np.ndarray
    log_prob: float
    value: float
    factors: list[FactorMenu] | None = None  # one FactorMenu per active agent
    global_state: np.ndarray | None = None   # critic input: active-pad contexts [P, C]


class _IndependentSharedActor(nn.Module):
    """One shared decentralized actor evaluated independently per drone.

    The per-drone scorer is the proposed (o,l)|b separable actor, so AS1
    differs from F2 only in the agent structure (synchronous joint step,
    central critic), not in how a candidate is scored.
    """

    # rows x menu entries scored in one call while acting; sized so the rank
    # term stays well under a GB (see forward_factors)
    ACT_CHUNK_ENTRIES = 2_000_000

    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int, rank: int = 8):
        super().__init__()
        self.actor = OLBSeparableActor(candidate_dim, context_dim, hidden_dim, rank=rank)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        agent_contexts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, agent_count, candidate_count, candidate_dim = (
            candidate_features.shape
        )
        logits = self.actor(
            candidate_features.reshape(
                batch_size * agent_count,
                candidate_count,
                candidate_dim,
            ),
            candidate_mask.reshape(batch_size * agent_count, candidate_count),
            agent_contexts.reshape(batch_size * agent_count, -1),
        )
        return logits.reshape(batch_size, agent_count, candidate_count)

    def forward_factors(self, fb: dict[str, torch.Tensor], batch_size: int, agent_count: int) -> torch.Tensor:
        """Factored scoring for all agents at once: fb holds [B*A, ...] factor tensors."""
        rows = fb["mask"].shape[0]
        entries = fb["mask"].shape[-1]
        # Scoring every agent in one call is the fast path, but the rank term is
        # [rows, entries, rank]: at meituanC scale (hundreds of drones acting in
        # the same step, menus in the 10^5 entries) that single allocation runs to
        # tens of GB and the GPU refuses it. Acting has no graph to keep, so the
        # rows can be scored in slices and the logits concatenated - same maths,
        # bounded peak. The learner's path keeps grad and is left untouched.
        chunk = max(1, self.ACT_CHUNK_ENTRIES // max(1, entries))
        if torch.is_grad_enabled() or chunk >= rows:
            logits = self.actor.forward_factors(
                fb["factor_ol"], fb["factor_b"], fb["factor_kind"], fb["factor_ol_idx"], fb["factor_b_idx"],
                fb["mask"], fb["context"],
            )
        else:
            logits = torch.cat([
                self.actor.forward_factors(
                    fb["factor_ol"][lo:lo + chunk], fb["factor_b"][lo:lo + chunk],
                    fb["factor_kind"][lo:lo + chunk], fb["factor_ol_idx"][lo:lo + chunk],
                    fb["factor_b_idx"][lo:lo + chunk], fb["mask"][lo:lo + chunk],
                    fb["context"][lo:lo + chunk],
                )
                for lo in range(0, rows, chunk)
            ], dim=0)
        return logits.reshape(batch_size, agent_count, -1)


class _FlatIndependentSharedActor(nn.Module):
    """Shared synchronous actor for the literature full-triplet adapter."""

    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.actor = FullCombinationActor(candidate_dim, context_dim, hidden_dim)

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        agent_contexts: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, agent_count, candidate_count, candidate_dim = candidate_features.shape
        logits = self.actor(
            candidate_features.reshape(batch_size * agent_count, candidate_count, candidate_dim),
            candidate_mask.reshape(batch_size * agent_count, candidate_count),
            agent_contexts.reshape(batch_size * agent_count, -1),
        )
        return logits.reshape(batch_size, agent_count, candidate_count)


class TorchSynchronousDroneMAPPO(BaseTorchMAPPOLearner):
    """Standard MAPPO for simultaneous drone agents.

    Actors share parameters but make independent decisions from the same
    pre-action state. PPO uses one shared team advantage per physical-time
    transition and a centralized critic over all active drone menus.
    """

    candidate_empty_error = "synchronous MAPPO requires a non-empty drone menu"
    actor_context_label = "synchronous MAPPO drone context"
    # One team V(s) per step is read at acting time and feeds that step's GAE.
    # A per-agent structure (AS4) cannot use it -- its value depends on which
    # drone is acting -- so it recomputes values when it stores, and turns this off.
    computes_acting_value = True

    @property
    def supports_factor_menu(self) -> bool:
        return getattr(self, "actor_type", "olb") != "flat"

    def __init__(self, *, candidate_dim: int, context_dim: int, interaction_rank: int = 8,
                 update_row_budget: int = 300_000, **kwargs):
        self.context_dim = int(context_dim)
        self.interaction_rank = int(interaction_rank)
        # bound on the padded menu work per actor call in the PPO update (memory, not maths)
        self.update_row_budget = int(update_row_budget)
        actor_type = str(kwargs.pop("actor_type", "olb"))
        self.actor_type = actor_type
        # AS1's critic is one team V(s) over the active-pad set, which is what a
        # team reward needs; it has no per-agent form. AS4 swaps the head, and it
        # pops critic_type before this runs, so what lands here is AS1's own and
        # must be the default. (AS4 then sets self.critic_type to its own value.)
        critic_type = str(kwargs.pop("critic_type", "global_state"))
        if critic_type != "global_state":
            raise ValueError(
                f"{type(self).__name__} reads a team V(s) over the pad set; "
                f"critic_type={critic_type!r} is supported by AS4 only"
            )
        self.critic_type = critic_type
        super().__init__(
            candidate_dim=candidate_dim,
            actor_context_dim=self.context_dim,
            **kwargs,
        )
        if actor_type == "flat":
            self.actor = _FlatIndependentSharedActor(
                self.candidate_dim,
                self.context_dim,
                self.hidden_dim,
            ).to(self.device)
        else:
            self.actor = _IndependentSharedActor(
                self.candidate_dim,
                self.context_dim,
                self.hidden_dim,
                rank=self.interaction_rank,
            ).to(self.device)
        # Centralised V(s) from the set of active-pad contexts (shared with AS3).
        self.critic = GlobalStateCritic(self.context_dim, self.hidden_dim).to(self.device)
        self._initialize_optimizers()
        self._joint_rollout: list[dict[str, Any]] = []

    def act_parallel(
        self,
        candidate_features: list[np.ndarray],
        agent_contexts: np.ndarray,
        global_state: np.ndarray | None = None,
        *,
        training: bool,
    ) -> tuple[np.ndarray, np.ndarray, SynchronousDroneMAPPOActionSample]:
        menus = list(candidate_features) if candidate_features and isinstance(candidate_features[0], FactorMenu) else None
        if menus is not None:
            # Factored round: the per-drone actor scores each drone's (o,l) rows and
            # battery rows once; rows are rebuilt only for the central critic (training).
            agent_count = len(menus)
            contexts = np.asarray(agent_contexts, dtype=np.float32).reshape(agent_count, self.context_dim)
            if any(len(m) == 0 for m in menus):
                raise ValueError(self.candidate_empty_error)
            if any(len(m) > self.max_candidates for m in menus):
                raise ValueError(self._candidate_overflow_error(max(map(len, menus))))
            arrays = stack_factor_menus(menus)
            candidate_mask = arrays["mask"]
            agent_mask = np.ones(agent_count, dtype=bool)
            fb = {k: torch.as_tensor(v, dtype=(torch.bool if k.endswith("mask") else torch.long if k.endswith("_idx") else torch.float32), device=self.device) for k, v in arrays.items()}
            fb["context"] = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
            features = np.zeros((agent_count, 0, self.candidate_dim), dtype=np.float32)
            features_t = None
        else:
            features, candidate_mask, contexts, agent_mask = self._pack_round(
                candidate_features,
                agent_contexts,
            )
            features_t = torch.as_tensor(features[None], dtype=torch.float32, device=self.device)
        candidate_mask_t = torch.as_tensor(
            candidate_mask[None], dtype=torch.bool, device=self.device
        )
        contexts_t = torch.as_tensor(contexts[None], dtype=torch.float32, device=self.device)
        agent_mask_t = torch.as_tensor(agent_mask[None], dtype=torch.bool, device=self.device)
        with torch.no_grad():
            if menus is not None:
                logits = self.actor.forward_factors(fb, 1, agent_count)
            else:
                logits = self.actor(features_t, candidate_mask_t, contexts_t)
            distribution = torch.distributions.Categorical(logits=logits)
            actions_t = (
                distribution.sample()
                if training or self.evaluation_action_mode == "stochastic"
                else torch.argmax(logits, dim=-1)
            )
            log_probs_t = distribution.log_prob(actions_t)
            if (not self.computes_acting_value or not training
                    or global_state is None or len(global_state) == 0):
                value_t = torch.zeros((1,), device=self.device)  # V only feeds GAE
            else:
                # centralised V from the active-pad context set (same input as AS3)
                value_t = self.critic(*global_state_batch([np.asarray(global_state, dtype=np.float32)], self.context_dim, self.device))
        actions = actions_t[0].cpu().numpy().astype(np.int64, copy=False)
        agent_log_probs = log_probs_t[0].cpu().numpy().astype(np.float32, copy=False)
        return actions, agent_log_probs, SynchronousDroneMAPPOActionSample(
            candidate_features=features,
            candidate_mask=candidate_mask,
            agent_contexts=contexts,
            agent_mask=agent_mask,
            actions=actions,
            agent_log_probs=agent_log_probs,
            log_prob=float((log_probs_t * agent_mask_t.float()).sum().item()),
            value=float(value_t.item()),
            factors=menus,
            global_state=global_state,
        )

    def store_joint_episode(
        self,
        samples: list[list[SynchronousDroneMAPPOActionSample]],
        rewards: list[float],
        decision_times: list[int],
        episode_end_time: int,
    ) -> None:
        if not samples:
            return
        if not (len(samples) == len(rewards) == len(decision_times)):
            raise ValueError("synchronous MAPPO joint episode fields must align")
        values = np.asarray(
            [float(group[0].value) if group else 0.0 for group in samples],
            dtype=np.float64,
        )
        scaled_rewards = np.asarray(rewards, dtype=np.float64) / self.reward_scale
        advantages = np.zeros(len(samples), dtype=np.float64)
        gae = 0.0
        for index in range(len(samples) - 1, -1, -1):
            terminal = index == len(samples) - 1
            next_time = episode_end_time if terminal else decision_times[index + 1]
            delta_t = max(0.0, float(next_time - decision_times[index]))
            discount = self.gamma ** (delta_t / self.time_unit_s)
            trace_discount = self.gae_lambda ** (delta_t / self.time_unit_s)
            next_value = 0.0 if terminal else values[index + 1]
            delta = scaled_rewards[index] + discount * next_value - values[index]
            gae = delta + (0.0 if terminal else discount * trace_discount * gae)
            advantages[index] = gae
        returns = advantages + values
        for samples_at_time, advantage, return_value in zip(
            samples,
            advantages,
            returns,
        ):
            if not samples_at_time:
                continue
            self._joint_rollout.append(
                {
                    "sample": samples_at_time[0],
                    "advantage": float(advantage),
                    "return": float(return_value),
                }
            )

    def update(self, epochs: int | None = None) -> dict[str, float]:
        if not self._joint_rollout:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "rollout_size": 0.0}
        rollout_size = len(self._joint_rollout)
        advantages = np.asarray(
            [row["advantage"] for row in self._joint_rollout], dtype=np.float32
        )
        if len(advantages) > 1 and float(advantages.std()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        update_epochs = self.ppo_epochs if epochs is None else max(1, int(epochs))
        for _ in range(update_epochs):
            indices = self._rng.permutation(rollout_size)
            for start in range(0, rollout_size, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                rows = [self._joint_rollout[int(index)] for index in batch_indices]
                adv_rows = torch.as_tensor(advantages[batch_indices], dtype=torch.float32, device=self.device)
                pairs = self._agent_pairs(rows)
                if pairs is None:   # flat actor (literature adapter): keep the dense path
                    batch = self._batch_rows(rows, advantages[batch_indices])
                    logits = self.actor(batch["candidate_features"], batch["candidate_mask"], batch["agent_contexts"])
                    distribution = torch.distributions.Categorical(logits=logits)
                    ratios = torch.exp(distribution.log_prob(batch["actions"]) - batch["old_log_probs"])
                    active = batch["agent_mask"].float()
                    expanded_advantage = batch["advantage"].unsqueeze(1)
                    per_agent_objective = torch.minimum(
                        ratios * expanded_advantage,
                        torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * expanded_advantage,
                    )
                    objective = (per_agent_objective * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
                    entropy = (distribution.entropy() * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
                    entropy_mean = entropy.mean()
                    actor_loss = -objective.mean() - self.entropy_coefficient * entropy_mean
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                else:
                    # Each agent contributes its own clipped-surrogate term (the objective is
                    # a mean over the transition's agents, then over transitions), so the loss
                    # is separable per (transition, agent) pair: the pairs go through the actor
                    # in memory-bounded chunks and the gradients accumulate. Identical to one
                    # pass, but never builds the [B, A, N] tensor that OOMs on a busy pad.
                    owner = torch.as_tensor([p[3] for p in pairs], dtype=torch.long, device=self.device)
                    counts = torch.bincount(owner, minlength=len(rows)).clamp_min(1).float()
                    self.actor_optimizer.zero_grad()
                    obj_sum = torch.zeros((), device=self.device)
                    ent_sum = torch.zeros(len(rows), device=self.device)
                    for chunk in self._pair_chunks(pairs):
                        arrays = stack_factor_menus([p[0] for p in chunk])
                        fb = {k: torch.as_tensor(v, dtype=(torch.bool if k.endswith("mask") else torch.long if k.endswith("_idx") else torch.float32), device=self.device) for k, v in arrays.items()}
                        fb["context"] = torch.as_tensor(np.stack([p[1] for p in chunk]), dtype=torch.float32, device=self.device)
                        logits = self.actor.actor.forward_factors(
                            fb["factor_ol"], fb["factor_b"], fb["factor_kind"],
                            fb["factor_ol_idx"], fb["factor_b_idx"], fb["mask"], fb["context"],
                        )
                        distribution = torch.distributions.Categorical(logits=logits)
                        own = torch.as_tensor([p[3] for p in chunk], dtype=torch.long, device=self.device)
                        actions_t = torch.as_tensor([p[2] for p in chunk], dtype=torch.long, device=self.device)
                        old_lp = torch.as_tensor([p[4] for p in chunk], dtype=torch.float32, device=self.device)
                        ratios = torch.exp(distribution.log_prob(actions_t) - old_lp)
                        adv = adv_rows[own]
                        per_agent = torch.minimum(
                            ratios * adv,
                            torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv,
                        )
                        ent = distribution.entropy()
                        w = 1.0 / (counts[own] * len(rows))   # mean over agents, then over transitions
                        (-(w * per_agent).sum() - self.entropy_coefficient * (w * ent).sum()).backward()
                        obj_sum = obj_sum + (w * per_agent).detach().sum()
                        ent_sum = ent_sum.index_add(0, own, ent.detach())
                    entropy_mean = (ent_sum / counts).mean()
                    actor_loss = -obj_sum - self.entropy_coefficient * entropy_mean
                    batch = {"pad_critic_inputs": global_state_batch(
                        [row["sample"].global_state if row["sample"].global_state is not None
                         else np.zeros((0, self.context_dim), dtype=np.float32) for row in rows],
                        self.context_dim, self.device,
                    ), "return": torch.as_tensor([row["return"] for row in rows], dtype=torch.float32, device=self.device),
                        "rows": rows}
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value = self._critic_forward(batch)      # overridable: AS4 adds the own context
                critic_loss = self.value_coefficient * nn.functional.mse_loss(
                    value,
                    batch["return"],
                )
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.detach().cpu().item()))
                critic_losses.append(float(critic_loss.detach().cpu().item()))
                entropies.append(float(entropy_mean.detach().cpu().item()))
        self._joint_rollout = []
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "rollout_size": float(rollout_size),
        }

    def _pack_round(self, candidate_features, agent_contexts):
        agent_count = len(candidate_features)
        if agent_count == 0:
            raise ValueError("synchronous MAPPO requires at least one active drone")
        contexts = np.asarray(agent_contexts, dtype=np.float32).reshape(
            agent_count, self.context_dim
        )
        menus = [
            np.asarray(menu, dtype=np.float32).reshape(-1, self.candidate_dim)
            for menu in candidate_features
        ]
        if any(len(menu) == 0 for menu in menus):
            raise ValueError(self.candidate_empty_error)
        if any(len(menu) > self.max_candidates for menu in menus):
            raise ValueError(self._candidate_overflow_error(max(map(len, menus))))
        max_candidates = max(map(len, menus))
        features = np.zeros(
            (agent_count, max_candidates, self.candidate_dim), dtype=np.float32
        )
        masks = np.zeros((agent_count, max_candidates), dtype=bool)
        for index, menu in enumerate(menus):
            features[index, : len(menu)] = menu
            masks[index, : len(menu)] = True
        return features, masks, contexts, np.ones(agent_count, dtype=bool)

    @staticmethod
    def _agent_pairs(rows):
        """Flatten transitions into (menu, context, action, transition index, old_log_prob)
        pairs, one per ACTIVE agent. None when the samples are not factored (flat actor)."""
        if not all(row["sample"].factors is not None for row in rows):
            return None
        pairs = []
        for t, row in enumerate(rows):
            sample = row["sample"]
            for a, menu in enumerate(sample.factors):
                if not bool(sample.agent_mask[a]):
                    continue
                pairs.append((menu, sample.agent_contexts[a], int(sample.actions[a]), t, float(sample.agent_log_probs[a])))
        return pairs

    def _pair_chunks(self, pairs):
        """Split (transition, agent) pairs into chunks whose padded menu work stays
        bounded: sum over the chunk of (menu entries + factor rows) <= update_row_budget."""
        budget = float(self.update_row_budget)
        chunk, load = [], 0.0
        for p in pairs:
            m = p[0]
            cost = float(len(m) + m.ol.shape[0] + m.b.shape[0])
            if chunk and load + cost > budget:
                yield chunk
                chunk, load = [], 0.0
            chunk.append(p); load += cost
        if chunk:
            yield chunk

    def _batch_rows(self, rows, advantages):
        max_agents = max(row["sample"].candidate_mask.shape[0] for row in rows)
        max_candidates = max(
            row["sample"].candidate_mask.shape[1] for row in rows
        )
        batch_size = len(rows)
        factored = all(row["sample"].factors is not None for row in rows)
        if factored:
            # every (batch, agent) slot gets a menu; absent agents get a 1-entry STOP menu
            flat_menus: list = []
            for row in rows:
                menus = list(row["sample"].factors)
                flat_menus.extend(menus + [empty_factor_menu()] * (max_agents - len(menus)))
            arrays = stack_factor_menus(flat_menus)
            max_candidates = arrays["mask"].shape[1]
        # the factored path scores menus, not rows: materialising [B, A, N, D] zeros there
        # would waste hundreds of MB of device memory on a busy meituan pad
        features = (
            np.zeros((0,), dtype=np.float32) if factored
            else np.zeros((batch_size, max_agents, max_candidates, self.candidate_dim), dtype=np.float32)
        )
        candidate_mask = np.zeros((batch_size, max_agents, max_candidates), dtype=bool)
        contexts = np.zeros((batch_size, max_agents, self.context_dim), dtype=np.float32)
        agent_mask = np.zeros((batch_size, max_agents), dtype=bool)
        actions = np.zeros((batch_size, max_agents), dtype=np.int64)
        old_log_probs = np.zeros((batch_size, max_agents), dtype=np.float32)
        for batch_index, row in enumerate(rows):
            sample = row["sample"]
            agents, candidates = sample.candidate_mask.shape[:2]
            if not factored:
                features[batch_index, :agents, :candidates] = sample.candidate_features
            candidate_mask[batch_index, :agents, :candidates] = sample.candidate_mask
            contexts[batch_index, :agents] = sample.agent_contexts
            agent_mask[batch_index, :agents] = sample.agent_mask
            actions[batch_index, :agents] = sample.actions
            old_log_probs[batch_index, :agents] = sample.agent_log_probs
            if agents < max_agents:
                candidate_mask[batch_index, agents:, 0] = True
        tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=self.device)
        extra = {"rows": rows, "pad_critic_inputs": global_state_batch(
            [row["sample"].global_state if row["sample"].global_state is not None else np.zeros((0, self.context_dim), dtype=np.float32) for row in rows],
            self.context_dim, self.device,
        )}
        if factored:
            extra.update({k: tensor(v, torch.bool if k == "mask" else torch.long if k.endswith("_idx") else torch.float32) for k, v in arrays.items()})
            extra["context"] = tensor(contexts.reshape(batch_size * max_agents, self.context_dim), torch.float32)
            candidate_mask = arrays["mask"].reshape(batch_size, max_agents, max_candidates)
        return {
            **extra,
            "candidate_features": tensor(features, torch.float32),
            "candidate_mask": tensor(candidate_mask, torch.bool),
            "agent_contexts": tensor(contexts, torch.float32),
            "agent_mask": tensor(agent_mask, torch.bool),
            "actions": tensor(actions, torch.long),
            "old_log_probs": tensor(old_log_probs, torch.float32),
            "advantage": tensor(advantages, torch.float32),
            "return": tensor([row["return"] for row in rows], torch.float32),
        }

    def clear_rollout(self) -> None:
        super().clear_rollout()
        self._joint_rollout = []

    def _sample_rollout_fields(self, sample):
        return {}

    def _actor_forward(self, batch):
        return self.actor(
            batch["candidate_features"],
            batch["candidate_mask"],
            batch["agent_contexts"],
        )

    def _critic_forward(self, batch):
        return self.critic(*batch["pad_critic_inputs"])

    def _rollout_array_fields(self):
        return {}

    def _checkpoint_metadata(self):
        return {
            "candidate_dim": self.candidate_dim,
            "context_dim": self.context_dim,
            "max_candidates": self.max_candidates,
            "feature_schema": self.feature_schema,
            "hidden_dim": self.hidden_dim,
            "interaction_rank": self.interaction_rank,
            "transition_mode": "synchronous_mappo_shared_actor_central_critic",
        }

    def _validate_checkpoint(self, payload):
        if payload.get("feature_schema") != self.feature_schema:
            raise ValueError("synchronous MAPPO feature schema is incompatible")
        if int(payload.get("interaction_rank", self.interaction_rank)) != self.interaction_rank:
            raise ValueError("checkpoint interaction_rank is incompatible with current policy")


class MAPPOStructure(F2Policy):
    """Agent structure MAPPO on top of F2: every waiting drone acts *simultaneously*
    at each physical step (one shared actor evaluated per drone on its pad menu),
    conflicts between simultaneous picks are resolved by rule, and the joint step
    is trained with a central critic. Learner must provide ``act_parallel`` and
    ``store_joint_episode``. Used by AS1 (this file) and L3 (MAPDP pointer)."""

    agent_structure = MAPPO

    def __init__(self, config: dict, *, seed: int = 0):
        self._joint_samples: dict[int, Any] = {}
        # (time, drone_id, order_id) of the dispatches the conflict rule accepted.
        # Under this decode a drone's chosen order is not necessarily the order it
        # flew: every drone picks from the same pre-commitment menu, so two can name
        # the same order and only the first in fleet order keeps it. AS1 itself never
        # reads this, because its credit is the step's team reward whatever happened,
        # but a per-drone credit assignment has to, or the loser is paid for the
        # winner's delivery and the reward is counted twice.
        self._accepted_dispatches: set[tuple[int, str, str]] = set()
        super().__init__(config, seed=seed)

    def clear_episode(self) -> None:
        self._accepted_dispatches = set()
        super().clear_episode()

    def dispatch_was_accepted(self, decision) -> bool:
        """Did this recorded decision actually become an action?

        A non-dispatch decision (defer, stop) commits nothing to an order, so it
        is counted as accepted: there is no order whose reward could be misplaced.
        """
        if decision.kind != "dispatch" or decision.order_id is None or decision.drone_id is None:
            return True
        return (int(decision.time), str(decision.drone_id), str(decision.order_id)) in self._accepted_dispatches

    def plan(
        self,
        state: StateDict,
        used_orders: set[str],
        used_batteries: set[str],
    ) -> list[TaskLevelActionDict]:
        del used_orders, used_batteries
        self._begin_plan(state)
        current_time = int(state.get("time", 0))
        active = self._collect_fleet_menus(state)

        if not active:
            return []
        # centralised critic input: the active-pad context set (shared with AS3); training only
        global_state = self.menu.global_state(state) if self.training else None
        actions_by_agent, _, joint_sample = self.learner.act_parallel(
            [item[5] for item in active],
            np.stack([item[4] for item in active]),
            global_state,
            training=self.training,
        )
        selected = [
            item[3][int(action)] for item, action in zip(active, actions_by_agent)
        ]
        if self.training:
            self._joint_samples[current_time] = joint_sample
            for item, candidate in zip(active, selected):
                self._episode_decisions.append(
                    _JointPlanDecision(
                        current_time,
                        joint_sample,
                        candidate.kind,
                        item[1],
                        item[0],
                        candidate.order_id,
                    )
                )
        return self._resolve_conflicts(state, active, selected, current_time)

    def _resolve_conflicts(self, state, active, selected, current_time):
        actions = []
        accepted_orders: set[str] = set()
        accepted_batteries: set[str] = set()
        planned_targets: dict[str, int] = defaultdict(int)
        planned_departures: dict[str, int] = defaultdict(int)
        deferred_launchpads: set[str] = set()
        for active_index, (
            (drone_id, launchpad_id, launchpad, candidates, _context, _menu),
            candidate,
        ) in enumerate(zip(active, selected)):
            accepted = False
            if candidate.kind == "defer":
                if launchpad_id not in deferred_launchpads and not planned_departures[launchpad_id]:
                    if candidate.order_id is not None:
                        self.menu.active_order_by_launchpad[launchpad_id] = candidate.order_id
                    deferred_launchpads.add(launchpad_id)
                accepted = launchpad_id in deferred_launchpads
            elif candidate.kind == "dispatch":
                if (
                    candidate.order_id is None
                    or candidate.battery_id is None
                    or candidate.return_launchpad_id is None
                    or candidate.order_id in accepted_orders
                    or candidate.battery_id in accepted_batteries
                ):
                    accepted = False
                else:
                    target = state.get("launchpads", {}).get(candidate.return_launchpad_id)
                    if target is not None and self.menu.target_has_projected_capacity(
                        launchpad_id,
                        target,
                        planned_targets,
                        planned_departures,
                    ):
                        accepted = True
                        accepted_orders.add(candidate.order_id)
                        accepted_batteries.add(candidate.battery_id)
                        planned_departures[launchpad_id] += 1
                        planned_targets[candidate.return_launchpad_id] += 1
            self._metrics.record_group(
                order_ids=[
                    item.order_id
                    for item in candidates
                    if item.kind == "dispatch" and item.order_id is not None
                ],
                battery_ids=[
                    item.battery_id
                    for item in candidates
                    if item.kind == "dispatch" and item.battery_id is not None
                ],
                selected_count=int(accepted and candidate.kind == "dispatch"),
                available_drones=1,
                selected_kind=candidate.kind,
            )
            if accepted and candidate.kind == "dispatch":
                self._accepted_dispatches.add(
                    (int(current_time), str(drone_id), str(candidate.order_id))
                )
                actions.append({
                    "action_id": (
                        f"{self.dispatch_action_prefix}_{current_time}_{drone_id}_"
                        f"{candidate.order_id}_{candidate.battery_id}_{candidate.return_launchpad_id}"
                    ),
                    "action_type": "order_battery_dispatch",
                    "order_id": candidate.order_id,
                    "battery_id": candidate.battery_id,
                    "target_launchpad_id": candidate.return_launchpad_id,
                    "drone_id": drone_id,
                    "feasibility": True,
                })
        return actions

    def finish_episode(
        self,
        on_time_gmv: float,
        *,
        reward_events: list[dict] | None = None,
        episode_end_time: int | None = None,
        gradient_steps: int | None = None,
        **kwargs,
    ) -> dict[str, float]:
        del on_time_gmv, kwargs
        decision_count = len(self._episode_decisions)
        if not self.training:
            result = {
                "order_battery_actor_loss": 0.0,
                "order_battery_critic_loss": 0.0,
                "order_battery_entropy": 0.0,
                "order_battery_rollout_size": 0.0,
                "order_battery_decisions": 0.0,
                "order_battery_feature_rows": float(self.menu.feature_rows),
            }
            result.update(self._metrics.summary())
            self.clear_episode()
            return result
        grouped: dict[int, list[Any]] = defaultdict(list)
        for decision in self._episode_decisions:
            grouped[int(decision.time)].append(self._joint_samples[int(decision.time)])
        times = sorted(grouped)
        rewards = temporal_team_rewards(times, reward_events)
        terminal_time = int(
            episode_end_time if episode_end_time is not None else (times[-1] if times else 0)
        )
        self.learner.store_joint_episode(
            [grouped[time] for time in times],
            rewards,
            times,
            terminal_time,
        )
        epochs = self.gradient_steps_per_episode if gradient_steps is None else int(gradient_steps)
        stats = self.learner.update(epochs=epochs)
        result = {
            "order_battery_actor_loss": float(stats["actor_loss"]),
            "order_battery_critic_loss": float(stats["critic_loss"]),
            "order_battery_entropy": float(stats["entropy"]),
            "order_battery_rollout_size": float(stats["rollout_size"]),
            "order_battery_decisions": float(decision_count),
            "order_battery_feature_rows": float(self.menu.feature_rows),
            "order_battery_active_drones_per_time": mean_unique_agents_per_time(
                [(decision.time, decision.drone_id or "unknown") for decision in self._episode_decisions]
            ),
        }
        result.update(self._metrics.summary())
        self.clear_episode()
        return result

    def clear_episode(self) -> None:
        super().clear_episode()
        self._joint_samples = {}


class AS1MAPPOOrderBatteryPolicy(MAPPOStructure):
    """AS1: MAPPO structure with the proposed (o,l)|b actor per drone and a central joint critic."""

    dispatch_action_prefix = "drone_synchronous_mappo_dispatch"
    joint_plan_config_key = "order_battery_drone_synchronous_mappo"
    joint_plan_feature_schema = "joint_dispatch_drone_synchronous_mappo_olb35_v3"
    learner_class = TorchSynchronousDroneMAPPO

    def _learner_kwargs(self, cfg: dict, *, seed: int) -> dict:
        return {
            "candidate_dim": self.candidate_dim,
            # the masked width, as in F2Policy: the context arrives already
            # projected, so a learner built at the raw 23 would reshape wrongly
            "context_dim": self.feat_mask.context_dim,
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
            "batch_size": int(cfg.get("batch_size", 8)),
            "ppo_epochs": self.gradient_steps_per_episode,
            "time_unit_s": float(cfg.get("time_unit_s", 60.0)),
            "reward_scale": float(cfg.get("reward_scale", 25.0)),
            "device": str(cfg.get("device", "cuda")),
            "seed": int(cfg.get("seed", seed)),
            "interaction_rank": int(cfg.get("interaction_rank", 8)),
            # only when set, so AS1 keeps its own default; AS4 reads it to choose
            # between V(s, drone) and F2's pooled-menu V(s)
            **({"critic_type": str(cfg["critic_type"])}
               if cfg.get("critic_type") is not None else {}),
        }
