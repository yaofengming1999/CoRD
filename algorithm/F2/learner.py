"""F2Learner -- PPO over per-decision candidate menus (one class).

Each decision of the policy is a softmax over its own feasible menu. Menus
differ in length, so the rollout stores every menu at its true length and each
PPO minibatch is padded only to its own longest menu. The learner scores a
``FactorMenu`` (ol rows + battery rows + index grid) directly with the (o,l)|b
actor -- no expanded [N, D] rows anywhere on the default path -- and keeps the
expanded-row path for the full-combination actor (F3).

Knobs (both set by the policy from the yaml section):
  actor_type   "olb"  (o,l)|b separable actor on FactorMenu inputs           (F2, AS2)
               "flat" full-combination actor on expanded flat rows          (F3)
  critic_type  "factor_v"     V(s) from the pooled factored menu, policy-independent (F2 default)
               "factor_q"     Q(o,l,b) in the same factored form as the actor; V = E_pi[Q]
               "candidate_q"  row-based Q over expanded rows; V = E_pi[Q]      (F3 default)
               "global_state" V(pad-set global state, own context)
The V-type critics regress V(s) on the return and use it as the GAE baseline;
the Q-type critics regress the selected Q on the return and use E_pi[Q] as the
baseline (the baseline then moves with the actor's distribution).

Shared PPO/GAE scaffolding (rollout storage, GAE, optimisers, save/load) comes
from ``algorithm/common/torch_learner_base.py``; AS3's joint-transition learner
extends this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from algorithm.F2.actor import OLBSeparableActor
from algorithm.F2.critic import (
    CandidateQCritic,
    FactorQCritic,
    FactorVCritic,
    GlobalStateAgentCritic,
    global_state_batch,
)
from algorithm.F2.menu import FactorMenu, stack_factor_menus, stack_factor_rows
from algorithm.common.torch_learner_base import BaseTorchMAPPOLearner
from algorithm.common.full_combination_actor import FullCombinationActor


@dataclass(frozen=True)
class MAPPOActionSample:
    """One decision as stored in the rollout."""

    candidate_features: np.ndarray        # expanded rows [N, D] (empty when scored from factors)
    mask: np.ndarray                      # [N]
    context: np.ndarray                   # actor context [C]
    action: int
    log_prob: float
    value: float
    critic_context: np.ndarray | None = None   # [C] or, for global_state, [P, C]
    decoder_history: np.ndarray | None = None
    factors: FactorMenu | None = None     # set when the decision was scored from a FactorMenu


class F2Learner(BaseTorchMAPPOLearner):
    """PPO over ragged candidate menus with a switchable actor and value baseline."""

    supported_actor_types = ("olb", "obl", "flat")
    supported_critic_types = ("factor_v", "factor_q", "candidate_q", "global_state")

    def __init__(
        self,
        *,
        candidate_dim: int,
        context_dim: int,
        max_candidates: int,
        feature_schema: str,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 10.0,
        batch_size: int = 64,
        ppo_epochs: int = 8,
        time_unit_s: float = 60.0,
        reward_scale: float = 25.0,
        device: str = "cuda",
        seed: int = 0,
        actor_type: str = "olb",
        critic_type: str | None = None,
        interaction_rank: int = 8,
        critic_context_dim: int | None = None,
    ):
        self.context_dim = int(context_dim)
        self.actor_type = str(actor_type)
        if self.actor_type not in self.supported_actor_types:
            raise ValueError(f"actor_type must be one of {self.supported_actor_types}, got {self.actor_type!r}")
        # the (o,l)|b actor consumes FactorMenu inputs; the flat actor needs expanded rows
        self.supports_factor_menu = self.actor_type in ("olb", "obl")
        self.critic_type = str(critic_type) if critic_type is not None else (
            "factor_v" if self.actor_type in ("olb", "obl") else "candidate_q"
        )
        if self.critic_type not in self.supported_critic_types:
            raise ValueError(f"critic_type must be one of {self.supported_critic_types}, got {self.critic_type!r}")
        if self.critic_type != "candidate_q" and not self.supports_factor_menu:
            # every critic but candidate_q reads a FactorMenu, and both separable
            # splits produce one -- (o,l) | b and (o,b) | l differ in what the two
            # blocks hold, not in the structure. Only the flat actor is excluded.
            raise ValueError(f"critic_type={self.critic_type!r} scores FactorMenu inputs only, "
                             f"which actor_type={self.actor_type!r} does not produce")
        self.interaction_rank = int(interaction_rank)
        self.critic_context_dim = int(self.context_dim if critic_context_dim is None else critic_context_dim)
        super().__init__(
            candidate_dim=candidate_dim,
            actor_context_dim=self.context_dim,
            max_candidates=max_candidates,
            feature_schema=feature_schema,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_ratio=clip_ratio,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            ppo_epochs=ppo_epochs,
            time_unit_s=time_unit_s,
            reward_scale=reward_scale,
            device=device,
            seed=seed,
        )
        # actor first, then critic: the actor's init is identical across critic types
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        self._initialize_optimizers()

    # ------------------------------------------------------------------ #
    # networks
    # ------------------------------------------------------------------ #
    def _build_actor(self) -> nn.Module:
        if self.actor_type == "olb":
            return OLBSeparableActor(self.candidate_dim, self.actor_context_dim, self.hidden_dim, rank=self.interaction_rank)
        if self.actor_type == "obl":
            # the same separable scorer on the other split: the first block is one
            # row per (order, battery), the second one row per return pad
            from algorithm.F2.menu import L_DIM, OB_DIM
            return OLBSeparableActor(self.candidate_dim, self.actor_context_dim, self.hidden_dim,
                                     rank=self.interaction_rank, a_dim=OB_DIM, b_dim=L_DIM)
        return FullCombinationActor(self.candidate_dim, self.actor_context_dim, self.hidden_dim)

    def _build_critic(self) -> nn.Module:
        if self.critic_type == "factor_v":
            if self.actor_type == "obl":
                from algorithm.F2.menu import L_DIM, OB_DIM
                return FactorVCritic(self.critic_context_dim, self.hidden_dim,
                                     a_dim=OB_DIM, b_dim=L_DIM)
            return FactorVCritic(self.critic_context_dim, self.hidden_dim)
        if self.critic_type == "factor_q":
            return FactorQCritic(self.critic_context_dim, self.hidden_dim, rank=self.interaction_rank)
        if self.critic_type == "global_state":
            return GlobalStateAgentCritic(self.critic_context_dim, self.hidden_dim)
        return CandidateQCritic(self.candidate_dim, self.critic_context_dim, self.hidden_dim)

    @property
    def critic_needs_rows(self) -> bool:
        """The row critic needs the expanded [N, D] rows even for factor menus."""
        return self.critic_type == "candidate_q"

    # ------------------------------------------------------------------ #
    # FactorMenu batching (also used by the joint-transition learner)
    # ------------------------------------------------------------------ #
    def _factor_batch(self, menus: list[FactorMenu]) -> dict[str, torch.Tensor]:
        """Pad a list of FactorMenus into batched factor tensors + flat mask."""
        arrays = stack_factor_menus(menus)
        return {
            key: torch.as_tensor(value, dtype=(torch.bool if key.endswith("mask") else torch.long if key.endswith("_idx") else torch.float32), device=self.device)
            for key, value in arrays.items()
        }

    def _factor_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor.forward_factors(
            batch["factor_ol"], batch["factor_b"], batch["factor_kind"],
            batch["factor_ol_idx"], batch["factor_b_idx"], batch["mask"], batch["context"],
        )

    def _factor_v(self, fb: dict[str, torch.Tensor], critic_context_t: torch.Tensor) -> torch.Tensor:
        return self.critic.value_from_factors(
            fb["factor_ol"], fb["factor_b"], fb["factor_ol_mask"], fb["factor_b_mask"], critic_context_t,
        )

    def _factor_q(self, fb: dict[str, torch.Tensor], critic_context_t: torch.Tensor) -> torch.Tensor:
        return self.critic.q_from_factors(
            fb["factor_ol"], fb["factor_b"], fb["factor_kind"],
            fb["factor_ol_idx"], fb["factor_b_idx"], fb["mask"], critic_context_t,
        )

    @staticmethod
    def _rows_from_factors(menus: list[FactorMenu], n_flat: int, dim: int) -> np.ndarray:
        """Expanded rows for the row critic, rebuilt from the factors (float64 exact)."""
        del dim
        return stack_factor_rows(menus, n_flat)

    # ------------------------------------------------------------------ #
    # critic input plumbing (shape depends on critic_type)
    # ------------------------------------------------------------------ #
    def _check_critic_context(self, critic_context: np.ndarray) -> np.ndarray:
        critic_context = np.asarray(critic_context, dtype=np.float32)
        if self.critic_type == "global_state":
            if critic_context.ndim != 2 or critic_context.shape[1] != self.critic_context_dim:
                raise ValueError(
                    f"global_state critic context must have shape [P, {self.critic_context_dim}], "
                    f"got {critic_context.shape}"
                )
            return critic_context
        if critic_context.shape != (self.critic_context_dim,):
            raise ValueError(
                f"critic context must have shape ({self.critic_context_dim},), got {critic_context.shape}"
            )
        return critic_context

    def _critic_context_tensor(self, critic_context: np.ndarray):
        if self.critic_type == "global_state":
            return global_state_batch([critic_context], self.critic_context_dim, self.device)
        return torch.as_tensor(critic_context[None], dtype=torch.float32, device=self.device)

    def _critic_context_batch(self, rows: list[dict]):
        if self.critic_type == "global_state":
            return global_state_batch([row["critic_context"] for row in rows], self.critic_context_dim, self.device)
        return torch.as_tensor(np.stack([row["critic_context"] for row in rows]), dtype=torch.float32, device=self.device)

    def _factor_value(self, menu, fb, distribution, critic_context_t, actor_context_t) -> torch.Tensor:
        """V(s, prefix) for one factor-menu decision at act time (training only)."""
        if self.critic_type == "factor_v":
            return self._factor_v(fb, critic_context_t)
        if self.critic_type == "factor_q":
            return (distribution.probs * self._factor_q(fb, critic_context_t)).sum(dim=-1)
        if self.critic_type == "global_state":
            pads_t, pad_mask_t = critic_context_t
            return self.critic(pads_t, pad_mask_t, actor_context_t)
        # row critic: rebuild the olb rows from the factors
        rows_t = torch.as_tensor(menu.rows()[None], dtype=torch.float32, device=self.device)
        return (distribution.probs * self.critic.q_values(rows_t, fb["mask"], critic_context_t)).sum(dim=-1)

    def _critic_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.critic_type == "factor_v":
            value = self._factor_v(batch, batch["critic_context"])
            return self.value_coefficient * nn.functional.mse_loss(value, batch["return"])
        if self.critic_type == "factor_q":
            q_values = self._factor_q(batch, batch["critic_context"])
        elif self.critic_type == "global_state":
            pads_t, pad_mask_t = batch["critic_context"]
            value = self.critic(pads_t, pad_mask_t, batch["context"])
            return self.value_coefficient * nn.functional.mse_loss(value, batch["return"])
        else:
            q_values = self.critic.q_values(batch["candidate_features"], batch["mask"], batch["critic_context"])
        q_selected = q_values.gather(1, batch["action"][:, None]).squeeze(1)
        return self.value_coefficient * nn.functional.mse_loss(q_selected, batch["return"])

    # ------------------------------------------------------------------ #
    # acting
    # ------------------------------------------------------------------ #
    def act(
        self,
        candidate_features,
        context: np.ndarray,
        *,
        training: bool,
        critic_context: np.ndarray | None = None,
    ) -> tuple[int, MAPPOActionSample]:
        actor_context = self._as_context(context)
        critic_context = self._check_critic_context(actor_context if critic_context is None else critic_context)
        context_t = torch.as_tensor(actor_context[None], dtype=torch.float32, device=self.device)
        critic_context_t = self._critic_context_tensor(critic_context)
        menu = candidate_features if isinstance(candidate_features, FactorMenu) else None
        if menu is None and self.critic_type != "candidate_q":
            raise TypeError(f"critic_type={self.critic_type!r} scores FactorMenu inputs only; got expanded rows")
        if menu is not None:
            if not self.supports_factor_menu:
                raise TypeError(f"{type(self).__name__} with actor_type={self.actor_type!r} cannot score a FactorMenu")
            n = len(menu)
            if n == 0:
                raise ValueError(self.candidate_empty_error)
            if n > self.max_candidates:
                raise ValueError(self._candidate_overflow_error(n))
            # No expanded rows on the factored path: the menu lives in ``sample.factors``
            # and a row-based critic rebuilds rows from it on demand (see _ragged_batch).
            # The empty [0, D] array keeps the sample's dtype and shape signature, and its
            # length 0 marks "this decision was scored from factors, not from rows".
            features = np.zeros((0, self.candidate_dim), dtype=np.float32)
            mask = np.ones(n, dtype=bool)
            fb = self._factor_batch([menu]); fb["context"] = context_t
            mask_t = fb["mask"]
            with torch.no_grad():
                logits = self._factor_logits(fb)
                distribution = torch.distributions.Categorical(logits=logits)
                # grouped_greedy needs the kind indicators; hand it the kind block as "features"
                action_t = self._select_action(logits, distribution, fb["factor_kind"], mask_t, training=training)
                log_prob_t = distribution.log_prob(action_t)
                if training:
                    value_t = self._factor_value(menu, fb, distribution, critic_context_t, context_t)
                else:
                    value_t = torch.zeros((1,), device=self.device)
        else:
            features = np.asarray(candidate_features, dtype=np.float32).reshape(-1, self.candidate_dim)
            if len(features) == 0:
                raise ValueError(self.candidate_empty_error)
            if len(features) > self.max_candidates:
                raise ValueError(self._candidate_overflow_error(len(features)))
            mask = np.ones(len(features), dtype=bool)
            features_t = torch.as_tensor(features[None], dtype=torch.float32, device=self.device)
            mask_t = torch.ones((1, len(features)), dtype=torch.bool, device=self.device)
            with torch.no_grad():
                logits = self.actor(features_t, mask_t, context_t)
                distribution = torch.distributions.Categorical(logits=logits)
                action_t = self._select_action(logits, distribution, features_t, mask_t, training=training)
                log_prob_t = distribution.log_prob(action_t)
                q_values = self.critic.q_values(features_t, mask_t, critic_context_t)
                value_t = (distribution.probs * q_values).sum(dim=-1)
        action = int(action_t.item())
        return action, MAPPOActionSample(
            candidate_features=features,
            mask=mask,
            context=actor_context,
            action=action,
            log_prob=float(log_prob_t.item()),
            value=float(value_t.item()),
            critic_context=critic_context,
            factors=menu,
        )

    # ------------------------------------------------------------------ #
    # rollout rows and PPO update
    # ------------------------------------------------------------------ #
    def _sample_rollout_fields(self, sample: MAPPOActionSample) -> dict[str, object]:
        return {
            "candidate_features": sample.candidate_features,
            "mask": sample.mask,
            "context": sample.context,
            "critic_context": np.asarray(
                sample.critic_context if sample.critic_context is not None else sample.context, dtype=np.float32,
            ),
            "factors": sample.factors,
        }

    def _rollout_array_fields(self) -> dict[str, tuple[str, torch.dtype]]:
        return {
            "candidate_features": ("candidate_features", torch.float32),
            "mask": ("mask", torch.bool),
            "context": ("context", torch.float32),
        }

    def update(self, epochs: int | None = None) -> dict[str, float]:
        if not self._rollout:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "rollout_size": 0.0}
        rollout_size = len(self._rollout)
        advantages = np.asarray([row["advantage"] for row in self._rollout], dtype=np.float32)
        if len(advantages) > 1 and float(advantages.std()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_losses, critic_losses, entropies = [], [], []
        update_epochs = self.ppo_epochs if epochs is None else max(1, int(epochs))
        for _ in range(update_epochs):
            indices = self._rng.permutation(rollout_size)
            for start in range(0, rollout_size, self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch = self._ragged_batch(batch_indices, advantages)
                logits = self._actor_forward(batch)
                distribution = torch.distributions.Categorical(logits=logits)
                log_prob = distribution.log_prob(batch["action"])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(log_prob - batch["old_log_prob"])
                objective = torch.minimum(
                    ratio * batch["advantage"],
                    torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch["advantage"],
                ).mean()
                actor_loss = -objective - self.entropy_coefficient * entropy
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = self._critic_loss(batch)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
        self._rollout = []
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "rollout_size": float(rollout_size),
        }

    def _ragged_batch(self, indices: np.ndarray, advantages: np.ndarray) -> dict[str, torch.Tensor]:
        rows = [self._rollout[int(index)] for index in indices]
        menus = [row.get("factors") for row in rows]
        if all(m is not None for m in menus):
            # factor-based minibatch: the actor scores the factors; only the row
            # critic needs the olb rows rebuilt from them (padded to this batch)
            batch = self._factor_batch(menus)
            if self.critic_needs_rows:
                n_flat = batch["mask"].shape[1]
                batch["candidate_features"] = torch.as_tensor(
                    self._rows_from_factors(menus, n_flat, self.candidate_dim), dtype=torch.float32, device=self.device,
                )
        else:
            batch = self._row_batch(rows)
        batch["context"] = torch.as_tensor(np.stack([row["context"] for row in rows]), dtype=torch.float32, device=self.device)
        batch["action"] = torch.as_tensor([row["action"] for row in rows], dtype=torch.long, device=self.device)
        batch["old_log_prob"] = torch.as_tensor([row["old_log_prob"] for row in rows], dtype=torch.float32, device=self.device)
        batch["advantage"] = torch.as_tensor(advantages[indices], dtype=torch.float32, device=self.device)
        batch["return"] = torch.as_tensor([row["return"] for row in rows], dtype=torch.float32, device=self.device)
        batch["critic_context"] = self._critic_context_batch(rows)
        return batch

    def _row_batch(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        """Pad expanded-row menus to the longest menu of this minibatch."""
        max_candidates = max(len(row["candidate_features"]) for row in rows)
        features = np.zeros((len(rows), max_candidates, self.candidate_dim), dtype=np.float32)
        mask = np.zeros((len(rows), max_candidates), dtype=bool)
        for row_index, row in enumerate(rows):
            candidate_features = np.asarray(row["candidate_features"], dtype=np.float32).reshape(-1, self.candidate_dim)
            count = len(candidate_features)
            features[row_index, :count] = candidate_features
            mask[row_index, :count] = True
        return {
            "candidate_features": torch.as_tensor(features, dtype=torch.float32, device=self.device),
            "mask": torch.as_tensor(mask, dtype=torch.bool, device=self.device),
        }

    def _actor_forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "factor_ol" in batch:
            return self._factor_logits(batch)
        return self.actor(batch["candidate_features"], batch["mask"], batch["context"])

    def _critic_forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # base-class hook, unused here
        raise NotImplementedError("F2Learner computes its value loss in _critic_loss")

    def _as_context(self, context: np.ndarray) -> np.ndarray:
        return self._as_vector(context, self.context_dim, self.actor_context_label)

    # ------------------------------------------------------------------ #
    # checkpoints
    # ------------------------------------------------------------------ #
    def _checkpoint_critic_type(self) -> str:
        # legacy name for the row critic keeps old checkpoints loadable
        return "candidate_conditioned_q" if self.critic_type == "candidate_q" else self.critic_type

    def _checkpoint_metadata(self) -> dict[str, object]:
        return {
            "candidate_dim": self.candidate_dim,
            "context_dim": self.context_dim,
            "max_candidates": self.max_candidates,
            "feature_schema": self.feature_schema,
            "hidden_dim": self.hidden_dim,
            "critic_type": self._checkpoint_critic_type(),
            "critic_context_dim": self.critic_context_dim,
            "actor_type": type(self.actor).__name__,
            "interaction_rank": self.interaction_rank,
        }

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self._checkpoint_metadata())
        payload.update({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        })
        torch.save(payload, output)

    def load(self, path: str) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        self._validate_checkpoint(payload)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])

    def _validate_checkpoint(self, payload: dict[str, object]) -> None:
        loaded_schema = payload.get("feature_schema")
        if loaded_schema != self.feature_schema:
            raise ValueError(
                f"checkpoint feature schema {loaded_schema!r} is incompatible with "
                f"{self.feature_schema!r}; retrain the policy for the current simulator model"
            )
        if payload.get("critic_type", "candidate_conditioned_q") != self._checkpoint_critic_type():
            raise ValueError("checkpoint critic type is incompatible with current policy")
        if payload.get("actor_type", type(self.actor).__name__) != type(self.actor).__name__:
            raise ValueError("checkpoint actor type is incompatible with current policy")
        if int(payload.get("interaction_rank", self.interaction_rank)) != self.interaction_rank:
            raise ValueError("checkpoint interaction_rank is incompatible with current policy")
