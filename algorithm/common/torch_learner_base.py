from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class _CandidateActor(nn.Module):
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
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_features: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        candidates = self.candidate_net(candidate_features)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (candidates * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        encoded_context = self.context_net(context)
        shared = torch.cat([pooled, encoded_context], dim=-1).unsqueeze(1)
        shared = shared.expand(-1, candidates.shape[1], -1)
        logits = self.score_head(torch.cat([candidates, shared], dim=-1)).squeeze(-1)
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


@dataclass(frozen=True)
class _ActorStep:
    candidate_features: np.ndarray
    candidate_mask: np.ndarray
    actor_context: np.ndarray
    candidate_features_t: torch.Tensor
    candidate_mask_t: torch.Tensor
    actor_context_t: torch.Tensor
    action: int
    log_prob: float


class BaseTorchMAPPOLearner(ABC):
    """Shared masked PPO/GAE lifecycle with pluggable critic inputs."""

    sample_length_error = "MAPPO samples, rewards, and decision times must have equal lengths"
    candidate_empty_error = "MAPPO requires at least one feasible candidate"
    actor_context_label = "MAPPO context"

    def __init__(
        self,
        *,
        candidate_dim: int,
        actor_context_dim: int,
        max_candidates: int,
        feature_schema: str,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        gae_lambda: float,
        clip_ratio: float,
        value_coefficient: float,
        entropy_coefficient: float,
        max_grad_norm: float,
        batch_size: int,
        ppo_epochs: int,
        time_unit_s: float,
        reward_scale: float,
        device: str,
        seed: int,
    ):
        torch.manual_seed(int(seed))
        self.candidate_dim = int(candidate_dim)
        self.actor_context_dim = int(actor_context_dim)
        self.max_candidates = int(max_candidates)
        self.feature_schema = str(feature_schema)
        self.hidden_dim = int(hidden_dim)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.max_grad_norm = float(max_grad_norm)
        self.batch_size = int(batch_size)
        self.ppo_epochs = int(ppo_epochs)
        self.time_unit_s = max(1e-9, float(time_unit_s))
        self.reward_scale = max(1e-9, float(reward_scale))
        self.device = self._resolve_device(device)
        self._rng = np.random.default_rng(seed)
        self.evaluation_action_mode = "greedy"

        self.actor = _CandidateActor(
            self.candidate_dim,
            self.actor_context_dim,
            self.hidden_dim,
        ).to(self.device)
        self.critic: nn.Module
        self.actor_optimizer: torch.optim.Optimizer
        self.critic_optimizer: torch.optim.Optimizer
        self._rollout: list[dict[str, object]] = []

    def _initialize_optimizers(self) -> None:
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
        )

    def _actor_step(
        self,
        candidate_features: np.ndarray,
        actor_context: np.ndarray,
        *,
        training: bool,
    ) -> _ActorStep:
        features, mask = self._pad_candidates(candidate_features)
        context = self._as_vector(
            actor_context,
            self.actor_context_dim,
            self.actor_context_label,
        )
        features_t = torch.as_tensor(
            features[None],
            dtype=torch.float32,
            device=self.device,
        )
        mask_t = torch.as_tensor(mask[None], dtype=torch.bool, device=self.device)
        context_t = torch.as_tensor(
            context[None],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            logits = self.actor(features_t, mask_t, context_t)
            distribution = torch.distributions.Categorical(logits=logits)
            action_t = self._select_action(
                logits,
                distribution,
                features_t,
                mask_t,
                training=training,
            )
            log_prob_t = distribution.log_prob(action_t)
        return _ActorStep(
            candidate_features=features,
            candidate_mask=mask,
            actor_context=context,
            candidate_features_t=features_t,
            candidate_mask_t=mask_t,
            actor_context_t=context_t,
            action=int(action_t.item()),
            log_prob=float(log_prob_t.item()),
        )

    def set_evaluation_action_mode(self, mode: str) -> None:
        if mode not in {"greedy", "stochastic", "grouped_greedy"}:
            raise ValueError(f"Unknown evaluation action mode: {mode}")
        self.evaluation_action_mode = mode

    def _select_action(
        self,
        logits: torch.Tensor,
        distribution: torch.distributions.Categorical,
        candidate_features: torch.Tensor,
        candidate_mask: torch.Tensor,
        *,
        training: bool,
        allow_grouped: bool = True,
    ) -> torch.Tensor:
        if training or self.evaluation_action_mode == "stochastic":
            return distribution.sample()
        if self.evaluation_action_mode != "grouped_greedy" or not allow_grouped:
            return torch.argmax(logits, dim=-1)
        if candidate_features.shape[-1] < 3:
            raise ValueError(
                "grouped_greedy requires dispatch/defer/stop indicators in "
                "the first three candidate features"
            )
        probabilities = distribution.probs.masked_fill(~candidate_mask, 0.0)
        kind_membership = candidate_features[..., :3] > 0.5
        if not torch.all(kind_membership.sum(dim=-1)[candidate_mask] == 1):
            raise ValueError(
                "grouped_greedy requires exactly one dispatch/defer/stop kind "
                "per candidate"
            )
        kind_mass = torch.einsum("bc,bck->bk", probabilities, kind_membership.float())
        selected_kind = torch.argmax(kind_mass, dim=-1)
        selected_kind_mask = kind_membership.gather(
            2,
            selected_kind[:, None, None].expand(-1, candidate_features.shape[1], 1),
        ).squeeze(-1)
        grouped_logits = logits.masked_fill(
            ~selected_kind_mask,
            torch.finfo(logits.dtype).min,
        )
        return torch.argmax(grouped_logits, dim=-1)

    def store_episode(
        self,
        samples: list[Any],
        rewards: list[float],
        decision_times: list[int],
        episode_end_time: int,
    ) -> None:
        if not samples:
            return
        if not (len(samples) == len(rewards) == len(decision_times)):
            raise ValueError(self.sample_length_error)
        values = np.asarray([sample.value for sample in samples], dtype=np.float64)
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
        for sample, advantage, return_value in zip(samples, advantages, returns):
            row = self._sample_rollout_fields(sample)
            row.update(
                {
                    "action": sample.action,
                    "old_log_prob": sample.log_prob,
                    "advantage": float(advantage),
                    "return": float(return_value),
                }
            )
            self._rollout.append(row)

    def update(self, epochs: int | None = None) -> dict[str, float]:
        if not self._rollout:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "rollout_size": 0.0,
            }
        tensors = self._rollout_tensors()
        rollout_size = len(self._rollout)
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        update_epochs = self.ppo_epochs if epochs is None else max(1, int(epochs))
        for _ in range(update_epochs):
            indices = self._rng.permutation(rollout_size)
            for start in range(0, rollout_size, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch = {key: value[batch_idx] for key, value in tensors.items()}
                logits = self._actor_forward(batch)
                distribution = torch.distributions.Categorical(logits=logits)
                log_prob = distribution.log_prob(batch["action"])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(log_prob - batch["old_log_prob"])
                unclipped = ratio * batch["advantage"]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.clip_ratio,
                    1.0 + self.clip_ratio,
                ) * batch["advantage"]
                actor_loss = (
                    -torch.minimum(unclipped, clipped).mean()
                    - self.entropy_coefficient * entropy
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value = self._critic_forward(batch)
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
                entropies.append(float(entropy.detach().cpu().item()))
        self._rollout = []
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "rollout_size": float(rollout_size),
        }

    def clear_rollout(self) -> None:
        self._rollout = []

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self._checkpoint_metadata())
        payload.update(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            }
        )
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

    def _pad_candidates(self, candidate_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidate_features, dtype=np.float32).reshape(
            -1,
            self.candidate_dim,
        )
        if len(candidates) == 0:
            raise ValueError(self.candidate_empty_error)
        if len(candidates) > self.max_candidates:
            raise ValueError(self._candidate_overflow_error(len(candidates)))
        padded = np.zeros((self.max_candidates, self.candidate_dim), dtype=np.float32)
        mask = np.zeros((self.max_candidates,), dtype=bool)
        padded[: len(candidates)] = candidates
        mask[: len(candidates)] = True
        return padded, mask

    def _rollout_tensors(self) -> dict[str, torch.Tensor]:
        advantages = np.asarray(
            [item["advantage"] for item in self._rollout],
            dtype=np.float32,
        )
        if len(advantages) > 1 and float(advantages.std()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        tensors = {
            output: torch.as_tensor(
                np.stack([item[source] for item in self._rollout]),
                dtype=dtype,
                device=self.device,
            )
            for output, (source, dtype) in self._rollout_array_fields().items()
        }
        tensors.update(
            {
                "action": torch.as_tensor(
                    [item["action"] for item in self._rollout],
                    dtype=torch.long,
                    device=self.device,
                ),
                "old_log_prob": torch.as_tensor(
                    [item["old_log_prob"] for item in self._rollout],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "advantage": torch.as_tensor(
                    advantages,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "return": torch.as_tensor(
                    [item["return"] for item in self._rollout],
                    dtype=torch.float32,
                    device=self.device,
                ),
            }
        )
        return tensors

    def _candidate_overflow_error(self, count: int) -> str:
        return (
            f"MAPPO candidate count {count} exceeds configured maximum "
            f"{self.max_candidates}; increase max_candidates instead of silently "
            "dropping feasible actions"
        )

    @staticmethod
    def _as_vector(value: np.ndarray, dimension: int, label: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (dimension,):
            raise ValueError(f"{label} must have shape {(dimension,)}, got {vector.shape}")
        return vector

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        requested = str(requested).lower()
        if requested == "cuda" and not torch.cuda.is_available():
            print("WARNING: requested CUDA but torch.cuda.is_available() is false; using CPU.")
            return torch.device("cpu")
        return torch.device(requested)

    @abstractmethod
    def _sample_rollout_fields(self, sample: Any) -> dict[str, object]:
        """Serialize critic-specific sample arrays into one rollout row."""

    @abstractmethod
    def _actor_forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute actor logits from a rollout minibatch."""

    @abstractmethod
    def _critic_forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute values from the concrete critic's minibatch inputs."""

    @abstractmethod
    def _rollout_array_fields(self) -> dict[str, tuple[str, torch.dtype]]:
        """Map output tensor names to rollout array fields and dtypes."""

    @abstractmethod
    def _checkpoint_metadata(self) -> dict[str, object]:
        """Return concrete schema metadata without model/optimizer state."""

    @abstractmethod
    def _validate_checkpoint(self, payload: dict[str, object]) -> None:
        """Reject checkpoints built for an incompatible concrete learner."""
