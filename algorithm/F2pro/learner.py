"""PPO on the three-block menu.

Self-contained on purpose: the rollout, the time-aware GAE and the clipped
surrogate are written here rather than inherited, so a change to the proposed
method's learner cannot move this one and vice versa. The decode and the
candidates come from F2 -- both methods must face the same actions -- but nothing
on the scoring or the update path is shared.

The GAE is the same time-aware form the rest of the project uses: a step's
discount is gamma^(dt/60), since decisions are minutes apart and unevenly spaced,
so a fixed per-decision discount would price a quiet minute like a busy one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithm.F2pro.actor import TriSeparableActor
from algorithm.F2pro.critic import TriVCritic
from algorithm.F2pro.menu import TriMenu, stack_tri_menus


@dataclass
class TriSample:
    menu: TriMenu
    context: np.ndarray
    action: int
    log_prob: float
    value: float


class F2ProLearner:
    """One actor, one critic, one rollout of per-decision transitions."""

    def __init__(self, *, context_dim: int, hidden_dim: int = 128, interaction_rank: int = 2,
                 coord_frequencies: int = 0, coord_scale: float = 2.0,
                 actor_lr: float = 3e-4, critic_lr: float = 1e-3, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_ratio: float = 0.2, value_coefficient: float = 0.5,
                 entropy_coefficient: float = 0.01, max_grad_norm: float = 10.0,
                 batch_size: int = 32, ppo_epochs: int = 8, time_unit_s: float = 60.0,
                 reward_scale: float = 25.0, device: str = "cuda", seed: int = 0, **unused):
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.interaction_rank = int(interaction_rank)
        self.coord_frequencies = int(coord_frequencies)
        self.coord_scale = float(coord_scale)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_coefficient = float(value_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.max_grad_norm = float(max_grad_norm)
        self.batch_size = int(batch_size)
        self.ppo_epochs = int(ppo_epochs)
        self.time_unit_s = float(time_unit_s)
        self.reward_scale = float(reward_scale)
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.evaluation_action_mode = "greedy"
        self._rng = np.random.default_rng(seed)
        self._rollout: list[dict[str, Any]] = []

        self.actor = TriSeparableActor(self.context_dim, self.hidden_dim,
                                       rank=self.interaction_rank,
                                       coord_frequencies=self.coord_frequencies,
                                       coord_scale=self.coord_scale,
                                       coord_seed=0).to(self.device)
        self.critic = TriVCritic(self.context_dim, self.hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=float(actor_lr))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=float(critic_lr))

    # ------------------------------------------------------------------ #
    # acting
    # ------------------------------------------------------------------ #
    def set_evaluation_action_mode(self, mode: str) -> None:
        if mode not in {"greedy", "stochastic"}:
            raise ValueError("evaluation action mode must be 'greedy' or 'stochastic'")
        self.evaluation_action_mode = mode

    def _batch_of_one(self, menu: TriMenu, context: np.ndarray) -> dict[str, torch.Tensor]:
        arrays = stack_tri_menus([menu])
        out = {}
        for key, value in arrays.items():
            dtype = (torch.bool if key.endswith("mask")
                     else torch.long if key.endswith("_idx") else torch.float32)
            out[key] = torch.as_tensor(value, dtype=dtype, device=self.device)
        out["context"] = torch.as_tensor(np.asarray(context, dtype=np.float32)[None],
                                         dtype=torch.float32, device=self.device)
        return out

    def _logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor(batch["o"], batch["l"], batch["b"], batch["kind"],
                          batch["o_idx"], batch["l_idx"], batch["b_idx"],
                          batch["mask"], batch["context"])

    def _values(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.critic(batch["o"], batch["l"], batch["b"],
                           batch["o_mask"], batch["l_mask"], batch["b_mask"], batch["context"])

    def act(self, menu: TriMenu, context: np.ndarray, *, training: bool) -> tuple[int, TriSample]:
        if len(menu) == 0:
            raise ValueError("F2pro requires a non-empty menu")
        batch = self._batch_of_one(menu, context)
        with torch.no_grad():
            logits = self._logits(batch)
            distribution = torch.distributions.Categorical(logits=logits)
            action = (distribution.sample() if training or self.evaluation_action_mode == "stochastic"
                      else torch.argmax(logits, dim=-1))
            log_prob = distribution.log_prob(action)
            value = self._values(batch) if training else torch.zeros(1, device=self.device)
        index = int(action.item())
        return index, TriSample(menu=menu, context=np.asarray(context, dtype=np.float32),
                                action=index, log_prob=float(log_prob.item()),
                                value=float(value.item()))

    # ------------------------------------------------------------------ #
    # rollout
    # ------------------------------------------------------------------ #
    def store_episode(self, samples: list[TriSample], rewards: list[float],
                      decision_times: list[int], episode_end_time: int) -> None:
        if not samples:
            return
        if not (len(samples) == len(rewards) == len(decision_times)):
            raise ValueError("F2pro trajectory fields must align")
        values = np.asarray([s.value for s in samples], dtype=np.float64)
        scaled = np.asarray(rewards, dtype=np.float64) / self.reward_scale
        advantages = np.zeros(len(samples), dtype=np.float64)
        gae = 0.0
        for index in range(len(samples) - 1, -1, -1):
            terminal = index == len(samples) - 1
            next_time = episode_end_time if terminal else decision_times[index + 1]
            delta_t = max(0.0, float(next_time - decision_times[index]))
            discount = self.gamma ** (delta_t / self.time_unit_s)
            trace = self.gae_lambda ** (delta_t / self.time_unit_s)
            next_value = 0.0 if terminal else values[index + 1]
            delta = scaled[index] + discount * next_value - values[index]
            gae = delta + (0.0 if terminal else discount * trace * gae)
            advantages[index] = gae
        returns = advantages + values
        for sample, advantage, ret in zip(samples, advantages, returns):
            self._rollout.append({"sample": sample, "advantage": float(advantage),
                                  "return": float(ret)})

    def clear_rollout(self) -> None:
        self._rollout = []

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #
    def _minibatch(self, rows: list[dict], advantages: np.ndarray) -> dict[str, torch.Tensor]:
        samples = [row["sample"] for row in rows]
        arrays = stack_tri_menus([s.menu for s in samples])
        batch = {}
        for key, value in arrays.items():
            dtype = (torch.bool if key.endswith("mask")
                     else torch.long if key.endswith("_idx") else torch.float32)
            batch[key] = torch.as_tensor(value, dtype=dtype, device=self.device)
        tensor = lambda value, dtype: torch.as_tensor(value, dtype=dtype, device=self.device)
        batch["context"] = tensor(np.stack([s.context for s in samples]), torch.float32)
        batch["action"] = tensor([s.action for s in samples], torch.long)
        batch["old_log_prob"] = tensor([s.log_prob for s in samples], torch.float32)
        batch["advantage"] = tensor(advantages, torch.float32)
        batch["return"] = tensor([row["return"] for row in rows], torch.float32)
        return batch

    def update(self, epochs: int | None = None) -> dict[str, float]:
        if not self._rollout:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "rollout_size": 0.0}
        size = len(self._rollout)
        advantages = np.asarray([row["advantage"] for row in self._rollout], dtype=np.float32)
        if size > 1 and float(advantages.std()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.ppo_epochs if epochs is None else max(1, int(epochs))):
            order = self._rng.permutation(size)
            for start in range(0, size, self.batch_size):
                picked = order[start : start + self.batch_size]
                rows = [self._rollout[int(i)] for i in picked]
                batch = self._minibatch(rows, advantages[picked])

                logits = self._logits(batch)
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

                value_loss = self.value_coefficient * nn.functional.mse_loss(
                    self._values(batch), batch["return"])
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
        self._rollout = []
        return {"actor_loss": float(np.mean(actor_losses)),
                "critic_loss": float(np.mean(critic_losses)),
                "entropy": float(np.mean(entropies)),
                "rollout_size": float(size)}

    # ------------------------------------------------------------------ #
    # checkpoints
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        # `path` already carries the .order_battery.pt suffix: the dispatcher's
        # _order_battery_model_path adds it, and F2's learner writes to it as
        # given. Appending it again here produced names ending in
        # ".order_battery.pt.order_battery.pt", which no loader looks for.
        # The parent may not exist yet -- the model root is named after the
        # algorithm, and this is a new one.
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                    "context_dim": self.context_dim, "hidden_dim": self.hidden_dim,
                    "interaction_rank": self.interaction_rank,
                    "coord_frequencies": self.coord_frequencies,
                    "feature_schema": "f2pro_o_l_b_v1"}, output)

    def load(self, path: str) -> None:
        payload = torch.load(Path(path), map_location=self.device)
        if payload.get("feature_schema") != "f2pro_o_l_b_v1":
            raise ValueError("F2pro checkpoint has an incompatible feature schema")
        if int(payload.get("interaction_rank", self.interaction_rank)) != self.interaction_rank:
            raise ValueError("checkpoint interaction_rank is incompatible with current policy")
        if int(payload.get("coord_frequencies", self.coord_frequencies)) != self.coord_frequencies:
            raise ValueError("checkpoint coord_frequencies is incompatible with current policy")
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
