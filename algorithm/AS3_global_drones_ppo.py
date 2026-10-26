"""AS3: <OL+B+LowRank>+GlobalDronesPPO -- global PPO with one action, reward, transition, and ratio per
physical time.

结构（GlobalDronesPPO）：整个机队是一个 agent。每个物理时刻 t，各架空闲无人机
仍按顺序各选一个候选（解码方式和 F2 相同，actor 也是同一个 (o,l)|b 可分解 actor），
但训练时这一时刻的所有决策被打包成 **一个 joint transition**：一个 V(s_t)、一个
advantage、一个 PPO ratio（各决策 log-prob 之和）。

V(s_t) 由 **全局状态 critic**（GlobalStateCritic，与 AS1 MAPPO 共用同一个类和
同一个输入）给出：这一步开始时、任何决策之前的全局状态 = **每个有空闲无人机的站
一行 23 维 context**（时间、全局待派/空闲/飞行/电池比例、该站队列/可用无人机/可用
电池/供需差…），按站集合池化。不含任何候选级 (o,l,b) 信息——候选只属于 actor。

和 F2 的唯一差别就是这个 agent 结构；候选构造、特征布局、actor 完全一致。
本文件两个类：
  * TorchGlobalJointPPOLearner         —— learner 侧：每步一次全局状态 critic、joint batch
  * AS3GlobalDronesPPOOrderBatteryPolicy —— policy 侧：每步先取全局状态算 V，再顺序解码

维度记号：
  N   = 当前这架无人机的菜单长度（扁平条目数）
  N_ol, N_b = 该菜单的 (o,l) 行数 / 电池行数（actor 只对这些行做前向）
  D   = candidate_dim = 29        （olb 行；分解路径下不再物化）
  C   = context_dim   = 23
  P   = 这一步有空闲无人机的站数（全局状态 [P, C] 的行数）
  R   = minibatch 里的决策数；T = 其中的 joint transition 数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithm.F2.menu import FactorMenu
from algorithm.F2.policy import F2Policy
from algorithm.dispatch.presets import GLOBAL_DRONES_PPO
from algorithm.common.torch_learner_base import _ActorStep
from algorithm.F2.critic import GlobalStateCritic, global_state_batch
from algorithm.F2.learner import F2Learner, MAPPOActionSample


@dataclass(frozen=True)
class GlobalJointPPOActionSample:
    """一次决策存进 rollout 的样本：普通 PPO 字段 + 这一步的全局状态（critic 输入）。
    同一时刻的所有决策引用同一份全局状态数组。"""

    candidate_features: np.ndarray      # [N, D] 行菜单（分解菜单时为空 [0, D]）
    candidate_mask: np.ndarray          # [N]
    actor_context: np.ndarray           # [C]
    action: int
    log_prob: float
    value: float                        # V(s_t)，同一时刻的所有决策相同（评估时为 0，不参与）
    factors: FactorMenu | None          # 分解菜单：ol/b 因子 + (kind, ol_idx, b_idx)
    global_state: np.ndarray            # [P, C] 活跃站的 context 集合


class JointTransitionPPOLearner(F2Learner):
    """F2Learner whose PPO transition is a whole physical step: the per-drone
    decisions taken at the same simulator time form one joint transition with one
    advantage, one ratio (product of the conditional probabilities) and one value
    (agent structure GlobalDronesPPO)."""

    def __init__(self, *, update_row_budget: int = 300_000, **kwargs):
        super().__init__(**kwargs)
        self._joint_transitions: list[dict[str, object]] = []
        # bound on the padded menu work per actor call in the PPO update (memory, not maths)
        self.update_row_budget = int(update_row_budget)

    def store_episode(
        self,
        samples: list[object],
        rewards: list[float],
        decision_times: list[int],
        episode_end_time: int,
        *,
        transition_groups: list[list[int]] | None = None,
    ) -> None:
        if not samples:
            return
        if not (len(samples) == len(rewards) == len(decision_times)):
            raise ValueError(self.sample_length_error)
        groups = transition_groups or [[index] for index in range(len(samples))]
        flat_indices = [index for group in groups for index in group]
        if sorted(flat_indices) != list(range(len(samples))):
            raise ValueError(
                "joint transition groups must partition every episode sample exactly once"
            )
        for group in groups:
            if not group:
                raise ValueError("joint transition groups cannot be empty")
            group_times = {int(decision_times[index]) for index in group}
            if len(group_times) != 1:
                raise ValueError(
                    "all conditional actions in a joint transition must share one decision time"
                )

        transition_times = [int(decision_times[group[0]]) for group in groups]
        transition_rewards = [
            sum(float(rewards[index]) for index in group)
            for group in groups
        ]
        values = np.asarray(
            [float(samples[group[0]].value) for group in groups],
            dtype=np.float64,
        )
        scaled_rewards = np.asarray(transition_rewards, dtype=np.float64) / self.reward_scale
        advantages = np.zeros(len(groups), dtype=np.float64)
        gae = 0.0
        for index in range(len(groups) - 1, -1, -1):
            terminal = index == len(groups) - 1
            next_time = episode_end_time if terminal else transition_times[index + 1]
            delta_t = max(0.0, float(next_time - transition_times[index]))
            discount = self.gamma ** (delta_t / self.time_unit_s)
            trace_discount = self.gae_lambda ** (delta_t / self.time_unit_s)
            next_value = 0.0 if terminal else values[index + 1]
            delta = scaled_rewards[index] + discount * next_value - values[index]
            gae = delta + (0.0 if terminal else discount * trace_discount * gae)
            advantages[index] = gae
        returns = advantages + values
        for transition_index, group in enumerate(groups):
            rows: list[dict[str, object]] = []
            for sample_index in group:
                sample = samples[sample_index]
                row = self._sample_rollout_fields(sample)
                row.update(
                    {
                        "action": sample.action,
                        "old_log_prob": sample.log_prob,
                    }
                )
                rows.append(row)
            self._joint_transitions.append(
                {
                    "rows": rows,
                    "advantage": float(advantages[transition_index]),
                    "return": float(returns[transition_index]),
                }
            )

    def update(self, epochs: int | None = None) -> dict[str, float]:
        if not self._joint_transitions:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "rollout_size": 0.0,
            }
        rollout_size = len(self._joint_transitions)
        advantages = np.asarray(
            [transition["advantage"] for transition in self._joint_transitions],
            dtype=np.float32,
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
                transition_indices = indices[start : start + self.batch_size]
                transitions = [
                    self._joint_transitions[int(index)]
                    for index in transition_indices
                ]
                rows = [
                    row
                    for transition in transitions
                    for row in transition["rows"]
                ]
                owner = torch.as_tensor(
                    [t for t, transition in enumerate(transitions) for _ in transition["rows"]],
                    dtype=torch.long, device=self.device,
                )
                counts = torch.bincount(owner, minlength=len(transitions)).clamp_min(1).float()
                old_log_prob = torch.as_tensor(
                    [sum(float(row["old_log_prob"]) for row in transition["rows"]) for transition in transitions],
                    dtype=torch.float32, device=self.device,
                )
                transition_advantage = torch.as_tensor(
                    advantages[transition_indices], dtype=torch.float32, device=self.device,
                )
                # Two-pass joint PPO step with bounded memory: a transition holds every
                # decision of one physical step (dozens of drones, each with a large
                # menu on a busy pad), so the whole minibatch does not fit through the
                # actor at once. Pass 1 (no grad) scores the decisions in chunks and gives
                # the joint log-prob per transition; the clipped-surrogate weight
                # dL/dlogpi is then a per-transition scalar, so pass 2 backpropagates
                # chunk by chunk with exactly that weight (same gradient as one pass).
                with torch.no_grad():
                    joint_log_prob = torch.zeros(len(transitions), device=self.device)
                    for chunk_rows, chunk_idx in self._row_chunks(rows):
                        cb = self._joint_batch(chunk_rows)
                        lp = torch.distributions.Categorical(logits=self._actor_forward(cb)).log_prob(cb["action"])
                        joint_log_prob = joint_log_prob.index_add(0, owner[torch.as_tensor(chunk_idx, device=self.device)], lp)
                ratio = torch.exp(torch.clamp(joint_log_prob - old_log_prob, -20.0, 20.0))
                unclipped = ratio * transition_advantage
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * transition_advantage
                # d(-min(unclipped, clipped))/d(joint): -ratio*adv on the unclipped branch, 0 on the clipped one
                weight = torch.where(unclipped <= clipped, -unclipped, torch.zeros_like(unclipped)) / len(transitions)
                self.actor_optimizer.zero_grad()
                entropy_sum = torch.zeros(len(transitions), device=self.device)
                for chunk_rows, chunk_idx in self._row_chunks(rows):
                    cb = self._joint_batch(chunk_rows)
                    distribution = torch.distributions.Categorical(logits=self._actor_forward(cb))
                    lp = distribution.log_prob(cb["action"]); ent = distribution.entropy()
                    own = owner[torch.as_tensor(chunk_idx, device=self.device)]
                    # entropy is regularised PER DECISION (mean over the transition's
                    # components, then over transitions) so entropy_coefficient means the same across structures
                    ent_w = self.entropy_coefficient / (counts[own] * len(transitions))
                    (weight[own] * lp - ent_w * ent).sum().backward()
                    entropy_sum = entropy_sum.index_add(0, own, ent.detach())
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                entropy = (entropy_sum / counts).mean()
                actor_loss = -torch.minimum(unclipped, clipped).mean() - self.entropy_coefficient * entropy

                first_rows = [transition["rows"][0] for transition in transitions]
                first_positions_t = torch.arange(len(transitions), dtype=torch.long, device=self.device)
                value = self._joint_critic_value(self._joint_batch(first_rows), first_positions_t)
                returns = torch.as_tensor(
                    [float(transition["return"]) for transition in transitions],
                    dtype=torch.float32,
                    device=self.device,
                )
                critic_loss = self.value_coefficient * nn.functional.mse_loss(
                    value,
                    returns,
                )
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu().item()))
                critic_losses.append(float(critic_loss.detach().cpu().item()))
                entropies.append(float(entropy.detach().cpu().item()))
        self._joint_transitions = []
        self._rollout = []
        return {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "entropy": float(np.mean(entropies)),
            "rollout_size": float(rollout_size),
        }

    def _row_chunks(self, rows):
        """Split decisions into chunks whose padded menu work stays bounded:
        sum over the chunk of (menu entries + factor rows) <= update_row_budget.
        Yields (rows, index array)."""
        budget = float(self.update_row_budget)
        chunk, idx, load = [], [], 0.0
        for k, row in enumerate(rows):
            m = row.get("factors")
            cost = float(len(m) + m.ol.shape[0] + m.b.shape[0]) if m is not None else float(len(row["candidate_features"]))
            if chunk and load + cost > budget:
                yield chunk, np.asarray(idx, dtype=np.int64)
                chunk, idx, load = [], [], 0.0
            chunk.append(row); idx.append(k); load += cost
        if chunk:
            yield chunk, np.asarray(idx, dtype=np.int64)

    def _joint_batch(self, rows: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        menus = [row.get("factors") for row in rows]
        if all(m is not None for m in menus):
            batch = self._factor_batch(menus)
            batch["context"] = torch.as_tensor(np.stack([row["context"] for row in rows]), dtype=torch.float32, device=self.device)
            batch["action"] = torch.as_tensor([row["action"] for row in rows], dtype=torch.long, device=self.device)
            return batch
        max_candidates = max(len(row["candidate_features"]) for row in rows)
        features = np.zeros(
            (len(rows), max_candidates, self.candidate_dim),
            dtype=np.float32,
        )
        mask = np.zeros((len(rows), max_candidates), dtype=bool)
        for row_index, row in enumerate(rows):
            candidate_features = np.asarray(
                row["candidate_features"],
                dtype=np.float32,
            ).reshape(-1, self.candidate_dim)
            count = len(candidate_features)
            features[row_index, :count] = candidate_features
            mask[row_index, :count] = True
        return {
            "candidate_features": torch.as_tensor(
                features,
                dtype=torch.float32,
                device=self.device,
            ),
            "mask": torch.as_tensor(mask, dtype=torch.bool, device=self.device),
            "context": torch.as_tensor(
                np.stack([row["context"] for row in rows]),
                dtype=torch.float32,
                device=self.device,
            ),
            "action": torch.as_tensor(
                [row["action"] for row in rows],
                dtype=torch.long,
                device=self.device,
            ),
        }

    def _joint_critic_value(
        self,
        batch: dict[str, torch.Tensor],
        first_positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(
            batch["candidate_features"][first_positions],
            batch["mask"][first_positions],
            batch["context"][first_positions],
        )

    def clear_rollout(self) -> None:
        super().clear_rollout()
        self._joint_transitions = []

    def _checkpoint_metadata(self) -> dict[str, object]:
        metadata = super()._checkpoint_metadata()
        metadata["transition_mode"] = "joint_same_time_action"
        return metadata

    def _validate_checkpoint(self, payload: dict[str, object]) -> None:
        super()._validate_checkpoint(payload)
        if payload.get("transition_mode") != "joint_same_time_action":
            raise ValueError(
                "joint-transition PPO checkpoint does not contain the O4 transition mode"
            )


class TorchGlobalJointPPOLearner(JointTransitionPPOLearner):
    """Joint PPO: (o,l)|b actor per decision, one fleet-critic value per physical step."""

    # the actor (olb | flat) comes from F2Learner; only the value baseline differs:
    # same critic class and input as AS1 MAPPO, the set of active-pad contexts
    supported_critic_types = ("global_state_fleet",)

    def __init__(self, **kwargs):
        kwargs["critic_type"] = "global_state_fleet"
        super().__init__(**kwargs)

    def _build_critic(self):
        return GlobalStateCritic(self.context_dim, self.hidden_dim)

    # ---- global state -> critic ------------------------------------------------
    def evaluate_step(self, global_state: np.ndarray, *, training: bool) -> float:
        """V(s_t) from the pre-commitment global state [P, C] (once per physical step)."""
        if not training or len(global_state) == 0:
            return 0.0
        with torch.no_grad():
            value = self.critic(*global_state_batch([global_state], self.context_dim, self.device))
        return float(value.item())

    # ---- per-decision action --------------------------------------------------
    def act(
        self,
        candidate_features,
        context: np.ndarray,
        global_state: np.ndarray,
        step_value: float,
        *,
        training: bool,
    ) -> tuple[int, GlobalJointPPOActionSample]:
        # actor：分解菜单走 forward_factors（N_ol + N_b 行前向），否则走行菜单。
        menu = candidate_features if isinstance(candidate_features, FactorMenu) else None
        if menu is not None:
            step = self._factor_actor_step(menu, context, training=training)
            features = step.candidate_features            # [0, D]
            mask = step.candidate_mask                    # [N]
        else:
            step = self._actor_step(candidate_features, context, training=training)
            menu_size = int(step.candidate_mask.sum())
            features = step.candidate_features[:menu_size]
            mask = step.candidate_mask[:menu_size]
        return step.action, GlobalJointPPOActionSample(
            candidate_features=features,
            candidate_mask=mask,
            actor_context=step.actor_context,
            action=step.action,
            log_prob=step.log_prob,
            value=float(step_value),
            factors=menu,
            global_state=global_state,
        )

    def _factor_actor_step(self, menu: FactorMenu, context: np.ndarray, *, training: bool) -> _ActorStep:
        n = len(menu)
        if n == 0:
            raise ValueError(self.candidate_empty_error)
        if n > self.max_candidates:
            raise ValueError(self._candidate_overflow_error(n))
        actor_context = self._as_vector(context, self.actor_context_dim, self.actor_context_label)
        context_t = torch.as_tensor(actor_context[None], dtype=torch.float32, device=self.device)
        fb = self._factor_batch([menu]); fb["context"] = context_t
        with torch.no_grad():
            logits = self._factor_logits(fb)                                  # [1, N]
            distribution = torch.distributions.Categorical(logits=logits)
            action_t = self._select_action(logits, distribution, fb["factor_kind"], fb["mask"], training=training)
            log_prob_t = distribution.log_prob(action_t)
        return _ActorStep(
            candidate_features=np.zeros((0, self.candidate_dim), dtype=np.float32),
            candidate_mask=np.ones(n, dtype=bool),
            actor_context=actor_context,
            action=int(action_t.item()),
            log_prob=float(log_prob_t.item()),
            candidate_features_t=None,
            candidate_mask_t=fb["mask"],
            actor_context_t=context_t,
        )

    # ---- rollout / batches ----------------------------------------------------
    def _sample_rollout_fields(self, sample: GlobalJointPPOActionSample):
        return {
            "candidate_features": sample.candidate_features,
            "candidate_mask": sample.candidate_mask,
            "factors": sample.factors,
            "context": sample.actor_context,
            "global_state": sample.global_state,
        }

    def _joint_batch(self, rows):
        # 父类拼 actor 张量（分解菜单：factor_* + mask；行菜单：candidate_features）；
        # 全局状态按 transition 取用（见 _joint_critic_value），这里只保留引用。
        batch = super()._joint_batch(rows)
        batch["global_state"] = [row["global_state"] for row in rows]
        return batch

    def _joint_critic_value(self, batch, first_positions):
        # 每个 joint transition 一个 value：取其第一条决策所引用的全局状态
        # （同一时刻的决策引用同一份）→ pad_contexts [T, Pmax, C] 喂给 critic。
        idx = [int(i) for i in first_positions.tolist()]
        return self.critic(*global_state_batch([batch["global_state"][i] for i in idx], self.context_dim, self.device))

    def _checkpoint_metadata(self):
        metadata = super()._checkpoint_metadata()
        metadata.update({
            "transition_mode": "global_joint_same_time_action",
            "critic_type": "global_state_critic",
            "actor_type": self.actor_type,
            "interaction_rank": self.interaction_rank,
        })
        return metadata

    def _validate_checkpoint(self, payload):
        if payload.get("feature_schema") != self.feature_schema:
            raise ValueError(
                f"checkpoint feature schema {payload.get('feature_schema')!r} is incompatible with "
                f"{self.feature_schema!r}; retrain the policy for the current simulator model"
            )
        if payload.get("transition_mode") != "global_joint_same_time_action":
            raise ValueError("checkpoint is not an AS3_GlobalDronesPPO checkpoint")
        if payload.get("critic_type", "global_state_critic") != "global_state_critic":
            raise ValueError("checkpoint critic type is incompatible with current policy")
        if int(payload.get("interaction_rank", self.interaction_rank)) != self.interaction_rank:
            raise ValueError("checkpoint interaction_rank is incompatible with current policy")


class GlobalDronesPPOStructure(F2Policy):
    """Agent structure GlobalDronesPPO on top of F2: the whole fleet is one agent.
    Decisions are still taken drone by drone, but the episode is ONE trajectory
    and the decisions of one physical step form ONE joint transition (one value,
    one joint log-prob, one GAE step). Used by AS3 (this file) and L4 (MoE pointer)."""

    agent_structure = GLOBAL_DRONES_PPO

    def _training_trajectory_groups(self) -> list[list[Any]]:
        return [self._episode_decisions]

    def _trajectory_reward_events(self, trajectory, reward_events):
        del trajectory
        return reward_events

    @staticmethod
    def _joint_step_groups(decision_times: list[int]) -> list[list[int]]:
        groups_by_time: dict[int, list[int]] = {}
        for index, decision_time in enumerate(decision_times):
            groups_by_time.setdefault(int(decision_time), []).append(index)
        return list(groups_by_time.values())

    def _store_training_trajectory(
        self,
        samples: list[Any],
        rewards: list[float],
        decision_times: list[int],
        terminal_time: int,
    ) -> None:
        self.learner.store_episode(
            samples,
            rewards,
            decision_times,
            terminal_time,
            transition_groups=self._joint_step_groups(decision_times),
        )


class AS3GlobalDronesPPOOrderBatteryPolicy(GlobalDronesPPOStructure):
    """AS3: per-drone candidates decoded in drone order, one joint transition per step,
    global-state critic on the pre-commitment set of active-pad contexts (same as AS1).

    policy 侧几乎全部继承：候选/特征（olb 因子）来自底座，"整条 episode 一条轨迹 +
    按时刻分组成 joint transition"来自 GlobalDronesPPOStructure。本类只做一件事：
    每步解码之前先用 _global_state 取全局状态（每个活跃站一行 context，和 AS1 critic
    的输入完全一样）、算一次 V(s_t)，然后把它交给每个决策的样本。
    """

    dispatch_action_prefix = "global_ppo_dispatch"
    joint_plan_config_key = "order_battery_global_ppo"
    joint_plan_feature_schema = "joint_dispatch_return_plan_global_ppo_olb35_v3"
    learner_class = TorchGlobalJointPPOLearner

    def _learner_kwargs(self, cfg: dict, *, seed: int) -> dict:
        kwargs = super()._learner_kwargs(cfg, seed=seed)
        kwargs["interaction_rank"] = int(cfg.get("interaction_rank", 8))
        return kwargs

    def plan(self, state, used_orders, used_batteries):
        # 每个物理时刻：先冻结机队视图 + 算 V(s_t)，再走底座的无人机顺序解码。
        self._begin_plan(state)
        if self.training:
            # V(s_t) only feeds GAE, so the global state is built for training steps only.
            self._step_state = self.menu.global_state(state)             # [P, C]
            self._step_value = self.learner.evaluate_step(self._step_state, training=True)
        else:
            self._step_state, self._step_value = np.zeros((0, self.context_dim), dtype=np.float32), 0.0
        try:
            return super().plan(state, used_orders, used_batteries)
        finally:
            self._step_state = None; self._step_value = 0.0

    def _select_action(self, state, candidate_features, actor_context):
        return self.learner.act(
            candidate_features,          # FactorMenu（或 [N, D] 行）
            actor_context,               # [C]
            self._step_state,            # [P, C] 活跃站 context 集合
            self._step_value,            # V(s_t)
            training=self.training,
        )
