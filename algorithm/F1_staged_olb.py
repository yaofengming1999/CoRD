"""F1: AutoregressiveOLB+SerializedMARL -- staged order -> return-launchpad -> battery PPO with a cheap stage-1.

The flat F3 actor (``F3_full_combination.py``) scores the full
``(order, battery, return_launchpad)`` product every decision. On bursty
real-world instances (meituan) that menu can exceed 2,000 candidates for a
single drone's turn. This policy never forms that product. Each per-drone
decision runs three small sequential stages instead:

  1. order        - cheap existence-only feasibility per pending order,
                     using the simulator's short-circuiting
                     ``order_battery_candidate_feasible`` against the
                     highest-charge local battery. No route/return
                     enumeration happens here, so this stage costs
                     O(|orders|) cheap boolean checks, not a full menu build.
  2. launchpad     - full return-target enumeration, but only for the ONE
     (return)        order selected in stage 1, using the same
                     highest-charge battery as a representative (charge is
                     monotonic, so it gives an honest superset of feasible
                     targets).
  3. battery       - order and return-launchpad are now both fixed, so the
                     total energy requirement is a single number; battery
                     choice is a cheap re-check over the drone's local
                     batteries (usually 3-5) confirming that specific
                     (order, battery, return_launchpad) triple.

Worst observed case on meituan test instances: flat menu 2,570 candidates
-> staged ~38 (stage 1) + ~42 (stage 2) + ~5 (stage 3) =~ 85 NN-scored
candidates, all three stages combined.

Order-stage and return-stage features intentionally do NOT mean-pool over
sub-candidates (a flaw identified in the deleted o7_1 autoregressive
design, which averaged a whole (battery, return) sub-menu into one vector
per order and likely diluted "this order has one great option" down to
noise). Each stage here only ever sees features for the option it is
actually being asked to score.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from algorithm.common.torch_learner_base import BaseTorchMAPPOLearner, _CandidateActor
from algorithm.common.geometry_utils import distance_m, safe_ratio
from algorithm.F2.menu import TARGET_KEEP
from algorithm.F2.policy import _JointPlanCandidate, _JointPlanDecision
from algorithm.F2 import featmask
from algorithm.F2.policy import F2Policy
from simulator.model.enums import BatteryStatus, OrderStatus
from simulator.model.types import StateDict, TaskLevelActionDict

# Shared, zero-padded candidate width across all three stages (max of the
# three stages' natural dims: order=10, return=18, battery=8).
CANDIDATE_DIM = 18
# Stage-1 (order) context: reuses SequentialJointPlanPPOOrderBatteryPolicy's
# `_joint_context` unchanged. Stage 2/3 contexts grow by one padded
# candidate width per stage, same scheme the deleted o7_1 design used.
# the unmasked joint-context width; what the learner is built at is the
# feature mask's context_dim, which equals this when nothing is dropped
CONTEXT_DIM = 23


@dataclass(frozen=True)
class StagedPending:
    order_features: np.ndarray
    order_context: np.ndarray
    order_action: int
    order_log_prob: float
    value: float
    selected_order_features: np.ndarray
    return_features: np.ndarray
    return_action: int
    return_log_prob: float
    selected_return_features: np.ndarray


@dataclass(frozen=True)
class StagedSample:
    order_features: np.ndarray
    return_features: np.ndarray
    battery_features: np.ndarray
    order_context: np.ndarray
    return_context: np.ndarray
    battery_context: np.ndarray
    action: int
    return_action: int
    battery_action: int
    has_return: bool
    has_battery: bool
    log_prob: float
    value: float


class _StagedActor(nn.Module):
    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.order_actor = _CandidateActor(candidate_dim, context_dim, hidden_dim)
        self.return_actor = _CandidateActor(
            candidate_dim, context_dim + candidate_dim, hidden_dim
        )
        self.battery_actor = _CandidateActor(
            candidate_dim, context_dim + 2 * candidate_dim, hidden_dim
        )


class _StagedCritic(nn.Module):
    """Single state-value head, evaluated once at the order stage."""

    def __init__(self, candidate_dim: int, context_dim: int, hidden_dim: int):
        super().__init__()
        self.candidate_net = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.context_net = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU())
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, mask, context):
        encoded = self.candidate_net(features)
        mask_f = mask.unsqueeze(-1).float()
        mean = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return self.value_net(
            torch.cat([mean, maximum, self.context_net(context)], dim=-1)
        ).squeeze(-1)


class TorchStagedOrderLaunchpadBatteryPPOLearner(BaseTorchMAPPOLearner):
    """Sequential order -> return-launchpad -> battery PPO.

    Same factorized-joint-log-prob PPO/GAE math as the deleted o7_1
    autoregressive learner (that part was never the problem); what differs
    is entirely upstream, in how the policy builds each stage's candidate
    menu (see module docstring).
    """

    candidate_empty_error = "staged order/launchpad/battery PPO requires a non-empty stage"
    actor_context_label = "staged order/launchpad/battery PPO context"

    scored_rows = 0        # rows scored across the three stages; read by the policy
    # F1b overrides this with three same-width heads; the construction order is
    # unchanged, so F1's own parameter initialisation is bit-identical.
    actor_class = _StagedActor

    def __init__(self, *, candidate_dim: int, context_dim: int, **kwargs):
        self.context_dim = int(context_dim)
        super().__init__(
            candidate_dim=candidate_dim,
            actor_context_dim=self.context_dim,
            **kwargs,
        )
        self.actor = self.actor_class(
            self.candidate_dim, self.context_dim, self.hidden_dim
        ).to(self.device)
        self.critic = _StagedCritic(
            self.candidate_dim, self.context_dim, self.hidden_dim
        ).to(self.device)
        self._initialize_optimizers()

    def begin_order(self, order_features, context, *, training):
        orders = self._stage(order_features, "order")
        context = self._context(context)
        logits, dist, action = self._sample(
            self.actor.order_actor, orders, context, training
        )
        with torch.no_grad():
            value = self.critic(
                self._batch(orders),
                self._mask(np.ones((1, len(orders)), dtype=bool)),
                self._batch(context),
            )
        index = int(action.item())
        return index, StagedPending(
            orders,
            context,
            index,
            float(dist.log_prob(action).item()),
            float(value.item()),
            orders[index].copy(),
            np.empty((0, self.candidate_dim), dtype=np.float32),
            0,
            0.0,
            np.empty((self.candidate_dim,), dtype=np.float32),
        )

    def complete_return(self, pending, return_features, *, training):
        returns = self._stage(return_features, "return launchpad")
        context = np.concatenate(
            [pending.order_context, pending.selected_order_features]
        ).astype(np.float32)
        _, dist, action = self._sample(
            self.actor.return_actor, returns, context, training
        )
        index = int(action.item())
        return index, StagedPending(
            pending.order_features,
            pending.order_context,
            pending.order_action,
            pending.order_log_prob,
            pending.value,
            pending.selected_order_features,
            returns,
            index,
            float(dist.log_prob(action).item()),
            returns[index].copy(),
        )

    def complete_battery(self, pending, battery_features, *, training):
        batteries = self._stage(battery_features, "battery")
        context = np.concatenate(
            [pending.order_context, pending.selected_order_features, pending.selected_return_features]
        ).astype(np.float32)
        _, dist, action = self._sample(
            self.actor.battery_actor, batteries, context, training
        )
        index = int(action.item())
        return index, StagedSample(
            pending.order_features,
            pending.return_features,
            batteries,
            pending.order_context,
            np.concatenate([pending.order_context, pending.selected_order_features]).astype(np.float32),
            context,
            pending.order_action,
            pending.return_action,
            index,
            True,
            True,
            pending.order_log_prob + pending.return_log_prob + float(dist.log_prob(action).item()),
            pending.value,
        )

    def complete_without_dispatch(self, pending):
        empty = np.empty((0, self.candidate_dim), dtype=np.float32)
        return StagedSample(
            pending.order_features,
            empty,
            empty,
            pending.order_context,
            np.zeros((self.context_dim + self.candidate_dim,), dtype=np.float32),
            np.zeros((self.context_dim + 2 * self.candidate_dim,), dtype=np.float32),
            pending.order_action,
            0,
            0,
            False,
            False,
            pending.order_log_prob,
            pending.value,
        )

    def update(self, epochs=None):
        if not self._rollout:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "rollout_size": 0.0}
        advantages = np.asarray([row["advantage"] for row in self._rollout], dtype=np.float32)
        if len(advantages) > 1 and float(advantages.std()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        n = len(self._rollout)
        actor_losses, critic_losses, entropies = [], [], []
        for _ in range(self.ppo_epochs if epochs is None else max(1, int(epochs))):
            indices = self._rng.permutation(n)
            for start in range(0, n, self.batch_size):
                ids = indices[start : start + self.batch_size]
                batch = self._batch_rows([self._rollout[int(i)] for i in ids], advantages[ids])
                order = torch.distributions.Categorical(logits=self.actor.order_actor(batch["order_features"], batch["order_mask"], batch["order_context"]))
                returns = torch.distributions.Categorical(logits=self.actor.return_actor(batch["return_features"], batch["return_mask"], batch["return_context"]))
                batteries = torch.distributions.Categorical(logits=self.actor.battery_actor(batch["battery_features"], batch["battery_mask"], batch["battery_context"]))
                log_prob = order.log_prob(batch["action"])
                log_prob = log_prob + batch["has_return"] * returns.log_prob(batch["return_action"])
                log_prob = log_prob + batch["has_battery"] * batteries.log_prob(batch["battery_action"])
                entropy = (order.entropy() + batch["has_return"] * returns.entropy() + batch["has_battery"] * batteries.entropy()).mean()
                ratio = torch.exp(torch.clamp(log_prob - batch["old_log_prob"], -20.0, 20.0))
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch["advantage"]
                actor_loss = -torch.minimum(ratio * batch["advantage"], clipped).mean() - self.entropy_coefficient * entropy
                self.actor_optimizer.zero_grad(); actor_loss.backward(); nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm); self.actor_optimizer.step()
                value = self.critic(batch["order_features"], batch["order_mask"], batch["order_context"])
                critic_loss = self.value_coefficient * nn.functional.mse_loss(value, batch["return"])
                self.critic_optimizer.zero_grad(); critic_loss.backward(); nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm); self.critic_optimizer.step()
                actor_losses.append(float(actor_loss.detach().cpu())); critic_losses.append(float(critic_loss.detach().cpu())); entropies.append(float(entropy.detach().cpu()))
        self._rollout = []
        return {"actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses)), "entropy": float(np.mean(entropies)), "rollout_size": float(n)}

    def _sample_rollout_fields(self, sample):
        return {
            "order_features": sample.order_features,
            "return_features": sample.return_features,
            "battery_features": sample.battery_features,
            "order_context": sample.order_context,
            "return_context": sample.return_context,
            "battery_context": sample.battery_context,
            "return_action": sample.return_action,
            "battery_action": sample.battery_action,
            "has_return": float(sample.has_return),
            "has_battery": float(sample.has_battery),
        }

    def _actor_forward(self, batch):
        return self.actor.order_actor(batch["order_features"], batch["order_mask"], batch["order_context"])

    def _critic_forward(self, batch):
        return self.critic(batch["order_features"], batch["order_mask"], batch["order_context"])

    def _rollout_array_fields(self):
        return {}

    def _checkpoint_metadata(self):
        return {
            "feature_schema": self.feature_schema,
            "candidate_dim": self.candidate_dim,
            "context_dim": self.context_dim,
            "transition_mode": "staged_order_launchpad_battery",
        }

    def _validate_checkpoint(self, payload):
        if payload.get("feature_schema") != self.feature_schema:
            raise ValueError("staged order/launchpad/battery feature schema is incompatible")

    def _batch_rows(self, rows, advantages):
        def pad(name):
            maximum = max(1, max(len(row[name]) for row in rows))
            values = np.zeros((len(rows), maximum, self.candidate_dim), dtype=np.float32)
            mask = np.zeros((len(rows), maximum), dtype=bool)
            for i, row in enumerate(rows):
                current = np.asarray(row[name], dtype=np.float32).reshape(-1, self.candidate_dim)
                if len(current):
                    values[i, :len(current)] = current
                    mask[i, :len(current)] = True
                else:
                    mask[i, 0] = True
            return self._features(values), self._mask(mask)
        order_features, order_mask = pad("order_features")
        return_features, return_mask = pad("return_features")
        battery_features, battery_mask = pad("battery_features")
        contexts = {
            name: self._features(np.nan_to_num(np.stack([row[name] for row in rows]).astype(np.float32)))
            for name in ("order_context", "return_context", "battery_context")
        }
        return {
            "order_features": order_features, "order_mask": order_mask,
            "return_features": return_features, "return_mask": return_mask,
            "battery_features": battery_features, "battery_mask": battery_mask,
            **contexts,
            "action": self._long([row["action"] for row in rows]),
            "return_action": self._long([row["return_action"] for row in rows]),
            "battery_action": self._long([row["battery_action"] for row in rows]),
            "has_return": self._tensor([row["has_return"] for row in rows]),
            "has_battery": self._tensor([row["has_battery"] for row in rows]),
            "old_log_prob": self._tensor([row["old_log_prob"] for row in rows]),
            "advantage": self._tensor(advantages),
            "return": self._tensor([row["return"] for row in rows]),
        }

    def _stage(self, value, label):
        values = np.nan_to_num(np.asarray(value, dtype=np.float32).reshape(-1, self.candidate_dim), nan=0.0, posinf=1e6, neginf=-1e6)
        if len(values) == 0:
            raise ValueError(f"missing {label} candidates")
        if len(values) > self.max_candidates:
            raise ValueError(f"{label} candidate count {len(values)} exceeds max_candidates={self.max_candidates}")
        return values

    def _context(self, value):
        return self._as_vector(value, self.context_dim, self.actor_context_label)

    def _features(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _mask(self, value):
        return torch.as_tensor(value, dtype=torch.bool, device=self.device)

    def _batch(self, value):
        value = np.asarray(value, dtype=np.float32)
        return self._features(value[None] if value.ndim in (1, 2) else value)

    def _long(self, value):
        return torch.as_tensor(value, dtype=torch.long, device=self.device)

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _sample(self, actor, features, context, training):
        # every row of every stage the actor scores, so a staged decode's cost is
        # comparable with a method that scores one menu once
        self.scored_rows += len(features)
        features_t = self._batch(features)
        mask_t = self._mask(np.ones((1, len(features)), dtype=bool))
        context_t = self._features(np.asarray(context, dtype=np.float32)[None])
        with torch.no_grad():
            logits = actor(features_t, mask_t, context_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample() if training or self.evaluation_action_mode == "stochastic" else torch.argmax(logits, dim=-1)
        return logits, dist, action


class F1StagedOLBOrderBatteryPolicy(F2Policy):
    """F1: cheap order feasibility -> return-launchpad -> battery mask.

    Reuses the single-drone sequential branch (STOP/DEFER active-order lock,
    config plumbing, per-drone GAE trajectory grouping) but overrides candidate
    generation and ``plan`` entirely: no flat (order, battery, return)
    triplet is ever built.
    """

    @property
    def scored_rows(self) -> int:
        """The three stages' rows, counted on the learner: this decode never
        builds a single menu, so ``menu.feature_rows`` stays at zero."""
        return int(getattr(self.learner, "scored_rows", 0))

    def clear_episode(self) -> None:
        # the menu's own counter is reset by the base class; this decode's rows
        # live on the learner and would otherwise accumulate across episodes
        self.learner.scored_rows = 0
        super().clear_episode()

    dispatch_action_prefix = "f1_staged_olb_dispatch"
    joint_plan_config_key = "order_battery_f1_staged_olb_ppo"
    joint_plan_feature_schema = "staged_order_launchpad_battery_v1"
    learner_class = TorchStagedOrderLaunchpadBatteryPPOLearner
    candidate_dim = CANDIDATE_DIM

    def __init__(self, config: dict, *, seed: int = 0):
        decision_cfg = config.setdefault("drone_decision", {})
        stage_cfg = decision_cfg.setdefault(self.joint_plan_config_key, {})
        # Stages stay small (tens, occasionally low hundreds) even on
        # bursty instances, so this cap is a generous safety margin, not a
        # working-set size like the flat design's 65536.
        stage_cfg.setdefault("max_candidates_per_launchpad", 1024)
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
            "batch_size": int(cfg.get("batch_size", 32)),
            "ppo_epochs": self.gradient_steps_per_episode,
            "time_unit_s": float(cfg.get("time_unit_s", 60.0)),
            "reward_scale": float(cfg.get("reward_scale", 25.0)),
            "device": str(cfg.get("device", "cuda")),
            "seed": int(cfg.get("seed", seed)),
        }

    # ------------------------------------------------------------------
    # Per-drone sequential plan loop: order -> return-launchpad -> battery.
    # ------------------------------------------------------------------

    def plan(
        self,
        state: StateDict,
        used_orders: set[str],
        used_batteries: set[str],
    ) -> list[TaskLevelActionDict]:
        current_time = int(state.get("time", 0))
        actions: list[TaskLevelActionDict] = []
        planned_targets: dict[str, int] = {}
        planned_departures: dict[str, int] = {}
        blocked_launchpads: set[str] = set()
        cleared_launchpads: set[str] = set()
        fleet_size = len(state.get("drones", {}))

        for agent_index, (raw_drone_id, drone) in enumerate(state.get("drones", {}).items()):
            drone_id = str(raw_drone_id)
            launchpad_id = drone.get("current_launchpad_id")
            if launchpad_id is None:
                continue
            launchpad_id = str(launchpad_id)
            if launchpad_id in blocked_launchpads:
                continue
            launchpad = state.get("launchpads", {}).get(launchpad_id)
            if launchpad is None or not self.menu.drone_is_available(state, launchpad, drone_id):
                continue
            if launchpad_id not in cleared_launchpads:
                self.menu.clear_terminal_lock(state, launchpad_id, launchpad)
                cleared_launchpads.add(launchpad_id)

            order_candidates = self._stage1_order_candidates(
                state, launchpad, drone_id, used_orders, used_batteries,
                current_time, planned_departures,
            )
            if not order_candidates:
                blocked_launchpads.add(launchpad_id)
                continue

            # One decision group per drone that reaches the actor, which is where
            # F2Policy records one too (its guards above this point are the same
            # sequence, and it also `continue`s on an empty menu before
            # recording). Without this the group counter stays at zero for F1 and
            # table 6's ms/decision divides by max(1, 0) -- a 64 s "decision".
            # NOTE order_ids here is the stage-1 ORDER menu, not the triple
            # product F1 never forms, so this policy's candidates/group is not
            # the other three's triples/decision and must not be read as it.
            # battery_ids is only known if stage 3 is reached, so the groups that
            # stop earlier contribute none and unique_batteries/group is a lower
            # bound for F1. Neither quantity feeds the ms columns, which need
            # only the group count.
            group_order_ids = [
                c.order_id for c in order_candidates
                if c.kind == "dispatch" and c.order_id
            ]

            def record_group(kind: str, battery_ids: list[str]) -> None:
                self._metrics.record_group(
                    order_ids=group_order_ids,
                    battery_ids=battery_ids,
                    selected_count=int(kind == "dispatch"),
                    available_drones=1,
                    selected_kind=kind,
                    selected_order_has_feasible_battery=None,
                )

            context = self.menu.context(
                state, launchpad, used_orders, used_batteries, 1,
                planned_targets, planned_departures,
                agent_index=agent_index, agent_count=fleet_size,
            )
            order_index, pending = self.learner.begin_order(
                np.stack([c.features for c in order_candidates]), context,
                training=self.training,
            )
            selected_order = order_candidates[order_index]
            if selected_order.kind in {"stop", "defer"}:
                sample = self.learner.complete_without_dispatch(pending)
                if self.training:
                    self._episode_decisions.append(_JointPlanDecision(
                        current_time, sample, selected_order.kind,
                        launchpad_id, drone_id, selected_order.order_id,
                    ))
                if selected_order.kind == "defer" and selected_order.order_id is not None:
                    self.menu.active_order_by_launchpad[launchpad_id] = selected_order.order_id
                record_group(selected_order.kind, [])
                blocked_launchpads.add(launchpad_id)
                continue
            order_id = selected_order.order_id
            if order_id is None:
                raise RuntimeError(self.dispatch_candidate_error)

            return_candidates = self._stage2_return_candidates(
                state, launchpad, order_id, used_batteries,
                current_time, planned_targets, planned_departures,
            )
            if not return_candidates:
                # Stage-1's feasibility check used the same representative
                # battery, so this should be rare (e.g. that battery was
                # claimed by another drone earlier in this same epoch).
                # Fail safe: defer this order rather than crash the episode.
                sample = self.learner.complete_without_dispatch(pending)
                if self.training:
                    self._episode_decisions.append(_JointPlanDecision(
                        current_time, sample, "defer", launchpad_id, drone_id, order_id,
                    ))
                self.menu.active_order_by_launchpad[launchpad_id] = order_id
                record_group("defer", [])
                blocked_launchpads.add(launchpad_id)
                continue
            return_index, pending = self.learner.complete_return(
                pending, np.stack([c.features for c in return_candidates]),
                training=self.training,
            )
            selected_return = return_candidates[return_index]
            return_launchpad_id = selected_return.return_launchpad_id
            if return_launchpad_id is None:
                raise RuntimeError(self.dispatch_candidate_error)

            battery_candidates = self._stage3_battery_candidates(
                state, launchpad, order_id, return_launchpad_id, used_batteries, current_time,
            )
            if not battery_candidates:
                # The representative battery reached this target, but the
                # specific battery masking below found nothing (e.g. only
                # the representative battery itself qualifies and it was
                # consumed elsewhere this epoch). Defer rather than crash.
                sample = self.learner.complete_without_dispatch(pending)
                if self.training:
                    self._episode_decisions.append(_JointPlanDecision(
                        current_time, sample, "defer", launchpad_id, drone_id, order_id,
                    ))
                self.menu.active_order_by_launchpad[launchpad_id] = order_id
                record_group("defer", [])
                blocked_launchpads.add(launchpad_id)
                continue
            battery_index, sample = self.learner.complete_battery(
                pending, np.stack([c.features for c in battery_candidates]),
                training=self.training,
            )
            selected_battery = battery_candidates[battery_index]
            battery_id = selected_battery.battery_id
            if battery_id is None:
                raise RuntimeError(self.dispatch_candidate_error)

            if self.training:
                self._episode_decisions.append(_JointPlanDecision(
                    current_time, sample, "dispatch", launchpad_id, drone_id, order_id,
                ))
            record_group("dispatch", [
                c.battery_id for c in battery_candidates if c.battery_id
            ])
            actions.append({
                "action_id": (
                    f"{self.dispatch_action_prefix}_{current_time}_{drone_id}_"
                    f"{order_id}_{battery_id}_{return_launchpad_id}"
                ),
                "action_type": "order_battery_dispatch",
                "order_id": order_id,
                "battery_id": battery_id,
                "target_launchpad_id": return_launchpad_id,
                "drone_id": drone_id,
                "feasibility": True,
            })
            used_orders.add(order_id)
            used_batteries.add(battery_id)
            planned_departures[launchpad_id] = planned_departures.get(launchpad_id, 0) + 1
            planned_targets[return_launchpad_id] = planned_targets.get(return_launchpad_id, 0) + 1
            if self.menu.active_order_by_launchpad.get(launchpad_id) == order_id:
                self.menu.active_order_by_launchpad.pop(launchpad_id, None)
        return actions

    # ------------------------------------------------------------------
    # Stage 1: order. Cheap existence check only, no route enumeration.
    # ------------------------------------------------------------------

    def _local_battery_ids(self, launchpad: dict, batteries: dict, used_batteries: set[str]) -> list[str]:
        return [
            str(bid) for bid in (
                list(launchpad.get("charged_battery_ids", []))
                + list(launchpad.get("charging_battery_ids", []))
            )
            if str(bid) not in used_batteries and str(bid) in batteries
        ]

    @staticmethod
    def _best_local_battery_id(local_battery_ids: list[str], batteries: dict) -> str | None:
        if not local_battery_ids:
            return None
        return max(
            local_battery_ids,
            key=lambda bid: float(batteries[bid].get("current_charge_Wh", 0.0) or 0.0),
        )

    def _stage1_order_candidates(
        self,
        state: StateDict,
        launchpad: dict,
        drone_id: str,
        used_orders: set[str],
        used_batteries: set[str],
        current_time: int,
        planned_departures: dict[str, int],
    ) -> list[_JointPlanCandidate]:
        orders = state.get("orders", {})
        batteries = state.get("batteries", {})
        launchpad_id = str(launchpad.get("launchpad_id"))
        active_order_id = self.menu.active_order_by_launchpad.get(launchpad_id)
        order_ids = [
            str(oid) for oid in launchpad.get("order_queue", [])
            if str(oid) not in used_orders
            and str(oid) in orders
            and orders[str(oid)].get("status") == OrderStatus.PENDING.value
            and (active_order_id is None or str(oid) == active_order_id)
        ]
        order_ids = self.menu.prune_order_ids(state, launchpad, order_ids, current_time, planned_departures)
        local_battery_ids = self._local_battery_ids(launchpad, batteries, used_batteries)
        best_battery_id = self._best_local_battery_id(local_battery_ids, batteries)
        best_battery = batteries.get(best_battery_id) if best_battery_id else None

        candidates: list[_JointPlanCandidate] = []
        for rank, order_id in enumerate(order_ids):
            order = orders[order_id]
            feasible = (
                best_battery is not None
                and self.menu.drone_can_carry(state, drone_id, order)
                and self.candidate_feasible_fn is not None
                and bool(self.candidate_feasible_fn(order, best_battery))
            )
            if feasible:
                candidates.append(_JointPlanCandidate(
                    "dispatch", order_id, None, None,
                    self._stage1_features("dispatch", order, launchpad, current_time, rank, len(order_ids)),
                ))
            candidates.append(_JointPlanCandidate(
                "defer", order_id, None, None,
                self._stage1_features("defer", order, launchpad, current_time, rank, len(order_ids)),
            ))
        if active_order_id is None and order_ids:
            candidates.append(_JointPlanCandidate(
                "stop", None, None, None,
                self._stage1_features("stop", None, launchpad, current_time, 0, len(order_ids)),
            ))
        return candidates

    def _stage1_features(
        self,
        kind: str,
        order: dict | None,
        launchpad: dict,
        current_time: int,
        queue_rank: int,
        queue_size: int,
    ) -> np.ndarray:
        horizon = max(1.0, float(self.config.get("simulation", {}).get("horizon_seconds", 1)))
        kind_features = [float(kind == "dispatch"), float(kind == "defer"), float(kind == "stop")]
        del queue_rank, queue_size  # F2 order block: value, slack, age, weight, delivery distance
        if order is None:
            order_features = [0.0] * 5
        else:
            order_features = [
                float(order.get("value", 0.0) or 0.0) / 50.0,
                (float(order.get("deadline", current_time)) - current_time) / horizon,
                max(0.0, current_time - float(order.get("created_at", current_time))) / horizon,
                float(order.get("weight", 0.0) or 0.0) / 5.0,
                distance_m(launchpad.get("current_location"), order.get("destination")) / 5000.0,
            ]
        raw = np.asarray(kind_features + order_features, dtype=np.float32)
        return self._pad(raw)

    # ------------------------------------------------------------------
    # Stage 2: return launchpad, full enumeration for the selected order
    # only, using the representative (highest-charge) local battery.
    # ------------------------------------------------------------------

    def _stage2_return_candidates(
        self,
        state: StateDict,
        launchpad: dict,
        order_id: str,
        used_batteries: set[str],
        current_time: int,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> list[_JointPlanCandidate]:
        order = state["orders"][order_id]
        batteries = state.get("batteries", {})
        launchpads = state.get("launchpads", {})
        departure_id = str(launchpad.get("launchpad_id"))
        local_battery_ids = self._local_battery_ids(launchpad, batteries, used_batteries)
        best_battery_id = self._best_local_battery_id(local_battery_ids, batteries)
        if best_battery_id is None or self.candidate_details_fn is None:
            return []
        best_battery = batteries[best_battery_id]
        details = self.candidate_details_fn(order, best_battery)
        if not bool(details.get("feasible", False)):
            return []
        return_plans = self.menu.prune_return_plans(
            order, launchpad, launchpads, details.get("return_plans", {}),
        )
        candidates: list[_JointPlanCandidate] = []
        for target_id, route in sorted(return_plans.items()):
            target = launchpads.get(str(target_id))
            if target is None or not bool(route.get("feasible", False)):
                continue
            if not self.menu.target_has_projected_capacity(
                departure_id, target, planned_targets, planned_departures,
            ):
                continue
            candidates.append(_JointPlanCandidate(
                "dispatch", order_id, None, str(target_id),
                self._stage2_features(order, launchpad, target, current_time, route, planned_targets, planned_departures),
            ))
        return candidates

    def _stage2_features(
        self,
        order: dict,
        departure: dict,
        target: dict,
        current_time: int,
        route_details: dict,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> np.ndarray:
        del current_time
        # same route block as F2: E_task/C, energy margin (representative battery), return distance
        route_features = [
            float(route_details.get("task_energy_ratio", 0.0) or 0.0),
            float(route_details.get("reserve_after_ratio", 0.0) or 0.0),
            float(route_details.get("distance_return_m", 0.0) or 0.0) / 5000.0,
        ]
        raw_target = self.menu.target_features(order, departure, target, planned_targets, planned_departures)
        target_features = [raw_target[0]] + [raw_target[1 + k] for k in TARGET_KEEP]   # same target block as F2
        raw = np.asarray(route_features + target_features, dtype=np.float32)
        return self._pad(raw)

    # ------------------------------------------------------------------
    # Stage 3: battery, order + return-launchpad both fixed, so this is a
    # feasibility re-check over the (usually 3-5) local batteries.
    # ------------------------------------------------------------------

    def _stage3_battery_candidates(
        self,
        state: StateDict,
        launchpad: dict,
        order_id: str,
        return_launchpad_id: str,
        used_batteries: set[str],
        current_time: int,
    ) -> list[_JointPlanCandidate]:
        order = state["orders"][order_id]
        batteries = state.get("batteries", {})
        local_battery_ids = self._local_battery_ids(launchpad, batteries, used_batteries)
        if self.candidate_details_fn is None:
            return []
        candidates: list[_JointPlanCandidate] = []
        for battery_id in local_battery_ids:
            battery = batteries[battery_id]
            details = self.candidate_details_fn(order, battery)
            if not bool(details.get("feasible", False)):
                continue
            route = details.get("return_plans", {}).get(str(return_launchpad_id))
            if route is None or not bool(route.get("feasible", False)):
                continue
            candidates.append(_JointPlanCandidate(
                "dispatch", order_id, battery_id, str(return_launchpad_id),
                self._stage3_features(battery, current_time, route),
            ))
        return candidates

    def _stage3_features(self, battery: dict, current_time: int, route_details: dict) -> np.ndarray:
        horizon = max(1.0, float(self.config.get("simulation", {}).get("horizon_seconds", 1)))
        capacity = float(battery.get("capacity_Wh", 0.0) or 0.0)
        charge = float(battery.get("current_charge_Wh", 0.0) or 0.0)
        completion = battery.get("charge_complete_at")
        battery_features = [
            safe_ratio(charge, capacity),
            charge / 1500.0,
            capacity / 1500.0,
            float(battery.get("status") == BatteryStatus.CHARGING.value),
            0.0 if completion is None else max(0.0, float(completion) - current_time) / horizon,
        ]
        route_features = [
            float(route_details.get("task_energy_ratio", 0.0) or 0.0),
            float(route_details.get("reserve_after_ratio", 0.0) or 0.0),
        ]
        raw = np.asarray(battery_features + route_features, dtype=np.float32)
        return self._pad(raw)

    @staticmethod
    def _pad(raw: np.ndarray) -> np.ndarray:
        if raw.shape[0] > CANDIDATE_DIM:
            raise RuntimeError(
                f"staged candidate feature width {raw.shape[0]} exceeds CANDIDATE_DIM={CANDIDATE_DIM}"
            )
        if raw.shape[0] == CANDIDATE_DIM:
            return raw
        padded = np.zeros((CANDIDATE_DIM,), dtype=np.float32)
        padded[: raw.shape[0]] = raw
        return padded
