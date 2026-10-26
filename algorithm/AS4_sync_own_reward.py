"""AS4: <OL+B+LowRank>+MAPPO with the OWN-ORDER reward.

AS1 with one knob moved. The decode is unchanged -- every idle drone of a step
decides from the same pre-commitment state, blind to the others, and conflicts
are resolved by rule afterwards -- and so is the actor. What changes is the
credit: instead of one shared team advantage per physical step, each drone gets
its own GAE trajectory over the steps it acted in, carrying only the rewards of
the orders it dispatched.

The critic has to follow. AS1's V(s) estimates the team's return from the pad
set, which is the right baseline for a team reward and the wrong one for an
individual reward: the advantage would subtract a team-sized value from an
own-order-sized reward. So AS4 uses ``GlobalStateAgentCritic``, V(s, drone) --
the same pooled pad state plus the acting drone's own context, as used by the
per-agent structures. That is not a free design choice; it is what makes the
individual advantage well defined.

AS4 completes the pair that identifies the two axes separately:

    AS1 vs AS4   simultaneous decode, team vs own-order credit
    F2  vs AS5   sequential decode,   own-order vs team credit
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import nn

from algorithm.AS1_mappo import (
    MAPPOStructure,
    SynchronousDroneMAPPOActionSample,
    TorchSynchronousDroneMAPPO,
)
from algorithm.F2.critic import FactorVCritic, GlobalStateAgentCritic, global_state_batch
from algorithm.F2.menu import stack_factor_menus
from algorithm.common.mappo_utils import temporal_team_rewards


class TorchSyncOwnRewardMAPPO(TorchSynchronousDroneMAPPO):
    """Simultaneous actors, per-drone credit.

    The rollout holds one entry per (step, drone) rather than one per step, each
    with its own advantage and return, so the inherited PPO update -- which
    already flattens transitions into per-agent pairs -- applies unchanged. The
    critic is the only part that differs: it is evaluated per agent.
    """

    # the value of a step depends on which drone is acting, so it cannot be read
    # once per step at acting time; store_agent_episode computes V(s, drone) instead
    computes_acting_value = False

    def __init__(self, **kwargs):
        # critic_type picks the value head. "global_state" is AS4 as the module
        # docstring argues it: V(s, drone) over the active-pad context set, the
        # baseline an own-order reward needs. "factor_v" swaps in F2's own head,
        # V(s) from the pooled factored menu, so that AS4 against F2 moves the
        # decode and nothing else -- without it the pair differs in two knobs
        # at once and the price of the within-step reservation is not isolated.
        critic_type = str(kwargs.pop("critic_type", "global_state"))
        if critic_type not in ("global_state", "factor_v"):
            raise ValueError(
                f"AS4 supports critic_type 'global_state' or 'factor_v', got {critic_type!r}"
            )
        super().__init__(**kwargs)
        self.critic_type = critic_type
        self.critic = (
            FactorVCritic(self.context_dim, self.hidden_dim)
            if critic_type == "factor_v"
            else GlobalStateAgentCritic(self.context_dim, self.hidden_dim)
        ).to(self.device)
        self._initialize_optimizers()

    def _factor_values(self, menus: list, contexts: np.ndarray) -> torch.Tensor:
        """F2's V(s) head, one menu per agent (critic_type='factor_v').

        ``stack_factor_menus`` pads the per-agent menus to one batch, exactly as
        F2's learner does for its own critic, so the two read the same quantity.
        """
        arrays = stack_factor_menus(menus)

        def t(key: str, dtype) -> torch.Tensor:
            return torch.as_tensor(arrays[key], dtype=dtype, device=self.device)

        return self.critic.value_from_factors(
            t("factor_ol", torch.float32), t("factor_b", torch.float32),
            t("factor_ol_mask", torch.bool), t("factor_b_mask", torch.bool),
            torch.as_tensor(contexts, dtype=torch.float32, device=self.device),
        )

    def _agent_values(self, sample: SynchronousDroneMAPPOActionSample) -> np.ndarray:
        """V(s, a) for every active agent of one step."""
        agents = int(sample.agent_contexts.shape[0])
        if self.critic_type == "factor_v":
            if not sample.factors:
                return np.zeros(agents, dtype=np.float64)
            with torch.no_grad():
                values = self._factor_values(sample.factors, sample.agent_contexts)
            return values.cpu().numpy().astype(np.float64, copy=False)
        state = sample.global_state
        if state is None or len(state) == 0:
            return np.zeros(agents, dtype=np.float64)
        pads, pad_mask = global_state_batch(
            [np.asarray(state, dtype=np.float32)] * agents, self.context_dim, self.device
        )
        own = torch.as_tensor(sample.agent_contexts, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.critic(pads, pad_mask, own)
        return values.cpu().numpy().astype(np.float64, copy=False)

    def store_agent_episode(
        self,
        trajectories: list[list[tuple[SynchronousDroneMAPPOActionSample, int, int]]],
        rewards: list[list[float]],
        episode_end_time: int,
    ) -> None:
        """One GAE pass per drone over (sample, agent index, time) triples.

        Each trajectory is the steps one physical drone acted in, in time order,
        with the rewards of the orders it dispatched. The discounting is the same
        time-aware form AS1 uses, over that drone's own step times, so a drone
        that acts rarely is not implicitly discounted differently from one that
        acts every step.
        """
        if not trajectories:
            return
        if len(trajectories) != len(rewards):
            raise ValueError("one reward series per drone trajectory is required")
        cache: dict[int, np.ndarray] = {}
        for steps, reward_series in zip(trajectories, rewards):
            if not steps or len(steps) != len(reward_series):
                continue
            values = np.empty(len(steps), dtype=np.float64)
            for position, (sample, agent_index, _time) in enumerate(steps):
                key = id(sample)
                if key not in cache:
                    cache[key] = self._agent_values(sample)
                row = cache[key]
                values[position] = row[agent_index] if agent_index < len(row) else 0.0
            times = [int(time) for _s, _a, time in steps]
            scaled = np.asarray(reward_series, dtype=np.float64) / self.reward_scale
            advantages = np.zeros(len(steps), dtype=np.float64)
            gae = 0.0
            for index in range(len(steps) - 1, -1, -1):
                terminal = index == len(steps) - 1
                next_time = episode_end_time if terminal else times[index + 1]
                delta_t = max(0.0, float(next_time - times[index]))
                discount = self.gamma ** (delta_t / self.time_unit_s)
                trace = self.gae_lambda ** (delta_t / self.time_unit_s)
                next_value = 0.0 if terminal else values[index + 1]
                delta = scaled[index] + discount * next_value - values[index]
                gae = delta + (0.0 if terminal else discount * trace * gae)
                advantages[index] = gae
            returns = advantages + values
            for position, (sample, agent_index, _time) in enumerate(steps):
                self._joint_rollout.append({
                    "sample": _single_agent_view(sample, agent_index),
                    "advantage": float(advantages[position]),
                    "return": float(returns[position]),
                })

    def _critic_forward(self, batch):
        """V(s, drone) for the batch.

        Every rollout entry is one (step, drone) pair, so the entry's single
        agent context is the drone whose return is being estimated. The actor
        side of the inherited update needs no change: it already flattens a
        transition into its active agents, and here there is exactly one.
        """
        contexts = np.stack([row["sample"].agent_contexts[0] for row in batch["rows"]])
        if self.critic_type == "factor_v":
            return self._factor_values(
                [row["sample"].factors[0] for row in batch["rows"]], contexts
            )
        pads, pad_mask = batch["pad_critic_inputs"]
        own = torch.as_tensor(contexts, dtype=torch.float32, device=self.device)
        return self.critic(pads, pad_mask, own)


def _single_agent_view(
    sample: SynchronousDroneMAPPOActionSample, agent_index: int
) -> SynchronousDroneMAPPOActionSample:
    """The one agent's slice of a step's joint sample, as its own sample.

    The PPO update reads a rollout entry as a transition and iterates its active
    agents; giving it a one-agent view is what turns AS1's shared-advantage
    update into a per-drone one without touching the update itself.
    """
    index = slice(agent_index, agent_index + 1)
    return SynchronousDroneMAPPOActionSample(
        candidate_features=sample.candidate_features[index],
        candidate_mask=sample.candidate_mask[index],
        agent_contexts=sample.agent_contexts[index],
        agent_mask=sample.agent_mask[index],
        actions=sample.actions[index],
        agent_log_probs=sample.agent_log_probs[index],
        log_prob=float(sample.agent_log_probs[agent_index]),
        value=0.0,                       # per-agent values are recomputed by the critic
        factors=None if sample.factors is None else [sample.factors[agent_index]],
        global_state=sample.global_state,
    )


class AS4SyncOwnRewardOrderBatteryPolicy(MAPPOStructure):
    """AS4 = AS1 with per-drone trajectories and own-order rewards."""

    agent_structure = "MAPPO_OwnReward"
    dispatch_action_prefix = "drone_synchronous_mappo_dispatch"
    joint_plan_config_key = "order_battery_drone_synchronous_mappo"
    joint_plan_feature_schema = "joint_dispatch_return_plan_synchronous_mappo_v3"

    learner_class = TorchSyncOwnRewardMAPPO

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

        terminal_time = int(
            episode_end_time
            if episode_end_time is not None
            else (self._episode_decisions[-1].time if self._episode_decisions else 0)
        )
        # The policy appends one decision per active drone, in the order
        # _collect_fleet_menus produced them, which is the agent order of that
        # step's joint sample. So a decision's slot is its position among the
        # decisions of its own step.
        slot_of: dict[int, int] = {}
        seen_per_time: dict[int, int] = defaultdict(int)
        for decision in self._episode_decisions:
            time = int(decision.time)
            slot_of[id(decision)] = seen_per_time[time]
            seen_per_time[time] += 1

        by_drone: dict[str, list] = defaultdict(list)
        for decision in self._episode_decisions:
            key = decision.drone_id or f"launchpad:{decision.launchpad_id}"
            by_drone[str(key)].append(decision)

        trajectories: list[list[tuple[Any, int, int]]] = []
        reward_series: list[list[float]] = []
        for decisions in by_drone.values():
            steps: list[tuple[Any, int, int]] = []
            for decision in sorted(decisions, key=lambda d: int(d.time)):
                sample = decision.sample
                agent_index = slot_of[id(decision)]
                if agent_index >= int(sample.agent_contexts.shape[0]):
                    continue        # defensive: a slot past the sample's agents
                steps.append((sample, agent_index, int(decision.time)))
            if not steps:
                continue
            # Only the orders this drone actually flew. Its pick is not always its
            # action here: all drones choose from the same pre-commitment menu, so
            # two can name one order and the rule keeps the first in fleet order.
            # Crediting the chosen order instead would pay the loser for the
            # winner's delivery and count that delivery's reward twice.
            own_orders = {
                str(decision.order_id)
                for decision in decisions
                if decision.kind == "dispatch" and decision.order_id is not None
                and self.dispatch_was_accepted(decision)
            }
            events = [
                event for event in (reward_events or [])
                if event.get("order_id") is not None and str(event["order_id"]) in own_orders
            ] if own_orders else []
            trajectories.append(steps)
            reward_series.append(
                temporal_team_rewards([time for _s, _a, time in steps], events)
            )

        self.learner.store_agent_episode(trajectories, reward_series, terminal_time)
        epochs = self.gradient_steps_per_episode if gradient_steps is None else int(gradient_steps)
        stats = self.learner.update(epochs=epochs)
        result = {
            "order_battery_actor_loss": float(stats["actor_loss"]),
            "order_battery_critic_loss": float(stats["critic_loss"]),
            "order_battery_entropy": float(stats["entropy"]),
            "order_battery_rollout_size": float(stats["rollout_size"]),
            "order_battery_decisions": float(decision_count),
            "order_battery_feature_rows": float(self.menu.feature_rows),
        }
        result.update(self._metrics.summary())
        self.clear_episode()
        return result
