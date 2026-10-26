"""F2 -- the proposed dispatch policy: <OL+B+LowRank>+SerializedMARL.

One concrete class. Per physical time step the policy walks the fleet in the
state's drone order; every waiting drone at a launchpad gets one decision over
the launchpad's factored (o,l)|b menu (``algorithm/F2/menu.py``): dispatch
(order, return launchpad, battery), defer, or stop. Decisions are PPO
transitions; at episode end they are cut into one GAE trajectory per drone
(``trajectory_split="drone"``), each seeing only the rewards of the orders it
dispatched (SerializedMARL). ``trajectory_split="episode"`` keeps one
trajectory with the team reward (AS2).

The other learned methods are this class plus one replaced part:
  F3  layout=flat + full-combination actor/row critic      (action expansion)
  F1  staged o -> l -> b decode learner                       (action expansion)
  AS2 trajectory_split="episode"                              (agent structure)
  AS1 synchronous plan + conflict resolution + global critic  (agent structure; L3 on top)
  AS3 one joint transition per step + joint learner           (agent structure; L4 on top)
  L2  independent per-drone learners (IPPO)                   (agent structure)

Sections: constructor / learner kwargs; per-step decision loop; episode end and
training hand-off; candidate and feature construction (kept here rather than in
a base class so the whole method reads top to bottom in one file).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithm.F2.features import MenuBuilder
from algorithm.F2 import featmask
from algorithm.F2.menu import BATTERY, CANDIDATE_DIMS, CROSS, FactorMenu, obl_from_factor_menu
from algorithm.F2.policy_types import _JointPlanCandidate, _JointPlanDecision
from algorithm.common.decision_metrics import OrderBatteryDecisionMetrics
from algorithm.common.mappo_utils import mean_unique_agents_per_time, temporal_team_rewards
from algorithm.F2.learner import F2Learner, MAPPOActionSample
from simulator.model.types import StateDict, TaskLevelActionDict


class F2Policy:
    """Sequential drone-ordered decisions on the factored (o,l)|b menu; per-drone trajectories."""

    # ---- identity / layout (subclasses override what differs) ----
    candidate_layout = "olb"                   # "olb" factored menu | "flat" expanded triplets (F3)
    menu_construction = "factored"             # "factored" | "triplet" (reference builder, tests only)
    trajectory_split = "drone"                 # "drone" (SerializedMARL) | "episode" (AS2)
    reward_attribution = "own"                 # "own" (this drone's orders) | "team" (AS5)
    battery_blind = False                      # True: no battery state, random battery (AS6)
    factor_split = "ol_b"                      # "ol_b" (proposed) | "ob_l" (F0OneDecision)
    plans_return = True                        # False (F0): the dispatch action names no return pad
    collapse_return = False                    # True (F0): one menu row per (order, battery)
    # F0 only: full | route_only | order_only, i.e. how much of the collapsed pad
    # l* the dispatch row keeps. The default is order_only -- the dispatch agent
    # observes the order and the batteries and nothing else, which is what "it
    # decides (o, b) only" means at the level of what it can see. Feasibility stays
    # the simulator's, so the action set still reflects what the cheapest return
    # costs even though no feature does; that is exactly the information a
    # dispatcher which does not model the return would have, namely which pairs are
    # legal and not why. The two looser specifications are reported as variants:
    # route_only adds min_l E_task(o,l) and the matching return distance, full adds
    # l*'s queue state as well. Measured against F2 at 62 matched episodes on Hong
    # Kong: -6.2% order_only, -4.1% route_only, -1.7% full.
    return_feature_mode = "order_only"
    agent_structure = "SerializedMARL"
    context_dim = 23
    candidate_dim = CANDIDATE_DIMS["olb"]
    learner_class = F2Learner
    actor_type = "olb"                         # "olb" (o,l)|b separable actor | "flat" full-combination actor (F3)
    dispatch_action_prefix = "d2_separable_ppo_dispatch"
    joint_plan_config_key = "order_battery_d2_separable_ppo"
    joint_plan_feature_schema = "joint_dispatch_return_plan_d2_separable_v1"
    dispatch_candidate_error = (
        "joint-plan PPO dispatch candidate is missing order, battery, or return launchpad"
    )

    # ------------------------------------------------------------------ #
    # constructor and learner
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # constructor and learner
    # ------------------------------------------------------------------ #
    def __init__(self, config: dict, *, seed: int = 0):
        if self.trajectory_split not in {"episode", "drone"}:
            raise ValueError("trajectory_split must be 'episode' or 'drone'")
        if self.reward_attribution not in {"own", "team"}:
            raise ValueError("reward_attribution must be 'own' or 'team'")
        self.config = config
        decision_cfg = config.get("drone_decision", {})
        cfg = dict(decision_cfg.get(self.joint_plan_config_key, {}))
        if decision_cfg.get("critic_type") is not None:  # CLI --critic-type override
            cfg["critic_type"] = decision_cfg["critic_type"]
        if decision_cfg.get("interaction_rank") is not None:  # CLI --interaction-rank
            cfg["interaction_rank"] = decision_cfg["interaction_rank"]
        # This is a pathological-menu guard. The learner pads only within each
        # PPO minibatch, and O1 applies full attention to each decision group.
        cfg.setdefault("max_candidates_per_launchpad", 262_144)
        if self.candidate_layout not in CANDIDATE_DIMS:
            raise ValueError(f"unknown candidate_layout {self.candidate_layout!r}")
        self.candidate_dim = CANDIDATE_DIMS[self.candidate_layout]
        self.training = bool(config.get("rl", {}).get("training", False))
        self.max_candidates = int(cfg.get("max_candidates_per_launchpad", 1024))
        self.gradient_steps_per_episode = int(cfg.get("ppo_epochs", 8))
        # A runtime/global value (used by experiment scripts) overrides the
        # policy section; otherwise each policy keeps its own default.
        urgent_order_multiplier = float(decision_cfg.get("urgent_order_multiplier", cfg.get("urgent_order_multiplier", 2.0)))
        # 0 (the default) disables the return-pad corridor: pruning acts on the order dimension only
        return_corridor_km = float(decision_cfg.get("return_corridor_km", cfg.get("return_corridor_km", 0.0)))
        if urgent_order_multiplier <= 0.0:
            raise ValueError("urgent_order_multiplier must be positive")
        if return_corridor_km < 0.0:
            raise ValueError("return_corridor_km must be non-negative (0 = no corridor)")
        # Which input columns reach the networks. Configured before the menu and
        # the learner are built, because both are sized from it: the menu writes
        # the masked blocks and the learner's layers must match their width.
        self.feat_mask = featmask.configure(
            decision_cfg.get("feature_preset", cfg.get("feature_preset", "full")),
            decision_cfg.get("feature_drop", cfg.get("feature_drop")),
        )
        # menu / feature construction (features.py); simulator hooks are bound on it by build_dispatcher
        self.menu = MenuBuilder(
            config,
            layout=self.candidate_layout,
            construction=self.menu_construction,
            collapse_return=self.collapse_return,
            return_feature_mode=decision_cfg.get(
                "return_feature_mode", cfg.get("return_feature_mode", self.return_feature_mode)
            ),
            context_dim=self.context_dim,
            max_candidates=self.max_candidates,
            pruning=bool(decision_cfg.get("candidate_pruning", cfg.get("candidate_pruning", False))),
            urgent_order_multiplier=urgent_order_multiplier,
            return_corridor_km=return_corridor_km,
        )
        self._step_global_state = None
        # the battery draw of the battery-blind ablation; seeded so a run repeats
        self._blind_rng = np.random.default_rng(seed + 977)
        self._episode_decisions: list[_JointPlanDecision] = []
        self._current_training_trajectory = None
        self._metrics = OrderBatteryDecisionMetrics()
        self.learner = self.learner_class(
            **self._learner_kwargs(cfg, seed=seed),
        )


    # ---- simulator hooks (set by build_dispatcher), forwarded to the menu builder ----
    @property
    def candidate_feasible_fn(self):
        return self.menu.candidate_feasible_fn

    @candidate_feasible_fn.setter
    def candidate_feasible_fn(self, fn):
        self.menu.candidate_feasible_fn = fn

    @property
    def candidate_details_fn(self):
        return self.menu.candidate_details_fn

    @candidate_details_fn.setter
    def candidate_details_fn(self, fn):
        self.menu.candidate_details_fn = fn

    @property
    def route_energy_fn(self):
        return self.menu.route_energy_fn

    @route_energy_fn.setter
    def route_energy_fn(self, fn):
        self.menu.route_energy_fn = fn

    @property
    def battery_state_fn(self):
        return self.menu.battery_state_fn

    @battery_state_fn.setter
    def battery_state_fn(self, fn):
        self.menu.battery_state_fn = fn

    def _learner_kwargs(self, cfg: dict, *, seed: int) -> dict:
        kwargs = {
            "candidate_dim": self.candidate_dim,
            # the masked width: what the context actually arrives as
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
            "batch_size": int(cfg.get("batch_size", 64)),
            "ppo_epochs": self.gradient_steps_per_episode,
            "time_unit_s": float(cfg.get("time_unit_s", 60.0)),
            "reward_scale": float(cfg.get("reward_scale", 25.0)),
            "device": str(cfg.get("device", "cuda")),
            "seed": int(cfg.get("seed", seed)),
        }
        kwargs["feature_schema"] = self.joint_plan_feature_schema
        kwargs["actor_type"] = self.actor_type
        kwargs["interaction_rank"] = int(cfg.get("interaction_rank", 8))
        # value baseline: candidate_q (row Q) | factor_q ((o,l)|b Q) | global_state (pad set + own context);
        # unset -> the learner's own default (F3 row learner: candidate_q; OLB learner: factor_q)
        if cfg.get("critic_type") is not None:
            kwargs["critic_type"] = str(cfg["critic_type"])
        return kwargs

    # ------------------------------------------------------------------ #
    # per-step decision loop
    # ------------------------------------------------------------------ #
    def _begin_plan(self, state: StateDict) -> None:
        """Per-step setup shared by every decode (also called by AS1's override)."""
        self.menu.begin_step(state)
        # one pre-commitment global state per step ([P, 23] active-pad contexts) for the
        # global_state critic; within-step commitments are carried by each drone's own context row
        self._step_global_state = self.menu.global_state(state) if self._uses_global_state_critic else None

    def plan(
        self,
        state: StateDict,
        used_orders: set[str],
        used_batteries: set[str],
    ) -> list[TaskLevelActionDict]:
        self._begin_plan(state)
        current_time = int(state.get("time", 0))
        actions: list[TaskLevelActionDict] = []
        planned_targets: dict[str, int] = {}
        planned_departures: dict[str, int] = {}
        blocked_launchpads: set[str] = set()
        cleared_launchpads: set[str] = set()

        # StateBuilder preserves the simulator's default drone insertion order.
        fleet_size = len(state.get("drones", {}))
        for agent_index, (raw_drone_id, drone) in enumerate(
            state.get("drones", {}).items()
        ):
            drone_id = str(raw_drone_id)
            launchpad_id = drone.get("current_launchpad_id")
            if launchpad_id is None:
                continue
            launchpad_id = str(launchpad_id)
            if launchpad_id in blocked_launchpads:
                continue
            launchpad = state.get("launchpads", {}).get(launchpad_id)
            if launchpad is None or not self.menu.drone_is_available(
                state,
                launchpad,
                drone_id,
            ):
                continue
            if launchpad_id not in cleared_launchpads:
                self.menu.clear_terminal_lock(state, launchpad_id, launchpad)
                cleared_launchpads.add(launchpad_id)

            candidates = self.menu.candidates(
                state,
                launchpad,
                used_orders,
                used_batteries,
                current_time,
                planned_targets,
                planned_departures,
            )
            candidates = self.menu.filter_for_drone(state, drone_id, candidates)
            if not candidates:
                blocked_launchpads.add(launchpad_id)
                continue

            menu = self._blind_battery(self._regroup(self._menu_input(candidates), candidates))
            selected_index, sample = self._select_action_for_agent(
                state,
                drone_id,
                menu,
                self.menu.context(
                    state,
                    launchpad,
                    used_orders,
                    used_batteries,
                    1,
                    planned_targets,
                    planned_departures,
                    agent_index=agent_index,
                    agent_count=fleet_size,
                ),
            )
            selected = candidates[selected_index]
            if self.battery_blind:
                selected = self._random_battery_for(selected, candidates)
            if self.training:
                self._episode_decisions.append(
                    _JointPlanDecision(
                        current_time,
                        sample,
                        selected.kind,
                        launchpad_id,
                        drone_id,
                        selected.order_id,
                    )
                )

            dispatch_candidates = [
                candidate
                for candidate in candidates
                if candidate.kind == "dispatch"
            ]
            self._metrics.record_group(
                order_ids=[
                    candidate.order_id
                    for candidate in dispatch_candidates
                    if candidate.order_id
                ],
                battery_ids=[
                    candidate.battery_id
                    for candidate in dispatch_candidates
                    if candidate.battery_id
                ],
                selected_count=int(selected.kind == "dispatch"),
                available_drones=1,
                selected_kind=selected.kind,
                selected_order_has_feasible_battery=(
                    any(
                        candidate.order_id == selected.order_id
                        for candidate in dispatch_candidates
                    )
                    if selected.kind == "defer"
                    else None
                ),
            )
            if selected.kind == "stop":
                blocked_launchpads.add(launchpad_id)
                continue
            if selected.kind == "defer":
                if selected.order_id is not None:
                    self.menu.active_order_by_launchpad[launchpad_id] = selected.order_id
                blocked_launchpads.add(launchpad_id)
                continue
            if selected.order_id is None or selected.battery_id is None or (
                self.plans_return and selected.return_launchpad_id is None
            ):
                raise RuntimeError(self.dispatch_candidate_error)

            action: TaskLevelActionDict = {
                "action_id": (
                    f"{self.dispatch_action_prefix}_{current_time}_"
                    f"{drone_id}_{selected.order_id}_{selected.battery_id}"
                ),
                "action_type": "order_battery_dispatch",
                "order_id": selected.order_id,
                "battery_id": selected.battery_id,
                "drone_id": drone_id,
                "feasibility": True,
            }
            if self.plans_return:
                action["action_id"] += f"_{selected.return_launchpad_id}"
                action["target_launchpad_id"] = selected.return_launchpad_id
            actions.append(action)
            used_orders.add(selected.order_id)
            used_batteries.add(selected.battery_id)
            planned_departures[launchpad_id] = (
                planned_departures.get(launchpad_id, 0) + 1
            )
            if self.plans_return:
                # only a committed return occupies the target pad; F0 has not
                # chosen one yet, so it projects no arrival
                planned_targets[selected.return_launchpad_id] = (
                    planned_targets.get(selected.return_launchpad_id, 0) + 1
                )
            if self.menu.active_order_by_launchpad.get(launchpad_id) == selected.order_id:
                self.menu.active_order_by_launchpad.pop(launchpad_id, None)

        return actions

    def _select_action(
        self,
        state: StateDict,
        candidate_features,
        actor_context: np.ndarray,
    ) -> tuple[int, MAPPOActionSample]:
        if self._step_global_state is None:
            return self.learner.act(candidate_features, actor_context, training=self.training)
        return self.learner.act(
            candidate_features, actor_context, training=self.training,
            critic_context=self._step_global_state,
        )

    def _select_action_for_agent(
        self,
        state: StateDict,
        drone_id: str,
        candidate_features,
        actor_context: np.ndarray,
    ):
        del drone_id
        return self._select_action(state, candidate_features, actor_context)

    def _collect_fleet_menus(self, state: StateDict) -> list[tuple]:
        """Pre-commitment fleet view of this step: for every available drone
        ``(drone_id, launchpad_id, launchpad, candidates, context, menu)`` with the
        pad menu built once per pad (no orders/batteries used yet) and filtered
        to what that drone can carry. Used as the synchronous action menus by
        MAPPO and as the centralised critic's state by MAPPO and GlobalDronesPPO."""
        current_time = int(state.get("time", 0))
        active: list[tuple] = []
        pad_menus: dict[str, tuple[list[_JointPlanCandidate], FactorMenu | None]] = {}
        fleet_size = len(state.get("drones", {}))
        for agent_index, (raw_drone_id, drone) in enumerate(state.get("drones", {}).items()):
            drone_id = str(raw_drone_id)
            launchpad_id = drone.get("current_launchpad_id")
            if launchpad_id is None:
                continue
            launchpad_id = str(launchpad_id)
            launchpad = state.get("launchpads", {}).get(launchpad_id)
            if launchpad is None or not self.menu.drone_is_available(state, launchpad, drone_id):
                continue
            self.menu.clear_terminal_lock(state, launchpad_id, launchpad)
            cached = pad_menus.get(launchpad_id)
            if cached is None:
                candidates = self.menu.candidates(state, launchpad, set(), set(), current_time, {}, {})
                pad_menus[launchpad_id] = (candidates, self.menu.factor_menu)
            else:
                candidates, self.menu.factor_menu = cached
            candidates = self.menu.filter_for_drone(state, drone_id, candidates)
            if not candidates:
                continue
            menu = self._menu_input(candidates)
            context = self.menu.context(
                state, launchpad, set(), set(), 1, {}, {},
                agent_index=agent_index, agent_count=fleet_size,
            )
            active.append((drone_id, launchpad_id, launchpad, candidates, context, menu))
        return active

    @property
    def _uses_global_state_critic(self) -> bool:
        return getattr(self.learner, "critic_type", "candidate_q") == "global_state"

    @property
    def scored_rows(self) -> int:
        """Rows the actor was actually given this episode.

        For the row path that is one row per menu entry; for the factored path it
        is N_ol + N_b, since the product is never built. A staged decode scores
        its stages separately and counts them on the learner, so it overrides
        this. The number is what a decision time has to be read next to: the
        same policy is cheap or dear according to how many rows it scores.
        """
        return int(self.menu.feature_rows)

    def _blind_battery(self, menu):
        """Blank the battery block, so the actor cannot tell the batteries apart.

        Table 4 shows the score matrix over (order, return pad) rows and battery
        columns needs only rank two: a row effect g and a column effect h. This
        removes the column effect as well, leaving rank one. The battery rows all
        become the same vector, so h is one constant across them and every entry
        sharing an (order, return pad) ties -- the actor is choosing the pair, and
        the battery is then drawn at random.

        Feasibility is untouched: the menu is still built from the real charges, so
        a drawn battery can always fly the task. Only the actor's view is blanked.
        """
        if not self.battery_blind or menu is None:
            return menu
        if isinstance(menu, FactorMenu):
            return FactorMenu(menu.ol_raw, menu.b_raw, menu.ol, np.zeros_like(menu.b),
                              menu.kind, menu.ol_idx, menu.b_idx, menu.n_ol, menu.n_b)
        rows = np.array(menu, dtype=np.float32, copy=True)      # [N, D] row path
        rows[:, BATTERY] = 0.0
        rows[:, CROSS] = 0.0          # E_task/C, margin, d_ret -- the first two carry the battery
        return rows

    def _random_battery_for(self, selected, candidates):
        """The same (order, return pad) with a battery drawn uniformly at random."""
        if selected.kind != "dispatch" or selected.order_id is None:
            return selected
        same = [
            candidate for candidate in candidates
            if candidate.kind == "dispatch"
            and candidate.order_id == selected.order_id
            and candidate.return_launchpad_id == selected.return_launchpad_id
            and candidate.battery_id is not None
        ]
        return same[int(self._blind_rng.integers(len(same)))] if same else selected

    def _regroup(self, menu, candidates: list[_JointPlanCandidate]):
        """Put the return pad on the other side of the split, if asked.

        The entries are unchanged -- this regroups the features of the menu the
        builder already produced -- so the two splits offer the same actions and
        differ only in which block holds the pad.
        """
        if self.factor_split == "ol_b" or not isinstance(menu, FactorMenu):
            return menu
        return obl_from_factor_menu(menu, candidates)

    def _menu_input(self, candidates: list[_JointPlanCandidate]):
        """What the learner scores for this decision: the FactorMenu when the
        builder produced one and the learner can consume it, else the [N, D]
        rows (rebuilt from the factors if needed). Also feeds ``_feature_rows``."""
        menu = self.menu.factor_menu
        if menu is not None and len(menu) == len(candidates):
            if getattr(self.learner, "supports_factor_menu", False):
                self.menu.feature_rows += menu.factor_rows
                return menu
            self.menu.feature_rows += len(candidates)
            return menu.rows35()
        self.menu.feature_rows += len(candidates)
        return np.stack([candidate.features for candidate in candidates])

    # ------------------------------------------------------------------ #
    # episode end: trajectories, rewards, learner update, checkpoints
    # ------------------------------------------------------------------ #
    def finish_episode(
        self,
        on_time_gmv: float,
        *,
        reward_events: list[dict] | None = None,
        episode_end_time: int | None = None,
        gradient_steps: int | None = None,
        **kwargs,
    ) -> dict[str, float]:
        decision_count = len(self._episode_decisions)
        active_launchpads_per_time = mean_unique_agents_per_time(
            [
                (decision.time, decision.launchpad_id)
                for decision in self._episode_decisions
            ]
        )
        if not self.training:
            result = {
                "order_battery_actor_loss": 0.0,
                "order_battery_critic_loss": 0.0,
                "order_battery_entropy": 0.0,
                "order_battery_rollout_size": 0.0,
                "order_battery_decisions": 0.0,
                "order_battery_feature_rows": float(self.scored_rows),
            }
            result.update(self._metrics.summary())
            self.clear_episode()
            return result
        terminal_time = int(
            episode_end_time
            if episode_end_time is not None
            else (self._episode_decisions[-1].time if self._episode_decisions else 0)
        )
        for trajectory in self._training_trajectory_groups():
            self._current_training_trajectory = trajectory
            times = [decision.time for decision in trajectory]
            trajectory_events = self._trajectory_reward_events(
                trajectory,
                reward_events,
            )
            rewards = temporal_team_rewards(times, trajectory_events)
            self._store_training_trajectory(
                [decision.sample for decision in trajectory],
                rewards,
                times,
                terminal_time,
            )
            self._current_training_trajectory = None
        epochs = self.gradient_steps_per_episode if gradient_steps is None else int(gradient_steps)
        stats = self.learner.update(epochs=epochs)
        result = {
            "order_battery_actor_loss": float(stats["actor_loss"]),
            "order_battery_critic_loss": float(stats["critic_loss"]),
            "order_battery_entropy": float(stats["entropy"]),
            "order_battery_rollout_size": float(stats["rollout_size"]),
            "order_battery_decisions": float(decision_count),
            "order_battery_feature_rows": float(self.scored_rows),
            "order_battery_active_launchpads_per_time": active_launchpads_per_time,
        }
        result.update(self._metrics.summary())
        self.clear_episode()
        return result

    def _training_trajectory_groups(
        self,
    ) -> list[list[_JointPlanDecision]]:
        if self.trajectory_split == "episode":
            return [self._episode_decisions]
        trajectories: dict[str, list[_JointPlanDecision]] = {}
        for decision in self._episode_decisions:
            # STOP/DEFER can have no physical drone assignment if the launchpad
            # has no eligible idle drone; keep those decisions in a stable
            # launchpad fallback trajectory instead of dropping them.
            key = decision.drone_id or f"launchpad:{decision.launchpad_id}"
            trajectories.setdefault(str(key), []).append(decision)
        return list(trajectories.values())

    def _trajectory_reward_events(
        self,
        trajectory: list[_JointPlanDecision],
        reward_events: list[dict] | None,
    ) -> list[dict]:
        """Keep only rewards for orders dispatched by this physical drone.

        With ``trajectory_split == "episode"`` there is one trajectory and it
        keeps the full team reward stream; there is nothing to attribute.

        Reward events are emitted at the order level. Passing the full team
        event list into every drone trajectory duplicates each delivery reward
        and destroys the per-drone credit assignment the per-drone split exists
        for -- which is precisely what ``reward_attribution == "team"`` (AS5)
        does on purpose, to price that attribution with the cut held fixed.
        """
        if self.trajectory_split == "episode" or self.reward_attribution == "team":
            return reward_events
        order_ids = {
            str(decision.order_id)
            for decision in trajectory
            if decision.kind == "dispatch" and decision.order_id is not None
        }
        if not order_ids:
            return []
        return [
            event
            for event in (reward_events or [])
            if event.get("order_id") is not None
            and str(event.get("order_id")) in order_ids
        ]

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
        )

    def clear_episode(self) -> None:
        self._episode_decisions = []
        self._current_training_trajectory = None
        self._metrics.clear()
        self.menu.reset_episode()
        self.learner.clear_rollout()

    def save(self, path: str) -> None:
        self.learner.save(path)

    def load(self, path: str) -> None:
        self.learner.load(path)

