"""F2pro's policy: F2's decode, F2pro's scoring.

What is inherited, and why. The decode loop, the candidate generation and the
feasibility test come from F2Policy. That is deliberate and it is the only reuse:
the two methods have to face the same actions or a difference between them is not
the scoring structure. Everything else is this package's -- the menu, the actor,
the critic, the learner, the rollout and the update.

What is replaced:
  _menu_input      the FactorMenu is regrouped into a TriMenu of o, l and b rows
  learner_class    F2ProLearner, which owns its own GAE and PPO step
  scored_rows      N_o + N_l + N_b rather than N_ol + N_b
  _learner_kwargs  the TriSeparableActor takes no candidate_dim: its widths are
                   fixed by the three blocks, so the row-layout arguments F2's
                   learner needs would be meaningless here
"""

from __future__ import annotations

from typing import Any

import numpy as np

from algorithm.F2.menu import FactorMenu
from algorithm.F2.policy import F2Policy
from algorithm.F2pro.learner import F2ProLearner
from algorithm.F2pro.menu import TriMenu, tri_from_candidates


class F2ProPolicy(F2Policy):
    """Three blocks, coupled only between order and return pad."""

    learner_class = F2ProLearner
    joint_plan_config_key = "order_battery_f2pro_ppo"
    joint_plan_feature_schema = "f2pro_o_l_b_v1"
    dispatch_action_prefix = "f2pro_dispatch"

    def __init__(self, config: dict, *, seed: int = 0):
        self._tri_rows = 0          # rows the actor scored this episode
        super().__init__(config, seed=seed)

    def _learner_kwargs(self, cfg: dict, *, seed: int) -> dict:
        """Only what F2ProLearner takes.

        F2's kwargs describe a row layout (candidate_dim, actor_type, critic_type)
        that has no counterpart here: the three blocks' widths are fixed by the
        menu, and there is one actor and one critic. Passing F2's set through
        would either be ignored silently or configure something that does not
        exist, so the arguments are listed explicitly.
        """
        decision_cfg = self.config.get("drone_decision", {})
        rank = cfg.get("interaction_rank", 2)
        if decision_cfg.get("interaction_rank") is not None:      # CLI --interaction-rank
            rank = decision_cfg["interaction_rank"]
        return {
            "context_dim": int(self.feat_mask.context_dim),   # the masked width
            "hidden_dim": int(cfg.get("hidden_dim", 128)),
            "interaction_rank": int(rank),
            "coord_frequencies": int(decision_cfg.get("coord_frequencies",
                                                      cfg.get("coord_frequencies", 0))),
            "coord_scale": float(cfg.get("coord_scale", 2.0)),
            "actor_lr": float(cfg.get("actor_lr", 3e-4)),
            "critic_lr": float(cfg.get("critic_lr", 1e-3)),
            "gamma": float(cfg.get("gamma", 0.99)),
            "gae_lambda": float(cfg.get("gae_lambda", 0.95)),
            "clip_ratio": float(cfg.get("clip_ratio", 0.2)),
            "value_coefficient": float(cfg.get("value_coefficient", 0.5)),
            "entropy_coefficient": float(cfg.get("entropy_coefficient", 0.01)),
            "max_grad_norm": float(cfg.get("max_grad_norm", 10.0)),
            "batch_size": int(cfg.get("batch_size", 32)),
            "ppo_epochs": int(cfg.get("ppo_epochs", 8)),
            "time_unit_s": float(cfg.get("time_unit_s", 60.0)),
            "reward_scale": float(cfg.get("reward_scale", 25.0)),
            "device": str(cfg.get("device", "cuda")),
            "seed": int(cfg.get("seed", seed)),
        }

    def _menu_input(self, candidates: list) -> TriMenu:
        """Regroup F2's menu into three blocks and count what the actor scores."""
        menu = self.menu.factor_menu
        if menu is None or len(menu.ol_idx) != len(candidates):
            raise RuntimeError("F2pro needs the factored menu F2's builder produces")
        tri = tri_from_candidates(menu, candidates)
        self._tri_rows += tri.scored_rows
        return tri

    @property
    def scored_rows(self) -> int:
        return int(self._tri_rows)

    def clear_episode(self) -> None:
        self._tri_rows = 0
        super().clear_episode()

    def _regroup(self, menu, candidates):
        return menu          # the regrouping already happened in _menu_input

    def _blind_battery(self, menu):
        return menu          # AS6's ablation does not apply to this layout

    def _select_action(self, state, candidate_features, actor_context) -> tuple[int, Any]:
        """One actor, one critic: no central-critic context to thread through."""
        return self.learner.act(candidate_features, actor_context, training=self.training)

    def _store_training_trajectory(self, samples, rewards, decision_times, terminal_time) -> None:
        self.learner.store_episode(samples, rewards, decision_times, terminal_time)
