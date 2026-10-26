"""F3: FullCombination+SerializedMARL.

SerializedMARL structure (one GAE trajectory per drone) scoring the FULL flat
(order, return, battery) menu with a full-capacity actor.

Control for F2_Separable: identical decode, trajectories, critic and training;
only the actor's factorisation of the triplet differs.
"""

from __future__ import annotations

from algorithm.F2.policy import F2Policy


class F3FullCombinationOrderBatteryPolicy(F2Policy):
    """F3 = F2 with the expanded flat33 triplet rows scored by the full-combination
    actor and the row-based candidate-Q critic (the learner's default actor)."""

    candidate_layout = "flat"
    actor_type = "flat"                # full-combination actor; the learner then defaults to the row critic
    dispatch_action_prefix = "drone_ordered_counterfactual_ppo_dispatch"
    joint_plan_config_key = "order_battery_drone_ordered_counterfactual_ppo"
    joint_plan_feature_schema = "joint_dispatch_return_plan_drone_ordered_counterfactual_v3"
