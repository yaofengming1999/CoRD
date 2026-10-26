"""AS2: <OL+B+LowRank>+SingleDronePPO.

Drone-ordered sequential decisions with ONE PPO trajectory per episode (team
reward).

Same decode, olb layout, (o,l)|b actor and candidate-Q critic as the proposed
F2_Separable; the only difference is that F2 cuts the stream into one trajectory
per drone. (Like AS1/AS3, this baseline compares the agent structure only, so it
uses the proposed input construction.)
"""

from __future__ import annotations

from algorithm.F2.policy import F2Policy


class AS2SingleDronePPOOrderBatteryPolicy(F2Policy):
    """AS2 = F2 with ONE GAE trajectory per episode and the team reward
    (F2 cuts one trajectory per drone with own-order rewards)."""

    trajectory_split = "episode"
    agent_structure = "SingleDronePPO"
    dispatch_action_prefix = "drone_trajectory_joint_plan_ppo_dispatch"
    joint_plan_config_key = "order_battery_drone_trajectory_joint_plan_ppo"
    joint_plan_feature_schema = "joint_dispatch_return_plan_single_drone_ppo_v3"
