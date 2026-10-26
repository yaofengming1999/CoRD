"""F2 Separable -- the proposed method: <OL+B+LowRank>+SerializedMARL.

The implementation is the package ``algorithm/F2`` (policy.py, features,
menu.py, actor.py, critic.py, learner). This module only exposes the preset
class name used by the dispatcher registry.
"""

from __future__ import annotations

from algorithm.F2.policy import F2Policy


class F2SeparableOrderBatteryPolicy(F2Policy):
    """F2: per-drone trajectories, olb35 layout, (o,l)|b separable actor + factor-V critic."""
