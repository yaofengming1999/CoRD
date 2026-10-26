"""AS5: <OL+B+LowRank>+SerializedMARL with the TEAM reward.

F2 with one knob moved. The decode is unchanged (drone-ordered sequential, each
drone seeing the menu with the previous drones' commitments already removed) and
the cut is unchanged (one GAE trajectory per physical drone); only the reward
stream each trajectory carries changes, from the orders that drone dispatched to
every order the fleet delivered.

This is not AS2. AS2 collapses the episode into a single trajectory, so it moves
the cut and the reward together; here the cut stays per-drone, which is what
isolates the attribution: a drone's trajectory is credited for deliveries it had
no part in, and every delivery reward is counted once per drone.
"""

from __future__ import annotations

from algorithm.F2.policy import F2Policy


class AS5TeamRewardOrderBatteryPolicy(F2Policy):
    """AS5 = F2 with the team reward stream on per-drone trajectories."""

    reward_attribution = "team"
    agent_structure = "SerializedMARL"
