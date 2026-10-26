"""L5: HRL4AG (KDD 2026), UAV-only simulator adapter.

The paper's manager chooses between air and ground fleets, then mode-specific
workers score feasible order-vehicle pairs. This simulator has no ground
carrier, so the manager is forced to the UAV branch and the worker still makes
the complete order-battery-return decision.
"""

from __future__ import annotations

from algorithm.AS3_global_drones_ppo import AS3GlobalDronesPPOOrderBatteryPolicy


class HRL4AGOrderBatteryPolicy(AS3GlobalDronesPPOOrderBatteryPolicy):
    """HRL4AG worker/manager adapter for a drone-only environment."""

    candidate_layout = "flat"
    actor_type = "flat"                 # full-triplet actor on the expanded rows (AS3 learner, flat actor)
    menu_construction = "triplet"
    dispatch_action_prefix = "hrl4ag_uav_worker_dispatch"
    joint_plan_config_key = "order_battery_global_ppo"
    joint_plan_feature_schema = "hrl4ag_uav_worker_flat33_v1"
