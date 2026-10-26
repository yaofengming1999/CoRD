from __future__ import annotations

from typing import Any


_DIRECT_FIELDS = {
    "episode": "runtime/episode",
    "seed": "runtime/seed",
    "steps": "runtime/steps",
    "episode_wall_time_s": "runtime/wall_time_s",
    "policy_time_s": "runtime/policy_time_s",
    "env_time_s": "runtime/env_time_s",
    "energy_calls": "runtime/energy_calls",
    "energy_calls_env": "runtime/energy_calls_env",
    "simulator_drone_count": "simulator/drones",
    "simulator_order_count": "simulator/orders",
    "simulator_order_value_total": "simulator/order_value_total",
    "simulator_launchpad_count": "simulator/launchpads",
    "simulator_battery_count": "simulator/batteries",
    "on_time_gmv": "A_profit/on_time_gmv",
    "gross_delivered_value": "A_profit/gross_delivered_value",
    "net_profit": "A_profit/net_profit",
    "delay_penalty_cost": "A_profit/delay_penalty",
    "completion_rate": "E1_orders/completion_rate",
    "on_time_rate": "E1_orders/on_time_rate",
    "on_time_order_rate": "E1_orders/on_time_order_rate",
    "avg_lateness_min": "E1_orders/avg_lateness_min",
    "avg_fulfillment_time_min": "E1_orders/avg_fulfillment_time_min",
    "avg_wait_time_min": "E1_orders/avg_wait_time_min",
    "avg_dispatch_wait_min": "E1_orders/avg_dispatch_wait_min",
    "expired_rate": "E1_orders/expired_rate",
    "drone_total_distance_m": "E2_fleet/total_distance_m",
    "drone_total_energy_Wh": "E2_fleet/total_energy_Wh",
    "drone_rebalances": "E2_fleet/rebalances",
}

_METRIC_FIELDS = {
    "battery_avg_selected_soc": "E3_battery/avg_selected_soc",
    # slack over the whole out-and-back flight (was avg_dispatch_margin_ratio before
    # the delivery-only margin existed; same quantity, clearer name), then the same
    # slack up to the customer only and what the flight back costs -- see
    # MetricsRecorder.log_battery_dispatch_metrics for the split
    "battery_avg_roundtrip_margin_ratio": "E3_battery/avg_roundtrip_margin_ratio",
    "battery_avg_delivery_margin_ratio": "E3_battery/avg_delivery_margin_ratio",
    "battery_avg_return_cost_ratio": "E3_battery/avg_return_cost_ratio",
    "battery_avg_feasible_return_launchpads": (
        "E3_battery/avg_feasible_return_launchpads_when_dispatch"
    ),
    "battery_total_discharged_Wh": "E3_battery/total_discharged_Wh",
    "order_battery_feature_rows": "runtime/feature_rows",
    # F1b only: how often three independently drawn heads produce a triple
    # that is jointly infeasible, so the decision dispatches nothing
    "order_battery_joint_attempts": "runtime/joint_attempts",
    "order_battery_joint_infeasible": "runtime/joint_infeasible",
    "order_battery_joint_infeasible_rate": "runtime/joint_infeasible_rate",
    "battery_final_charged": "E3_battery/final_charged",
    "battery_final_charging": "E3_battery/final_charging",
    "battery_final_in_use": "E3_battery/final_in_use",
    "return_decisions": "D2_return/decisions",
    "return_avg_feasible_launchpads": "D2_return/avg_feasible_launchpads",
    "return_avg_distance_m": "D2_return/avg_distance_m",
    "return_avg_nearest_feasible_distance_m": (
        "D2_return/avg_nearest_feasible_distance_m"
    ),
    "return_avg_extra_distance_vs_nearest_m": (
        "D2_return/avg_extra_distance_vs_nearest_m"
    ),
    "return_avg_arrival_margin_ratio": "D2_return/avg_arrival_margin_ratio",
    "return_origin_rate": "D2_return/origin_rate",
    "return_avg_capacity_blocked_launchpads": (
        "D2_return/avg_capacity_blocked_launchpads"
    ),
    "rebalance_total_distance_m": "D3_rebalance/total_distance_m",
    "rebalance_total_energy_Wh": "D3_rebalance/total_energy_Wh",
    "rebalance_avg_selected_soc": "D3_rebalance/avg_selected_soc",
    "rebalance_avg_arrival_margin_ratio": "D3_rebalance/avg_arrival_margin_ratio",
    "order_battery_actor_loss": "P1_order_battery_learning/actor_loss",
    "order_battery_critic_loss": "P1_order_battery_learning/critic_loss",
    "order_battery_entropy": "P1_order_battery_learning/entropy",
    "order_battery_rollout_size": "P1_order_battery_learning/rollout_size",
    "order_battery_decision_groups": "D1_order_battery_decisions/decision_groups",
    "order_battery_candidates_per_group": (
        "D1_order_battery_decisions/candidates_per_group"
    ),
    "order_battery_selected_per_group": (
        "D1_order_battery_decisions/selected_per_group"
    ),
    "order_battery_defer_rate": "D1_order_battery_decisions/defer_rate",
    "order_battery_active_launchpads_per_time": (
        "D1_order_battery_decisions/active_launchpads_per_time"
    ),
    "order_battery_active_drones_per_time": (
        "D1_order_battery_decisions/active_drones_per_time"
    ),
    "order_battery_parallel_rounds": "D1_order_battery_decisions/parallel_rounds",
    "order_battery_parallel_agents_per_round": (
        "D1_order_battery_decisions/parallel_agents_per_round"
    ),
    "order_battery_conflicts": "D1_order_battery_decisions/conflicts",
    "order_battery_conflict_rate": "D1_order_battery_decisions/conflict_rate",
    "order_battery_halting_ratio": "D1_order_battery_decisions/halting_ratio",
    "drone_return_active_drones_per_time": (
        "D2_return/active_return_drones_per_time"
    ),
    "drone_rebalance_active_drones_per_time": (
        "D2_return/active_rebalance_drones_per_time"
    ),
    "drone_return_actor_loss": "P2_drone_learning/actor_loss",
    "drone_return_critic_loss": "P2_drone_learning/critic_loss",
    "drone_return_entropy": "P2_drone_learning/entropy",
    "drone_return_rollout_size": "P2_drone_learning/rollout_size",
    "drone_rebalance_actor_loss": "P3_rebalance_learning/actor_loss",
    "drone_rebalance_critic_loss": "P3_rebalance_learning/critic_loss",
    "drone_rebalance_entropy": "P3_rebalance_learning/entropy",
    "drone_rebalance_rollout_size": "P3_rebalance_learning/rollout_size",
}

_SVM_SUFFIXES = {
    "models",
    "candidate_evaluations",
    "auctions",
    "bids",
    "bid_rate",
    "no_bid_orders",
    "safety_vetoes",
    "dispatches",
    "updates",
    "positive_labels",
    "negative_labels",
    "mean_update_loss",
    "mean_winning_bid",
}


def episode_wandb_payload(
    row: dict[str, Any],
    *,
    namespace: str = "train",
    data_namespace: str = "instance_path",
) -> dict[str, Any]:
    prefix = _namespace(namespace)
    payload = {
        _prefixed_target(target, prefix): row[source]
        for source, target in _DIRECT_FIELDS.items()
        if source in row
    }
    payload[f"{prefix}_data/instance_id"] = row.get("selected_instance_id")
    payload[f"{prefix}_data/{data_namespace}"] = row.get("selected_instance_path")
    for source, target in _METRIC_FIELDS.items():
        if isinstance((value := row.get(source)), (int, float)):
            payload[_prefixed_target(target, prefix)] = value
    for suffix in _SVM_SUFFIXES:
        source = f"order_battery_svm_{suffix}"
        if isinstance((value := row.get(source)), (int, float)):
            payload[f"{prefix}_P1_order_battery_learning/svm_{suffix}"] = value
    return payload


def summary_wandb_payload(summary: dict[str, Any], *, namespace: str) -> dict[str, Any]:
    """Run metadata kept in Summary; full aggregate metrics remain on disk."""
    prefix = _namespace(namespace)
    fields = {
        "episodes": f"{prefix}_runtime/episodes",
        "episode_wall_time_s_mean": f"{prefix}_runtime/wall_time_s_mean",
        "steps_mean": f"{prefix}_runtime/steps_mean",
    }
    return {
        target: value
        for source, target in fields.items()
        if isinstance((value := summary.get(source)), (int, float))
    }


def validation_wandb_payload(
    summary: dict[str, Any],
) -> dict[str, Any]:
    """The single validation curve used for checkpoint selection."""
    return {"valid/on_time_gmv": float(summary.get("on_time_gmv_mean", 0.0))}


_EVAL_HISTORY_FIELDS = (
    "episode",
    "seed",
    "steps",
    "selected_instance_id",
    "selected_instance_path",
    "episode_wall_time_s",
    "policy_time_s",
    "env_time_s",
    "energy_calls",
    "energy_calls_env",
    "order_battery_feature_rows",
    "simulator_drone_count",
    "simulator_battery_count",
    "simulator_launchpad_count",
    "simulator_order_count",
    "simulator_order_value_total",
    "avg_wait_time_min",
    "avg_dispatch_wait_min",
    "on_time_gmv",
    "gross_delivered_value",
    "net_profit",
    "delay_penalty_cost",
    "completion_rate",
    "expired_rate",
    "on_time_rate",
    "on_time_order_rate",
    "avg_lateness_min",
    "avg_fulfillment_time_min",
    "drone_total_distance_m",
    "drone_total_energy_Wh",
    "drone_rebalances",
    "battery_avg_selected_soc",
    "battery_avg_roundtrip_margin_ratio",
    "battery_avg_delivery_margin_ratio",
    "battery_avg_return_cost_ratio",
    "battery_avg_feasible_return_launchpads",
    "battery_total_discharged_Wh",
    "return_avg_extra_distance_vs_nearest_m",
    "return_avg_arrival_margin_ratio",
    "return_origin_rate",
    "rebalance_total_energy_Wh",
    "order_battery_parallel_rounds",
    "order_battery_parallel_agents_per_round",
    "order_battery_conflicts",
    "order_battery_conflict_rate",
    "order_battery_halting_ratio",
)

def evaluation_episode_wandb_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Curated per-instance evaluation history, before aggregate reporting."""
    filtered_row = {key: row[key] for key in _EVAL_HISTORY_FIELDS if key in row}
    return episode_wandb_payload(filtered_row, namespace="eval")


def evaluation_wandb_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Core held-out metrics; the complete aggregate remains in eval_summary.json."""
    fields = {
        "episode_wall_time_s_mean": "eval_runtime/wall_time_s_mean",
        "policy_time_s_mean": "eval_runtime/policy_time_s_mean",
        "env_time_s_mean": "eval_runtime/env_time_s_mean",
        "energy_calls_mean": "eval_runtime/energy_calls_mean",
        "energy_calls_env_mean": "eval_runtime/energy_calls_env_mean",
        "order_battery_feature_rows_mean": "eval_runtime/feature_rows_mean",
        "on_time_gmv_mean": "eval_A_profit/on_time_gmv_mean",
        "gross_delivered_value_mean": "eval_A_profit/gross_delivered_value_mean",
        "net_profit_mean": "eval_A_profit/net_profit_mean",
        "delay_penalty_cost_mean": "eval_A_profit/delay_penalty_mean",
        # same eight as the per-episode E1_orders block, as held-out means
        "completion_rate_mean": "eval_E1_orders/completion_rate_mean",
        "expired_rate_mean": "eval_E1_orders/expired_rate_mean",
        "on_time_rate_mean": "eval_E1_orders/on_time_rate_mean",
        "on_time_order_rate_mean": "eval_E1_orders/on_time_order_rate_mean",
        "avg_lateness_min_mean": "eval_E1_orders/avg_lateness_min_mean",
        "avg_wait_time_min_mean": "eval_E1_orders/avg_wait_time_min_mean",
        "avg_dispatch_wait_min_mean": "eval_E1_orders/avg_dispatch_wait_min_mean",
        "avg_fulfillment_time_min_mean": "eval_E1_orders/avg_fulfillment_time_min_mean",
        "drone_total_distance_m_mean": "eval_E2_fleet/total_distance_m_mean",
        "drone_total_energy_Wh_mean": "eval_E2_fleet/total_energy_Wh_mean",
        "drone_total_energy_Wh_std": "eval_E2_fleet/total_energy_Wh_std",
        "drone_rebalances_mean": "eval_E2_fleet/rebalances_mean",
        "battery_avg_selected_soc_mean": "eval_E3_battery/avg_selected_soc_mean",
        "battery_avg_roundtrip_margin_ratio_mean": (
            "eval_E3_battery/avg_roundtrip_margin_ratio_mean"
        ),
        "battery_avg_delivery_margin_ratio_mean": (
            "eval_E3_battery/avg_delivery_margin_ratio_mean"
        ),
        "battery_avg_return_cost_ratio_mean": (
            "eval_E3_battery/avg_return_cost_ratio_mean"
        ),
        "battery_avg_feasible_return_launchpads_mean": (
            "eval_E3_battery/avg_feasible_return_launchpads_when_dispatch_mean"
        ),
        "battery_total_discharged_Wh_mean": (
            "eval_E3_battery/total_discharged_Wh_mean"
        ),
        "return_avg_extra_distance_vs_nearest_m_mean": (
            "eval_D2_return/avg_extra_distance_vs_nearest_m_mean"
        ),
        "return_avg_arrival_margin_ratio_mean": (
            "eval_D2_return/avg_arrival_margin_ratio_mean"
        ),
        "return_origin_rate_mean": "eval_D2_return/origin_rate_mean",
        "rebalance_total_energy_Wh_mean": "eval_D3_rebalance/total_energy_Wh_mean",
        "order_battery_parallel_rounds_mean": (
            "eval_D1_order_battery_decisions/parallel_rounds_mean"
        ),
        "order_battery_parallel_agents_per_round_mean": (
            "eval_D1_order_battery_decisions/parallel_agents_per_round_mean"
        ),
        "order_battery_active_drones_per_time_mean": (
            "eval_D1_order_battery_decisions/active_drones_per_time_mean"
        ),
        "order_battery_conflicts_mean": (
            "eval_D1_order_battery_decisions/conflicts_mean"
        ),
        "order_battery_conflict_rate_mean": (
            "eval_D1_order_battery_decisions/conflict_rate_mean"
        ),
        "order_battery_halting_ratio_mean": (
            "eval_D1_order_battery_decisions/halting_ratio_mean"
        ),
        "order_battery_svm_bid_rate_mean": (
            "eval_P1_order_battery_learning/svm_bid_rate_mean"
        ),
        "order_battery_svm_no_bid_orders_mean": (
            "eval_P1_order_battery_learning/svm_no_bid_orders_mean"
        ),
        "order_battery_svm_safety_vetoes_mean": (
            "eval_P1_order_battery_learning/svm_safety_vetoes_mean"
        ),
        "order_battery_svm_dispatches_mean": (
            "eval_P1_order_battery_learning/svm_dispatches_mean"
        ),
        "order_battery_svm_mean_winning_bid_mean": (
            "eval_P1_order_battery_learning/svm_mean_winning_bid_mean"
        ),
    }
    payload = {
        target: float(summary[source])
        for source, target in fields.items()
        if source in summary
    }
    payload["eval_runtime/episodes"] = int(summary.get("episodes", 0))
    return payload


def _namespace(value: str) -> str:
    return str(value).strip().replace("/", "_")


def _prefixed_target(target: str, namespace: str) -> str:
    group, metric = target.split("/", 1)
    return f"{namespace}_{group}/{metric}"
