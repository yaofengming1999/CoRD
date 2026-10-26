from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from algorithm.dispatch.dispatch import DroneDecisionDispatcher
from algorithm.dispatch.presets import ALGORITHM_CHOICES
from simulator.core.simulator_env import SimulatorEnv


def build_dispatcher(name: str, config: dict, env: SimulatorEnv, seed: int):
    if name in ALGORITHM_CHOICES:
        dispatcher = DroneDecisionDispatcher(config, seed=seed)
        dispatcher.order_battery_policy.candidate_feasible_fn = env.order_battery_candidate_feasible
        if hasattr(dispatcher.order_battery_policy, "candidate_details_fn"):
            dispatcher.order_battery_policy.candidate_details_fn = env.order_battery_candidate_details
        if hasattr(dispatcher.order_battery_policy, "route_energy_fn"):
            dispatcher.order_battery_policy.route_energy_fn = env.order_route_energy_options
            dispatcher.order_battery_policy.battery_state_fn = env.battery_dispatch_state
        dispatcher.drone_return_policy.return_feasible_fn = env.drone_return_candidate_feasible
        dispatcher.drone_return_policy.rebalance_feasible_fn = env.drone_rebalance_candidate_feasible
        return dispatcher
    raise ValueError(f"Unknown dispatcher: {name}")


def scenario_key(config_name: str) -> str:
    scenario_name = Path(config_name).name
    if scenario_name.endswith(".yaml"):
        scenario_name = Path(scenario_name).stem
    if scenario_name.startswith("scenario_"):
        scenario_name = scenario_name[len("scenario_"):]
    return scenario_name


def save_run_record(
    env: SimulatorEnv,
    run_dir: Path,
    metrics: dict,
    city_name: str,
    scenario_name: str,
    dispatcher_name: str,
    seed: int,
    simulator_config_name: str,
    algorithm_config_name: str | None,
    config_sources: dict | None = None,
) -> Path:
    metadata = {
        "city": city_name,
        "scenario": scenario_name,
        "dispatcher": dispatcher_name,
        "seed": seed,
        "recorded_at": datetime.now().isoformat(),
        "config_name": simulator_config_name,
        "simulator_config_name": simulator_config_name,
        "algorithm_config_name": algorithm_config_name,
        "config_sources": config_sources or {},
        "horizon_seconds": env.horizon,
        "max_horizon_seconds": env.max_horizon,
        "resolved_max_horizon_seconds": env.config.get("_runtime", {}).get("resolved_max_horizon_seconds"),
        "delta_t_seconds": env.delta_t,
        "final_time_seconds": env.current_time,
        "termination_reason": env.last_termination_reason,
        "time_advancement": env.config.get("simulation", {}).get("time_advancement"),
        "selected_instance_path": env.config.get("_runtime", {}).get("selected_instance_path"),
    }
    env.metrics_recorder.save_recording(run_dir, metadata=metadata)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return run_dir
