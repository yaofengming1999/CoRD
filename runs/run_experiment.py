#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from algorithm.dispatch.presets import ALGORITHM_CHOICES, apply_algorithm_preset
from algorithm.dispatch.dispatch import ORDER_BATTERY_POLICY_NAMES
from runs.utils.experiment_helpers import build_dispatcher, save_run_record, scenario_key
from runs.utils.metrics_payload import (
    episode_wandb_payload,
    evaluation_episode_wandb_payload,
    evaluation_wandb_payload,
    summary_wandb_payload,
    validation_wandb_payload,
)
from runs.utils.output_paths import explicit_run_output_dir, slug, timestamp
from runs.utils.wandb_utils import add_wandb_args, init_wandb
from simulator.core.simulator_env import SimulatorEnv
from simulator.metrics.compute import compute_episode_metrics
from simulator.model.enums import OrderStatus
from simulator.io.city_registry import known_cities
from simulator.utils.config import load_config


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def launchpad_capacity_arg(value: str) -> str | int:
    normalized = str(value).strip().lower()
    if normalized in {"unlimited", "inf", "infinity"}:
        return "unlimited"
    try:
        capacity = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "launchpad capacity must be 'unlimited' or a positive integer"
        ) from exc
    if capacity <= 0:
        raise argparse.ArgumentTypeError(
            "launchpad capacity must be 'unlimited' or a positive integer"
        )
    return capacity


def positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone-only battery dispatch experiment runner")
    parser.add_argument("--is-train", type=str_to_bool, required=True)
    parser.add_argument("--is-learning", type=str_to_bool, required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--time-advance-mode", choices=["fixed_step", "event_driven"], required=True)
    parser.add_argument(
        "--delta-t-seconds",
        type=positive_int_arg,
        default=None,
        help=(
            "Fixed-step interval in seconds. Defaults to simulation.delta_t_seconds "
            "from the simulator config and is only valid with fixed_step."
        ),
    )
    parser.add_argument("--algorithm", choices=ALGORITHM_CHOICES, default="drone_decision")
    parser.add_argument("--config", "--sim-config", dest="sim_config", default="scenario_default")
    parser.add_argument("--algorithm-config", default="drone_decision")
    parser.add_argument(
        "--order-battery-policy",
        choices=ORDER_BATTERY_POLICY_NAMES,
        default=None,
    )
    parser.add_argument(
        "--drone-return-policy",
        choices=[
            "heuristic",
            "nearest",
            "random",
            "preplanned",
        ],
        default=None,
    )
    parser.add_argument("--drone-rebalance", type=str_to_bool, default=None)
    parser.add_argument("--candidate-pruning", type=str_to_bool, default=None)
    # which input columns reach the networks: a preset name from
    # algorithm/F2/featmask.py, plus any extra feature ids to drop on top
    parser.add_argument("--feature-preset", default=None,
                        choices=["full", "lossless", "aggressive"])
    parser.add_argument("--feature-drop", nargs="+", default=None,
                        metavar="ID", help="feature ids (C5, T3, B1, ...) to drop")
    parser.add_argument("--urgent-order-multiplier", type=float, default=None)
    parser.add_argument("--return-corridor-km", type=float, default=None)
    parser.add_argument(
        "--critic-type", choices=("factor_v", "factor_q", "candidate_q", "global_state"), default=None,
        help="value baseline of the sequential-drones learners (F2/AS2); overrides the yaml",
    )
    parser.add_argument(
        "--return-feature-mode", choices=("full", "route_only", "order_only"), default=None,
        help="F0 only: how much of the collapsed return pad the dispatch row keeps",
    )
    parser.add_argument(
        "--coord-frequencies", type=int, default=None,
        help="F2pro only: replace each block's (x, y) with [p, |p|^2, sin(Omega p), "
             "cos(Omega p)] using this many frequencies. 0 keeps the raw coordinates. "
             "A rank-2m bilinear form can then approximate the return distance, so "
             "m=4 is the setting matched to rank 8.",
    )
    parser.add_argument(
        "--interaction-rank", type=int, default=None,
        help="rank of the separable actor's bilinear (o,l)x b coupling; 0 drops the "
             "term, leaving a purely additive score. Overrides the yaml.",
    )
    parser.add_argument(
        "--freeze-order-battery",
        type=str_to_bool,
        default=False,
        help="Keep a loaded order-battery policy fixed during training.",
    )
    parser.add_argument(
        "--freeze-drone-return",
        type=str_to_bool,
        default=False,
        help=(
            "Keep the loaded return learner fixed during training."
        ),
    )
    parser.add_argument(
        "--launchpad-capacity",
        type=launchpad_capacity_arg,
        default=None,
        help="Runtime launchpad holding capacity: 'unlimited' or a positive "
             "integer. Defaults to simulation.launchpad_capacity (unlimited in "
             "the default scenario).",
    )
    # --- the scenario path: build the fleet from counts instead of reading an
    # instance folder. Giving --scenario-city switches the loader over; the
    # instance flags are then unused. Only the fleet is parametric here -- the
    # pads come from the frozen layout, see simulator/io/launchpad_layout.py.
    parser.add_argument("--scenario-city", default=None,
                        help="city under data/dataset/order_instance; enables the scenario loader")
    parser.add_argument("--scenario-drones-per-launchpad", type=int, default=None)
    parser.add_argument("--scenario-batteries-per-drone", type=int, default=None)
    # None, not 1: an unset flag must not override the frozen battery ladder
    parser.add_argument("--scenario-battery-types", type=int, default=None)
    parser.add_argument("--scenario-pad-spacing-km", type=float, default=None)
    parser.add_argument("--scenario-extent", default=None)
    parser.add_argument("--scenario-window", default=None,
                        help="episode window on the source day axis, START-END hours "
                             "(default 11-12), or 'all' for the whole 08:00-12:00 file")
    parser.add_argument("--scenario-drone-placement", choices=["uniform", "demand"], default=None)
    parser.add_argument("--scenario-min-drones-per-pad", type=int, default=None)
    parser.add_argument("--instance-root", default="data/instances")
    parser.add_argument("--instance-selection", choices=["random_saved", "fixed", "all_saved"], default=None)
    parser.add_argument("--fixed-instance-id", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--records-root", default="records")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--save-model-path", default=None)
    parser.add_argument("--valid-every", type=int, default=0)
    parser.add_argument("--valid-episodes", type=int, default=None)
    parser.add_argument("--valid-instance-selection", choices=["random_saved", "fixed", "all_saved"], default="all_saved")
    parser.add_argument("--valid-fixed-instance-id", default=None)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--eval-instance-selection", choices=["random_saved", "fixed", "all_saved"], default="all_saved")
    parser.add_argument("--eval-fixed-instance-id", default=None)
    parser.add_argument(
        "--eval-action-mode",
        choices=["greedy", "stochastic", "grouped_greedy"],
        default="greedy",
        help=(
            "Evaluation-only action selection. grouped_greedy first selects "
            "dispatch/defer/stop by total probability mass, then selects the "
            "highest-probability action within that kind."
        ),
    )
    parser.add_argument(
        "--record-replay",
        nargs="?",
        const=True,
        type=str_to_bool,
        default=False,
        help="Record per-step frames + save the HTML replay (dominant episode cost; "
        "off by default; metrics are unaffected).",
    )
    parser.add_argument(
        "--verbose",
        nargs="?",
        const=True,
        type=str_to_bool,
        default=False,
    )
    add_wandb_args(parser, default_project="drones-experiments")
    return parser.parse_args()


def scenario_plan(
    *,
    city: str,
    split: str,
    selection: str,
    episodes: int | None,
    seed: int,
    fixed_id: str | None,
) -> list[dict]:
    """The scenario path's analogue of ``instance_plan``: a day is an episode.

    There are no instance folders to enumerate, so the unit is a day CSV of the
    split. ``fixed_id`` is the day index as a string, and ``scenario_day`` is what
    ``run_episode`` feeds the loader, so a valid or eval pass replays exactly the
    days it names.
    """
    from simulator.io.city_orders import day_files

    days = len(day_files(city, split))
    if selection == "fixed":
        if fixed_id is None:
            raise ValueError("fixed instance selection requires --fixed-instance-id")
        day = int(fixed_id)
        if not 0 <= day < days:
            raise ValueError(f"{city}/{split} has {days} days; --fixed-instance-id {fixed_id} is out of range")
        count = int(episodes or 1)
        return [
            {"episode": i, "seed": seed + i, "fixed_id": str(day), "scenario_day": day}
            for i in range(count)
        ]
    if selection == "all_saved":
        return [
            {"episode": day, "seed": seed + day, "fixed_id": str(day), "scenario_day": day}
            for day in range(days)
        ]
    count = int(episodes or 1)
    return [
        {
            "episode": i,
            "seed": seed + i,
            "fixed_id": str((seed + i) % days),
            "scenario_day": (seed + i) % days,
        }
        for i in range(count)
    ]


def instance_plan(
    *,
    root: str,
    city: str,
    split: str,
    selection: str,
    episodes: int | None,
    seed: int,
    fixed_id: str | None,
    scenario_city: str | None = None,
) -> list[dict]:
    if scenario_city is not None:
        return scenario_plan(
            city=scenario_city, split=split, selection=selection,
            episodes=episodes, seed=seed, fixed_id=fixed_id,
        )
    split_dir = _REPO_ROOT / root / city / split
    if selection == "fixed":
        if fixed_id is None:
            raise ValueError("fixed instance selection requires --fixed-instance-id")
        count = int(episodes or 1)
        return [{"episode": index, "seed": seed + index, "fixed_id": fixed_id} for index in range(count)]
    candidates = sorted(
        path.name
        for path in split_dir.iterdir()
        if path.is_dir()
        and all((path / filename).is_file() for filename in ("orders.yaml", "drones.yaml", "launchpads.yaml", "batteries.yaml", "weather.yaml"))
    )
    if not candidates:
        raise FileNotFoundError(f"No drone instances found under {split_dir}")
    if selection == "all_saved":
        return [{"episode": index, "seed": seed + index, "fixed_id": fixed_id} for index, fixed_id in enumerate(candidates)]
    count = int(episodes or 1)
    return [
        {"episode": index, "seed": seed + index, "fixed_id": candidates[(seed + index) % len(candidates)]}
        for index in range(count)
    ]


def set_learning_mode(
    dispatcher,
    enabled: bool,
    *,
    evaluation_action_mode: str = "greedy",
) -> None:
    runtime = dispatcher.config.get("_runtime", {})
    dispatcher.order_battery_policy.training = bool(
        enabled and not runtime.get("freeze_order_battery", False)
    )
    return_policy = dispatcher.drone_return_policy
    if hasattr(return_policy, "set_training_mode"):
        return_policy.set_training_mode(
            enabled,
            train_return=not runtime.get("freeze_drone_return", False),
            train_rebalance=True,
        )
    else:
        return_policy.training = enabled
    order_learner = getattr(dispatcher.order_battery_policy, "learner", None)
    if order_learner is not None and hasattr(
        order_learner,
        "set_evaluation_action_mode",
    ):
        order_learner.set_evaluation_action_mode(evaluation_action_mode)
    if hasattr(return_policy, "set_evaluation_action_mode"):
        return_mode = (
            "stochastic"
            if evaluation_action_mode == "stochastic"
            else "greedy"
        )
        return_policy.set_evaluation_action_mode(return_mode)
    if not enabled:
        dispatcher.clear_episode()


def run_episode(
    *,
    env: SimulatorEnv,
    dispatcher,
    config: dict,
    item: dict,
    split: str,
    learning: bool,
    record_dir: Path | None,
    evaluation_action_mode: str = "greedy",
) -> dict:
    config["instance"]["split"] = split
    config["instance"]["selection"] = "fixed"
    config["instance"]["fixed_id"] = item["fixed_id"]
    if config.get("scenario", {}).get("enabled"):
        # the scenario loader reads its own split and day; `fixed_id` names the
        # day so a valid/eval pass is reproducible the same way
        config["scenario"]["split"] = split
        config["scenario"]["day"] = item.get("scenario_day")
    set_learning_mode(
        dispatcher,
        learning,
        evaluation_action_mode=evaluation_action_mode,
    )
    state = env.reset(seed=int(item["seed"]))
    total_reward = 0.0
    started = perf_counter()
    policy_time = 0.0  # dispatcher.plan: candidate build + features + forward + selection
    env_time = 0.0  # apply_actions + step
    # task-energy evaluations split the same way: inside plan() (candidate
    # construction, the efficiency metric) vs inside the simulator transition
    energy_calls_policy = 0
    energy_calls_env = 0
    done = False
    while not done:
        t0 = perf_counter()
        c0 = int(getattr(env.air_router, "task_energy_calls", 0))
        actions = dispatcher.plan(state)
        c1 = int(getattr(env.air_router, "task_energy_calls", 0))
        t1 = perf_counter()
        env.apply_actions(actions)
        state, reward, done, _ = env.step()
        env_time += perf_counter() - t1
        policy_time += t1 - t0
        energy_calls_policy += c1 - c0
        energy_calls_env += int(getattr(env.air_router, "task_energy_calls", 0)) - c1
        total_reward += reward

    metrics = compute_episode_metrics(
        env.orders,
        env.drones,
        env.horizon,
        config,
        env.launchpads,
        env.batteries,
    )
    # The env's step-reward stream is sum(value - lateness penalty) over deliveries, so it must
    # reconcile exactly with the episode metrics (every delivery lands before the episode closes).
    expected_reward = metrics["gross_delivered_value"] - metrics["delay_penalty_cost"]
    assert abs(total_reward - expected_reward) <= 1e-6 * max(1.0, abs(expected_reward)), (
        f"total_reward {total_reward} != gross_delivered_value - delay_penalty {expected_reward}"
    )
    order_rewards = {
        order.order_id: order.value
        if order.status == OrderStatus.DELIVERED
        and order.delivery_time is not None
        and order.delivery_time <= order.deadline
        else 0.0
        for order in env.orders.values()
    }
    lateness_penalty_per_min = float(config["cost"]["lateness_penalty_per_min"])
    reward_events = sorted(
        [
            {
                "time": int(order.delivery_time),
                "reward": float(order.value) - max(0.0, (order.delivery_time - order.deadline) / 60.0) * lateness_penalty_per_min,
                "order_id": order.order_id,
            }
            for order in env.orders.values()
            if order.status == OrderStatus.DELIVERED and order.delivery_time is not None
        ],
        key=lambda event: event["time"],
    )
    learner_stats = dispatcher.finish_episode(
        metrics["on_time_gmv"],
        order_rewards=order_rewards,
        reward_events=reward_events,
        episode_end_time=env.current_time,
        gradient_steps=int(config.get("drone_decision", {}).get("gradient_steps_per_episode", 8)),
    )
    row = {
        "episode": int(item["episode"]),
        "seed": int(item["seed"]),
        "dispatcher": dispatcher_label(config),
        "selected_instance_id": item["fixed_id"],
        "selected_instance_path": str(env.selected_instance_path),
        "steps": env._step_count,
        "total_reward": total_reward,
        "simulator_drone_count": len(env.drones),
        "simulator_order_count": len(env.orders),
        "simulator_order_value_total": sum(order.value for order in env.orders.values()),
        "simulator_launchpad_count": len(env.launchpads),
        "simulator_battery_count": len(env.batteries),
        "episode_wall_time_s": perf_counter() - started,
        "policy_time_s": policy_time,
        "env_time_s": env_time,
        "energy_calls": energy_calls_policy,  # candidate construction only (N_energy in the paper)
        "energy_calls_env": energy_calls_env,  # simulator-side (dispatch execution / return checks)
        "recorded_at": datetime.now().isoformat(),
        **metrics,
        **learner_stats,
        **env.metrics_recorder.operational_summary(),
    }
    delivered = float(row.get("orders_delivered", 0.0) or 0.0)
    row["energy_per_delivery_Wh"] = (
        float(row.get("drone_total_energy_Wh", 0.0) or 0.0) / delivered
        if delivered > 0.0
        else 0.0
    )
    if record_dir is not None:
        save_run_record(
            env,
            record_dir,
            row,
            config["instance"]["city"],
            scenario_key(config.get("_config_sources", {}).get("simulator", "scenario_default")),
            dispatcher_label(config),
            int(item["seed"]),
            Path(config.get("_config_sources", {}).get("simulator", "scenario_default")).stem,
            Path(config.get("_config_sources", {}).get("algorithm", "drone_decision")).stem,
            config.get("_config_sources"),
        )
    return row


def dispatcher_label(config: dict) -> str:
    algorithm = str(config.get("_runtime", {}).get("algorithm", "drone_decision"))
    if algorithm != "drone_decision":
        return algorithm
    decision = config.get("drone_decision", {})
    rebalance = "__rebalance" if decision.get("rebalance", {}).get("enabled", False) else ""
    return (
        f"ob_{decision.get('order_battery_policy', 'heuristic')}"
        f"__ret_{decision.get('drone_return_policy', 'heuristic')}{rebalance}"
    )


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, object] = {"episodes": len(rows)}
    numeric_keys = {
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float)) and key not in {"episode", "seed"}
    }
    for key in sorted(numeric_keys):
        values = [float(row[key]) for row in rows if key in row]
        summary[f"{key}_mean"] = statistics.mean(values)
        summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[f"{key}_min"] = min(values)
        summary[f"{key}_max"] = max(values)
    summary["instance_ids"] = [row["selected_instance_id"] for row in rows]
    return summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    if args.is_learning and not args.is_train:
        raise ValueError("Learning is only valid with --is-train true")
    config = load_config(args.sim_config, algorithm_path=args.algorithm_config, dispatcher=args.algorithm)
    config.setdefault("instance", {}).update({
        "root": args.instance_root,
        "city": args.city,
        "split": "train" if args.is_train else "test",
    })
    if args.scenario_city is not None:
        # A registered city carries its own frozen fleet and battery ladder
        # (data/dataset/order_instance/city_instances.json), so those flags are
        # only required for a city the registry does not know.
        if args.scenario_city not in known_cities():
            for required in ("scenario_drones_per_launchpad", "scenario_batteries_per_drone"):
                if getattr(args, required) is None:
                    raise ValueError(
                        f"--{required.replace('_', '-')} is required with "
                        f"--scenario-city {args.scenario_city} (not in the frozen registry)"
                    )
        scenario = config.setdefault("scenario", {})
        scenario.update({"enabled": True, "city": args.scenario_city})
        for flag, key in [
            ("scenario_drones_per_launchpad", "drones_per_launchpad"),
            ("scenario_batteries_per_drone", "batteries_per_drone"),
            ("scenario_battery_types", "n_battery_types"),
        ]:
            value = getattr(args, flag)
            if value is not None:
                scenario[key] = int(value)
        # the split follows the run phase, exactly as the instance path's does;
        # the per-phase helpers below override it for valid/eval
        scenario["split"] = "train" if args.is_train else "test"
        for flag, key in [
            ("scenario_pad_spacing_km", "pad_spacing_km"),
            ("scenario_extent", "extent"),
            ("scenario_window", "window"),
            ("scenario_drone_placement", "drone_placement"),
            ("scenario_min_drones_per_pad", "min_drones_per_pad"),
        ]:
            value = getattr(args, flag)
            if value is not None:
                scenario[key] = value
    config.setdefault("simulation", {})["time_advancement"] = args.time_advance_mode
    if args.delta_t_seconds is not None:
        if args.time_advance_mode != "fixed_step":
            raise ValueError("--delta-t-seconds is only valid with --time-advance-mode fixed_step")
        config["simulation"]["delta_t_seconds"] = int(args.delta_t_seconds)
    preset = apply_algorithm_preset(config, args.algorithm)
    if args.urgent_order_multiplier is not None and args.urgent_order_multiplier <= 0:
        raise ValueError("--urgent-order-multiplier must be positive")
    if args.return_corridor_km is not None and args.return_corridor_km < 0:
        raise ValueError("--return-corridor-km must be non-negative")
    decision = config.setdefault("drone_decision", {})
    if args.candidate_pruning is not None:
        decision["candidate_pruning"] = args.candidate_pruning
    if args.feature_preset is not None:
        decision["feature_preset"] = args.feature_preset
    if args.feature_drop is not None:
        decision["feature_drop"] = list(args.feature_drop)
    if args.urgent_order_multiplier is not None:
        decision["urgent_order_multiplier"] = args.urgent_order_multiplier
    if args.return_corridor_km is not None:
        decision["return_corridor_km"] = args.return_corridor_km
    if args.critic_type is not None:
        decision["critic_type"] = args.critic_type
    if args.return_feature_mode is not None:
        decision["return_feature_mode"] = args.return_feature_mode
    if args.coord_frequencies is not None:
        if args.coord_frequencies < 0:
            raise ValueError("--coord-frequencies must be >= 0")
        decision["coord_frequencies"] = args.coord_frequencies
    if args.interaction_rank is not None:
        if args.interaction_rank < 0:
            raise ValueError("--interaction-rank must be >= 0")
        decision["interaction_rank"] = args.interaction_rank
    if preset is not None:
        overrides = {
            "--order-battery-policy": args.order_battery_policy,
            "--drone-return-policy": args.drone_return_policy,
            "--drone-rebalance": args.drone_rebalance,
        }
        selected_overrides = [name for name, value in overrides.items() if value is not None]
        if selected_overrides:
            raise ValueError(
                f"Algorithm {args.algorithm} is a fixed preset and cannot be combined with: "
                + ", ".join(selected_overrides)
            )
        if args.is_learning and not preset.trainable:
            raise ValueError(f"Algorithm {args.algorithm} is a non-learning baseline")
        if args.model_path and not preset.trainable:
            raise ValueError(f"Algorithm {args.algorithm} has no trainable model to load")
        if args.save_model_path and not preset.trainable:
            raise ValueError(f"Algorithm {args.algorithm} has no trainable model to save")
    else:
        decision = config.setdefault("drone_decision", {})
        if args.order_battery_policy is not None:
            decision["order_battery_policy"] = args.order_battery_policy
        if args.drone_return_policy is not None:
            decision["drone_return_policy"] = args.drone_return_policy
        if args.drone_rebalance is not None:
            decision.setdefault("rebalance", {})["enabled"] = args.drone_rebalance
    if (args.freeze_order_battery or args.freeze_drone_return) and not args.model_path:
        raise ValueError("freezing a policy requires --model-path")
    config.setdefault("rl", {})["training"] = bool(args.is_learning)
    config.setdefault("_runtime", {})["eval_action_mode"] = args.eval_action_mode
    config["_runtime"]["freeze_order_battery"] = bool(args.freeze_order_battery)
    config["_runtime"]["freeze_drone_return"] = bool(args.freeze_drone_return)
    if args.launchpad_capacity is not None:
        config.setdefault("simulation", {})["launchpad_capacity"] = args.launchpad_capacity
    config.setdefault("output", {})["record_replay"] = bool(args.record_replay)

    selection = args.instance_selection or ("random_saved" if args.is_train else "all_saved")
    split = "train" if args.is_train else "test"
    plan = instance_plan(
        root=args.instance_root,
        city=args.city,
        split=split,
        selection=selection,
        episodes=args.episodes,
        seed=args.seed,
        fixed_id=args.fixed_instance_id,
        scenario_city=args.scenario_city,
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else explicit_run_output_dir(
            records_root=args.records_root,
            split=split,
            city=args.city,
            drone_mode="drone_only",
            time_advance_mode=args.time_advance_mode,
            algorithm=dispatcher_label(config),
            seed=args.seed,
            # the run name carries suffixes (e.g. critic type) that the timestamp+seed
            # folder does not; without it concurrent same-seed variants overwrite each other
            run_stamp=f"{timestamp()}_{slug(args.wandb_name)}" if args.wandb_name else None,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb = init_wandb(
        args,
        config=config,
        output_dir=output_dir,
        job_type="train" if args.is_train else "test",
        default_name=dispatcher_label(config),
    )
    env = SimulatorEnv(config)
    dispatcher = build_dispatcher(args.algorithm, config, env, args.seed)
    if args.model_path:
        dispatcher.load(args.model_path)

    rows: list[dict] = []
    best_valid_score: float | None = None
    best_valid_episode: int | None = None
    if args.save_model_path:
        final_model_base = Path(args.save_model_path)
        best_model_path = str(
            final_model_base.parent / f"{final_model_base.stem}_best"
        )
    else:
        best_model_path = str(output_dir / "models" / "best")
    for item in plan:
        row = run_episode(
            env=env,
            dispatcher=dispatcher,
            config=config,
            item=item,
            split=split,
            learning=bool(args.is_learning),
            evaluation_action_mode=(
                "greedy" if args.is_learning else args.eval_action_mode
            ),
            record_dir=(output_dir / "last_replay") if args.record_replay else None,
        )
        rows.append(row)
        print(
            f"[{split}] ep={len(rows) - 1} seed={row.get('seed')} "
            f"orders={int(row.get('orders_generated', 0))} "
            f"deliv={int(row.get('orders_delivered', 0))} "
            f"completion={float(row.get('completion_rate', 0.0)):.3f} "
            f"on_time_gmv={float(row.get('on_time_gmv', 0.0)):.0f} "
            f"instance={row.get('selected_instance_id', '')}",
            flush=True,
        )
        if args.is_train:
            wandb_payload = episode_wandb_payload(row, namespace="train")
        else:
            wandb_payload = evaluation_episode_wandb_payload(row)
            wandb_payload["eval/instance"] = len(rows)
        valid_best_for_summary: float | None = None

        if args.is_train and args.valid_every > 0 and len(rows) % args.valid_every == 0:
            valid_plan = instance_plan(
                root=args.instance_root,
                city=args.city,
                split="valid",
                selection=args.valid_instance_selection,
                episodes=args.valid_episodes,
                seed=args.seed,
                fixed_id=args.valid_fixed_instance_id,
                scenario_city=args.scenario_city,
            )
            valid_rows = [
                run_episode(
                    env=env,
                    dispatcher=dispatcher,
                    config=config,
                    item=valid_item,
                    split="valid",
                    learning=False,
                    evaluation_action_mode="greedy",
                    record_dir=None,
                )
                for valid_item in valid_plan
            ]
            valid_summary = summarize(valid_rows)
            valid_score = float(valid_summary.get("on_time_gmv_mean", 0.0))
            if (
                args.is_learning
                and (best_valid_score is None or valid_score > best_valid_score)
            ):
                best_valid_score = valid_score
                best_valid_episode = len(rows)
                dispatcher.save(best_model_path)
            valid_payload = validation_wandb_payload(valid_summary)
            wandb_payload.update(valid_payload)
            valid_best_for_summary = (
                best_valid_score if best_valid_score is not None else valid_score
            )
        wandb.log(wandb_payload, step=len(rows))
        if valid_best_for_summary is not None:
            wandb.summary_update({"valid/on_time_gmv": valid_best_for_summary})

    primary_summary = summarize(rows)
    primary_namespace = "train" if args.is_train else "eval"
    wandb.summary_update(summary_wandb_payload(primary_summary, namespace=primary_namespace))
    if not args.is_train:
        wandb.summary_update(evaluation_wandb_payload(primary_summary))
    if args.save_model_path:
        dispatcher.save(args.save_model_path)
    if args.eval_episodes > 0:
        if best_valid_episode is not None:
            dispatcher.load(best_model_path)
            print(
                f"[eval] loaded best validation checkpoint from episode "
                f"{best_valid_episode} (on_time_gmv={best_valid_score:.0f})",
                flush=True,
            )
        eval_plan = instance_plan(
            root=args.instance_root,
            city=args.city,
            split="test",
            selection=args.eval_instance_selection,
            episodes=args.eval_episodes,
            seed=args.seed,
            fixed_id=args.eval_fixed_instance_id,
            scenario_city=args.scenario_city,
        )
        print(f"[eval] running {len(eval_plan)} held-out test episodes ...", flush=True)
        eval_rows: list[dict] = []
        for eval_index, eval_item in enumerate(eval_plan, start=1):
            eval_row = run_episode(
                env=env,
                dispatcher=dispatcher,
                config=config,
                item=eval_item,
                split="test",
                learning=False,
                evaluation_action_mode=args.eval_action_mode,
                record_dir=None,
            )
            eval_rows.append(eval_row)
            print(
                f"[eval] instance={eval_index}/{len(eval_plan)} "
                f"seed={eval_row.get('seed')} "
                f"orders={int(eval_row.get('orders_generated', 0))} "
                f"deliv={int(eval_row.get('orders_delivered', 0))} "
                f"completion={float(eval_row.get('completion_rate', 0.0)):.3f} "
                f"on_time_gmv={float(eval_row.get('on_time_gmv', 0.0)):.0f} "
                f"id={eval_row.get('selected_instance_id', '')}",
                flush=True,
            )
            eval_history_payload = evaluation_episode_wandb_payload(eval_row)
            eval_history_payload["eval/instance"] = eval_index
            wandb.log(eval_history_payload, step=len(rows) + eval_index)

        eval_summary = summarize(eval_rows)
        eval_payload = evaluation_wandb_payload(eval_summary)
        wandb.summary_update(eval_payload)
        print(
            f"[eval] on_time_gmv={eval_summary.get('on_time_gmv_mean', 0.0):.0f} "
            f"completion={eval_summary.get('completion_rate_mean', 0.0):.3f} "
            f"(mean over {len(eval_rows)} test instances)",
            flush=True,
        )
        with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(eval_summary, handle, indent=2)
    if best_valid_episode is not None:
        with (output_dir / "best_checkpoint.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "episode": best_valid_episode,
                "valid_on_time_gmv": best_valid_score,
                "model_path": best_model_path,
            }, handle, indent=2)

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(primary_summary, handle, ensure_ascii=False, indent=2)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        # `_runtime` is per-episode scratch the loader writes back (the scenario
        # report among it, which is a dataclass), not configuration -- it has no
        # place in a resolved-config record and is not serialisable anyway
        json.dump(
            {key: value for key, value in config.items() if key != "_runtime"},
            handle, ensure_ascii=False, indent=2,
        )
    wandb.finish()
    print(f"Done ({len(rows)} episodes). Records: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
