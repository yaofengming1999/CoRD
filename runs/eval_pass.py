"""One serial evaluation pass over saved checkpoints, for the metrics a training
sweep cannot produce.

Two of them need a run of their own:

  * the delivery-only battery margin. It did not exist when the formal sweep ran,
    so no record in ``records/`` carries it; the checkpoints do, because the
    quantity is a property of the decisions a policy makes, not of the training.
  * anything timed. The formal sweep ran eight jobs at once on one GPU, so every
    wall clock in it measures contention. This script runs one cell at a time in
    one process and reports the policy's own decision time separately from the
    simulator's.

Everything else -- instances, action mode, pruning, the checkpoint that is loaded
-- matches the ``[eval]`` phase of ``runs/run_experiment.py``, so on_time_gmv here
should reproduce the sweep's eval numbers.

    python runs/eval_pass.py --cities hongkong --algos F2_Separable R3_fifoLower_backlogHeu
    python runs/eval_pass.py --out records_scratch/eval_pass.csv      # everything
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithm.dispatch.presets import ALGORITHM_PRESETS, apply_algorithm_preset
from runs.utils.experiment_helpers import build_dispatcher
from simulator.core.simulator_env import SimulatorEnv
from simulator.metrics.compute import compute_episode_metrics
from simulator.utils.config import load_config

# The eleven cities of the 30-column re-run: grubhub and shenzhen joined it and
# meituan-ssa left it, so this list is not the 44-column sweep's.
CITIES = [
    "hongkong", "london", "lade_cq", "lade_sh", "grubhub", "shenzhen",
    "meituan-ssn", "meituan-sln", "meituan-san", "meituan-lsn", "meituan-asn",
]
ALGOS = [
    "R1_urgentLower_nearest", "R2_random_random", "R3_fifoLower_backlogHeu",
    "R4_fifoFull_backlogHeu", "R5_hungarianGmv_backlogHeu",
    "L1_RBG", "L2_MSRA", "L3_MAPDP", "L4_MoEPointer",
    "F0_TwoStage", "F1_StagedOLB", "F2_Separable", "F2pro", "F3_FullCombination",
    "AS1_MAPPO", "AS2_SingleDronePPO", "AS3_GlobalDronesPPO",
    "AS4_SyncOwnReward", "AS5_TeamReward",
]
GROUP_SUFFIX = "_fixed60_3kw_720wh"

# what the pass is for, on top of on_time_gmv
METRIC_KEYS = [
    "battery_avg_delivery_margin_ratio",
    "battery_avg_roundtrip_margin_ratio",
    "battery_avg_return_cost_ratio",
    "battery_avg_selected_soc",
    "return_avg_arrival_margin_ratio",
    "return_avg_extra_distance_vs_nearest_m",
]


def checkpoint_base(records_root: str, algo: str, city: str, seed: int,
                    run_suffix: str = "") -> Path | None:
    """The best-validation checkpoint the sweep saved for this cell.

    ``run_suffix`` is what the sweep appended to the run name after the seed --
    "_rank0" for the rank-0 runs, empty for the main ones. It has to be matched
    exactly: a glob loose enough to accept "_rank0" would also accept "_rank16",
    and the wrong checkpoint would be evaluated silently.
    """
    # Two layouts, because the sweeps that wrote them named their runs
    # differently. records_f30/ is flat and the run name is the launcher's tag:
    # "{algo}_{city}{run_suffix}_s{seed}". The older records/ grouped by algorithm
    # and led with the city. The flat one is tried first and both are exact, not
    # globbed loosely: a pattern permissive enough to accept "_r0" would also
    # accept "_r8", and the wrong checkpoint would load in silence.
    flat = (Path(records_root) / "models"
            / f"{algo}_{city}{run_suffix}_s{seed}_best.order_battery.pt")
    if flat.exists():
        return Path(str(flat)[: -len(".order_battery.pt")])
    group = Path(records_root) / "models" / f"{algo}{GROUP_SUFFIX}"
    hits = sorted(group.glob(f"{city}_{algo}_*_s{seed}{run_suffix}_best.order_battery.pt"))
    if not hits:
        return None
    return Path(str(hits[0])[: -len(".order_battery.pt")])


def instance_ids(root: str, city: str, limit: int | None = None) -> list[str]:
    """The city's test instances, in name order, optionally the first ``limit``.

    A limit is for the cost columns only. Row counts and decision times are
    properties of the policy and the instance and settle in a handful of
    episodes, while on_time_gmv does not: a truncated pass must not be compared
    against the sweep's 30-instance eval, and `n_instances` is written into every
    row so a reader can tell which kind of pass produced it.
    """
    split = Path(root) / city / "test"
    ids = sorted(path.name for path in split.iterdir() if path.is_dir())
    return ids if limit is None else ids[:limit]


def build(city: str, algo: str, seed: int, *, device: str, instance_root: str,
          interaction_rank: int | None = None, sim_config: str = "scenario_default",
          feature_preset: str | None = None, feature_drop: list[str] | None = None):
    """``sim_config`` is deep-merged over SCENARIO_DEFAULT, the same way
    run_experiment.py's --sim-config is, so an evaluation and a future training
    run read one scenario file rather than two copies of a constant."""
    config = load_config(sim_config, "drone_decision")
    apply_algorithm_preset(config, algo)
    config["drone_decision"]["candidate_pruning"] = False
    if interaction_rank is not None:
        # The network has to be built at the rank its checkpoint was trained at.
        # The learner refuses a mismatch rather than loading silently, so without
        # this a rank-0 checkpoint fails against the yaml default of 8 -- which is
        # the guard working, not a problem to route around.
        config["drone_decision"]["interaction_rank"] = int(interaction_rank)
    # Same reason as the rank: the layer widths are the ablation, so a checkpoint
    # must be evaluated under the feature set it was trained with.
    if feature_preset is not None:
        config["drone_decision"]["feature_preset"] = feature_preset
    if feature_drop is not None:
        config["drone_decision"]["feature_drop"] = list(feature_drop)
    for value in config["drone_decision"].values():
        if isinstance(value, dict) and "device" in value:
            value["device"] = device
    config.setdefault("rl", {})["training"] = False
    config.setdefault("instance", {}).update({
        "city": city, "root": instance_root, "selection": "fixed", "split": "test",
    })
    config.setdefault("simulation", {}).update({
        "time_advancement": "fixed_step", "delta_t_seconds": 60,
        "launchpad_capacity": "unlimited",
    })
    env = SimulatorEnv(config)
    dispatcher = build_dispatcher(algo, config, env, seed)
    return config, env, dispatcher


def evaluate_cell(city, algo, seed, *, device, instance_root, records_root, action_mode,
                  source_city=None, run_suffix="", interaction_rank=None,
                  sim_config="scenario_default", max_instances=None):
    """Evaluate `algo` on `city`, loading the checkpoint trained on `source_city`.

    source_city defaults to city, which is the in-distribution pass. Pointing it at
    another scenario is the transfer pass: the weights never saw this city, and
    because the variant scenarios are symlink overlays on one base instance set, the
    order stream and the network are identical to the source -- only the perturbed
    resource differs. A rule-based method has no weights, so its number is the same
    either way and the source city is ignored.
    """
    preset = ALGORITHM_PRESETS[algo]
    config, env, dispatcher = build(
        city, algo, seed, device=device, instance_root=instance_root,
        interaction_rank=interaction_rank, sim_config=sim_config,
    )
    if preset.trainable:
        base = checkpoint_base(records_root, algo, source_city or city, seed, run_suffix)
        if base is None:
            return None, f"no checkpoint for {source_city or city}"
        dispatcher.load(str(base))
    for policy in (dispatcher.order_battery_policy, dispatcher.drone_return_policy):
        policy.training = False
        learner = getattr(policy, "learner", None)
        if learner is not None and hasattr(learner, "set_evaluation_action_mode"):
            learner.set_evaluation_action_mode(action_mode)

    rows = []
    ids = instance_ids(instance_root, city, max_instances)
    for index, instance_id in enumerate(ids):
        config["instance"]["fixed_id"] = instance_id
        state = env.reset(seed=seed + index)
        policy_time = env_time = 0.0
        decisions = 0
        energy_at_start = int(getattr(env.air_router, "task_energy_calls", 0))
        started = perf_counter()
        done = False
        while not done:
            mark = perf_counter()
            actions = dispatcher.plan(state)
            policy_time += perf_counter() - mark
            decisions += len(actions)
            mark = perf_counter()
            env.apply_actions(actions)
            state, _, done, _ = env.step()
            env_time += perf_counter() - mark
        wall = perf_counter() - started
        metrics = compute_episode_metrics(
            env.orders, env.drones, env.horizon, config, env.launchpads, env.batteries,
        )
        operational = env.metrics_recorder.operational_summary()
        stats = dispatcher.finish_episode(0.0, reward_events=[], episode_end_time=env.current_time)
        row = {
            "city": city, "algorithm": algo, "seed": seed, "instance": instance_id,
            "n_instances": len(ids),
            # the scenario is part of the cell's identity: two passes over the same
            # checkpoint under different reserves must not pool into one row set
            "sim_config": sim_config,
            "reserve_ratio": float(env.air_router.energy_model.params.reserve_ratio),
            "on_time_gmv": float(metrics.get("on_time_gmv", 0.0)),
            # The two rates the tables report alongside the value. They were
            # missing, so the transfer table could fill only its GMV block:
            # orders_delivered is a count, and a count cannot be compared across
            # cities that generate different numbers of orders.
            "on_time_order_rate": float(metrics.get("on_time_order_rate", 0.0)),
            "completion_rate": float(metrics.get("completion_rate", 0.0)),
            "orders_delivered": float(metrics.get("orders_delivered", 0.0)),
            "orders_generated": float(metrics.get("total_orders_generated",
                                                 metrics.get("orders_generated", 0.0))),
            "wall_time_s": wall,
            "policy_time_s": policy_time,
            "env_time_s": env_time,
            "actions": float(decisions),
            # The two sizes table 6 needs. "candidates" is the menu the decision
            # faces: one entry per feasible (order, return pad, battery) triple.
            # "feature_rows" is what the actor was actually given -- the same
            # number for a row-based method, but N_ol + N_b for the factored one,
            # which never builds the product. Their ratio is what the factored
            # menu saves, and it is why a decision time has to be read next to
            # the size it was measured at.
            "candidates": float(stats.get("order_battery_candidates", 0.0)),
            "decision_groups": float(stats.get("order_battery_decision_groups", 0.0)),
            "candidates_per_group": float(stats.get("order_battery_candidates_per_group", 0.0)),
            "feature_rows": float(stats.get("order_battery_feature_rows", 0.0)),
            "policy_decisions": float(stats.get("order_battery_decisions", 0.0)),
            "energy_calls": float(
                int(getattr(env.air_router, "task_energy_calls", 0)) - energy_at_start
            ),
        }
        row.update({key: float(operational.get(key, 0.0)) for key in METRIC_KEYS})
        rows.append(row)
    return rows, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=CITIES)
    parser.add_argument("--algos", nargs="+", default=ALGOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-mode", default="greedy")
    parser.add_argument("--instance-root", default="data/instances_drone")
    parser.add_argument("--records-root", default="records")
    parser.add_argument(
        "--interaction-rank", type=int, default=None,
        help="build the actor at this rank; must match the rank its checkpoint was "
             "trained at, which the learner checks",
    )
    parser.add_argument(
        "--run-suffix", default="",
        help="what the sweep appended to the run name after the seed, e.g. _rank0; "
             "selects which batch of checkpoints to evaluate",
    )
    parser.add_argument(
        "--source-city", default=None,
        help="load the checkpoint trained on this city instead of the one being "
             "evaluated; the zero-shot transfer pass of table 5",
    )
    parser.add_argument(
        "--sim-config", default="scenario_default",
        help="simulator scenario deep-merged over SCENARIO_DEFAULT, e.g. lowreserve",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="evaluate only the first N test instances. For the cost columns, "
             "which settle in a few episodes; on_time_gmv from a truncated pass "
             "is not the sweep's eval number and must not be pooled with it",
    )
    parser.add_argument("--out", default="records_scratch/eval_pass.csv")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    everything: list[dict] = []
    for city in args.cities:
        for algo in args.algos:
            preset = ALGORITHM_PRESETS[algo]
            seeds = args.seeds if preset.trainable else args.seeds[:1]
            for seed in seeds:
                mark = perf_counter()
                rows, skipped = evaluate_cell(
                    city, algo, seed,
                    device=args.device, instance_root=args.instance_root,
                    records_root=args.records_root, action_mode=args.action_mode,
                    source_city=args.source_city, run_suffix=args.run_suffix,
                    interaction_rank=args.interaction_rank,
                    sim_config=args.sim_config,
                    max_instances=args.max_instances,
                )
                if skipped is not None:
                    print(f"skip  {city:13s} {algo:26s} s{seed}  ({skipped})", flush=True)
                    continue
                for row in rows:
                    row["source_city"] = args.source_city or city
                everything.extend(rows)
                gmv = np.mean([row["on_time_gmv"] for row in rows])
                soc = np.mean([row["battery_avg_selected_soc"] for row in rows])
                delivery = np.mean([row["battery_avg_delivery_margin_ratio"] for row in rows])
                roundtrip = np.mean([row["battery_avg_roundtrip_margin_ratio"] for row in rows])
                cost = np.mean([row["battery_avg_return_cost_ratio"] for row in rows])
                per_action = 1000.0 * sum(r["policy_time_s"] for r in rows) / max(
                    1.0, sum(r["actions"] for r in rows)
                )
                print(
                    f"done  {city:13s} {algo:26s} s{seed}  gmv {gmv:8.0f}  soc {soc:.3f}  "
                    f"delivery {delivery:.3f}  roundtrip {roundtrip:.3f}  return_cost {cost:.3f}  "
                    f"policy {per_action:6.2f} ms/action  ({perf_counter() - mark:5.0f}s)",
                    flush=True,
                )
                with out.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(everything[0]))
                    writer.writeheader()
                    writer.writerows(everything)
    print(f"\n{len(everything)} episodes -> {out}")


if __name__ == "__main__":
    main()
