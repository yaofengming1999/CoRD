# CoRD

Research code for drone delivery and battery dispatch, including an event-driven
simulator, rule-based baselines, and PPO-based dispatch policies.

## Setup

Use Python 3.10+ and run commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy PyYAML torch
```

Weights & Biases is optional (`python -m pip install wandb`). The examples below
disable online logging. Learned policies use CUDA when available and fall back
to CPU otherwise.

**Current prerequisite:** this snapshot is missing
`data/instance_generator_synthetic/`, which the runner imports for
`scenario.order_revenue` and `run_generator.SCENARIO_DEFAULT`. Restore that package
and its dependencies from the original project before running the commands below.
Without it, even `--help` raises `ModuleNotFoundError`.

## Run

The main entry point is `runs/run_experiment.py`. The examples use the saved
Meituan instances under `data/instances_drone/meituan/`.

Run one test episode with the rule-based baseline:

```bash
python runs/run_experiment.py \
  --is-train false --is-learning false \
  --city meituan --instance-root data/instances_drone \
  --time-advance-mode event_driven \
  --algorithm R1_urgentLower_nearest \
  --instance-selection random_saved --episodes 1 \
  --wandb-mode disabled
```

Train the separable policy with periodic validation and a final test pass:

```bash
python runs/run_experiment.py \
  --is-train true --is-learning true \
  --city meituan --instance-root data/instances_drone \
  --time-advance-mode event_driven \
  --algorithm F2_Separable \
  --instance-selection random_saved --episodes 100 \
  --valid-every 10 --valid-episodes 2 \
  --valid-instance-selection random_saved \
  --eval-episodes 5 --eval-instance-selection random_saved \
  --save-model-path records/meituan_f2/model \
  --wandb-mode disabled
```

Evaluate the saved final model:

```bash
python runs/run_experiment.py \
  --is-train false --is-learning false \
  --city meituan --instance-root data/instances_drone \
  --time-advance-mode event_driven \
  --algorithm F2_Separable \
  --model-path records/meituan_f2/model \
  --instance-selection random_saved --episodes 5 \
  --wandb-mode disabled
```

`--is-train` selects the `train` or `test` split; `--is-learning` controls policy
updates. Validation uses the `valid` split. Change `--algorithm` to run other
presets, such as `F3_FullCombination`, `F1_StagedOLB`, or `AS1_MAPPO`.
Use `--instance-selection all_saved` to evaluate the entire split; this mode
ignores `--episodes`.
See `algorithm/dispatch/presets.py` for all preset names and
`python runs/run_experiment.py --help` for available arguments.

Results are saved under `records/` by default, including `history.json`,
`summary.json`, and `resolved_config.json`. Use `--output-dir` to choose an exact
output directory. Training with validation saves the best checkpoint; the final
test pass uses it when available. `--save-model-path` saves the final model, and
`--model-path` takes the same base path when loading it.

## File structure

```text
algorithm/
  F2/                 Separable actor, critic, candidate menus, and PPO learner
  F2pro/              Three-factor policy variant
  common/             Shared features, metrics, and learning utilities
  dispatch/           Algorithm presets and dispatch/return coordination
  AS*.py, F*.py, L*.py, R_rule_based.py
                      Policy variants and baselines
simulator/
  core/               Environment, state, transitions, and time advancement
  io/                 Instance loading and scenario construction
  model/              Orders, drones, batteries, and status definitions
  routing/            Flight routing and energy models
  metrics/            Episode metrics and recording
  world/, utils/      Airspace, weather, configuration, and geometry
runs/
  run_experiment.py   Training and evaluation entry point
  utils/              Output paths, metrics, and optional W&B logging
  *.py                Result collection and analysis scripts
yamls/
  algorithm/          Policy and learning settings
  simulator/          Simulator configuration overrides
data/instances_drone/
  meituan/            Base dataset: train/, valid/, test/
  meituan*/           Dataset variants and relative symlink aliases
records/              Local experiment outputs (created at runtime, ignored)
```

Each saved instance contains order, drone, launchpad, battery, and weather YAML
files. Some variants share files through relative symlinks, so keep the dataset
directory layout intact. The default runtime physics are loaded through
`simulator/utils/config.py`; an instance's `simulator_config.yaml` is a provenance
snapshot and is not automatically used as the runtime configuration.
