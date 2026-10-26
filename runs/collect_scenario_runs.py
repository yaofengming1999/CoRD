#!/usr/bin/env python3
"""Collect the scenario-path runs of 2026-10-11 into one comparison table.

Reads each run's records directory (history.json + summary.json) and reports the
final evaluation alongside the training curve, so a run that is still going is
summarised by what it has rather than skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = ["on_time_gmv", "orders_delivered", "completion", "expired_rate"]


def newest_run(root: Path, algorithm: str) -> Path | None:
    matches = sorted(root.rglob(f"*/{algorithm}/*/history.json"))
    return matches[-1].parent if matches else None


def describe(run_dir: Path) -> dict:
    history = json.loads((run_dir / "history.json").read_text())
    train = [r for r in history if r.get("split") == "train"]
    evaluation = [r for r in history if r.get("split") in ("test", "eval")]
    out = {"run": run_dir.name, "episodes": len(train), "eval_episodes": len(evaluation)}
    if train:
        tail = train[-20:]
        for key in METRICS:
            values = [float(r[key]) for r in tail if r.get(key) is not None]
            out[f"train_{key}"] = round(float(np.mean(values)), 4) if values else None
    for key in METRICS:
        values = [float(r[key]) for r in evaluation if r.get(key) is not None]
        out[f"eval_{key}"] = round(float(np.mean(values)), 4) if values else None
    summary = run_dir / "summary.json"
    if summary.is_file():
        out["summary"] = json.loads(summary.read_text()).get("on_time_gmv")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-root", default="records_scenario")
    parser.add_argument("--algorithms", nargs="*", default=[
        "F2_Separable", "L2_MSRA", "AS4_SyncOwnReward", "R5_hungarianGmv_backlogHeu",
    ])
    args = parser.parse_args()
    root = Path(args.records_root)
    if not root.is_dir():
        raise SystemExit(f"no records under {root}")

    print(f"{'run':52s}{'eps':>5s}{'train gmv':>11s}{'train compl':>12s}"
          f"{'eval gmv':>10s}{'eval compl':>11s}{'eval exp':>9s}")
    for algorithm in args.algorithms:
        for path in sorted(root.rglob(f"*/{algorithm}/*/history.json")):
            row = describe(path.parent)
            print(f"{row['run'][:52]:52s}{row['episodes']:5d}"
                  f"{row.get('train_on_time_gmv') or 0:11.0f}"
                  f"{row.get('train_completion') or 0:12.3f}"
                  f"{row.get('eval_on_time_gmv') or 0:10.0f}"
                  f"{row.get('eval_completion') or 0:11.3f}"
                  f"{row.get('eval_expired_rate') or 0:9.3f}")


if __name__ == "__main__":
    main()
