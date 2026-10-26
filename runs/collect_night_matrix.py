#!/usr/bin/env python3
"""Collect the overnight city x algorithm matrix from the run logs.

A learned run ends with an ``[eval] on_time_gmv=... completion=...`` line over
the held-out test days; a rule baseline has no training phase and reports one
``[test]`` line per test day instead. Both are reduced here to the same pair, so
the table does not quietly mix a training number with an evaluation one.

    python -m runs.collect_night_matrix --logs <dir> [--out mdtable/night_matrix.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

EVAL = re.compile(r"^\[eval\] on_time_gmv=([0-9.]+) completion=([0-9.]+)")
TEST = re.compile(r"^\[test\].*completion=([0-9.]+) on_time_gmv=([0-9.]+)")
TRAIN = re.compile(r"^\[train\] ep=(\d+).*completion=([0-9.]+) on_time_gmv=([0-9.]+)")
CITIES = ["grubhub", "meituanA", "meituanB", "meituanC", "shenzhen"]
ARMS = ["R5_hungarian", "MSRA", "F2_rank0", "F2_rank8"]


def read(path: Path) -> dict | None:
    gmv = comp = None
    tests: list[tuple[float, float]] = []
    train: list[tuple[int, float, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        m = EVAL.match(line)
        if m:
            gmv, comp = float(m.group(1)), float(m.group(2))
            continue
        m = TEST.match(line)
        if m:
            tests.append((float(m.group(2)), float(m.group(1))))
            continue
        m = TRAIN.match(line)
        if m:
            train.append((int(m.group(1)), float(m.group(3)), float(m.group(2))))
    if gmv is None and tests:                    # rule baseline: mean over test days
        gmv = statistics.fmean(g for g, _ in tests)
        comp = statistics.fmean(c for _, c in tests)
    if gmv is None:
        return None
    last10 = train[-10:]
    return {
        "eval_gmv": gmv,
        "eval_completion": comp,
        "train_episodes": len(train),
        "train_gmv_last10": statistics.fmean(g for _, g, _ in last10) if last10 else "",
        "train_completion_last10": statistics.fmean(c for _, _, c in last10) if last10 else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("mdtable/night_matrix.csv"))
    args = ap.parse_args()

    rows = []
    for path in sorted(args.logs.glob("*.log")):
        stem = path.stem
        city = next((c for c in CITIES if stem.startswith(c)), None)
        if city is None:
            continue
        rest = stem[len(city) + 1 :]
        seed = ""
        if "_s" in rest and rest.rsplit("_s", 1)[1].isdigit():
            rest, seed = rest.rsplit("_s", 1)
        got = read(path)
        if got is None:
            rows.append({"city": city, "arm": rest, "seed": seed, "status": "unfinished"})
            continue
        rows.append({"city": city, "arm": rest, "seed": seed, "status": "ok", **got})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["city", "arm", "seed", "status", "eval_gmv", "eval_completion",
              "train_episodes", "train_gmv_last10", "train_completion_last10"]
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    done = [r for r in rows if r["status"] == "ok"]
    print(f"{len(done)}/{len(rows)} finished -> {args.out}\n")
    width = max((len(a) for a in ARMS), default=10)
    for city in CITIES:
        here = [r for r in done if r["city"] == city]
        if not here:
            continue
        print(f"--- {city}")
        for arm in ARMS:
            vals = [r for r in here if r["arm"] == arm]
            if not vals:
                continue
            gmvs = [r["eval_gmv"] for r in vals]
            comps = [r["eval_completion"] for r in vals]
            spread = f" (n={len(gmvs)}, {min(gmvs):.0f}-{max(gmvs):.0f})" if len(gmvs) > 1 else ""
            print(f"  {arm:<{width}}  gmv {statistics.fmean(gmvs):8.0f}  "
                  f"completion {statistics.fmean(comps):.3f}{spread}")
        print()


if __name__ == "__main__":
    main()
