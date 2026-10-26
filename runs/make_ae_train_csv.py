"""The raw training curves behind table 3, the action-expansion axis.

  mdtable/table3_action_expansion_train.csv   one row per (method, city, seed, episode)

Table 3 itself reports held-out eval; this is the curve underneath it, kept in
full so a convergence claim or a plot can be made without re-reading 120 logs.
The four methods share one log schema (ep, orders, deliv, completion,
on_time_gmv, instance), so the csv has one set of columns and no per-method
special cases.

Two things worth knowing before using the numbers:

  The `seed=` field of a training line is the INSTANCE seed, which cycles over
  the training set episode by episode -- it is not the run's seed. It is kept as
  `instance_seed`; the run seed comes from the file name and is `seed`.

  F2 and F0 are read at rank 0, the setting table 3 reports for both, so their
  curves come from the rank sweeps' logs rather than logs/formal. F1 and F3 have
  no rank at all, so their formal runs are their rank-0 runs.

--check cross-validates every log against the eval_summary.json that table 3
reads for that cell, so a wrong log directory cannot pass silently.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_mdtables import AE_METHODS, CITIES, RANK_ROOT, per_seed  # noqa: E402

SEEDS = (42, 43, 44)
OUT = Path("mdtable/table3_action_expansion_train.csv")

# where each row of table 3 trained, and which records the table reads for it.
# F2 and F0 are the rank-0 runs from their sweeps, which is what the table
# reports for both; F1 and F3 carry no rank, so their formal runs stand.
SOURCES = {
    "F3_FullCombination": ("logs/formal/{city}_F3_FullCombination_s{seed}.log", "records", ""),
    "F1_StagedOLB":       ("logs/formal/{city}_F1_StagedOLB_s{seed}.log", "records", ""),
    "F0_TwoStage":        ("logs/f0_rank0/{city}_F0_TwoStage_rank0_s{seed}.log", RANK_ROOT, "_rank0_"),
    "F2_Separable":       ("logs/rank/{city}_rank0_s{seed}.log", RANK_ROOT, "_rank0_"),
}

TRAIN = re.compile(r"^\[train\] ep=(?P<ep>\d+)\s+(?P<body>.*)$", re.M)
# the summary line, not the 30 per-instance ones -- those carry on_time_gmv too
EVAL = re.compile(r"^\[eval\] on_time_gmv=(?P<gmv>[0-9.]+).*mean over \d+ test instances", re.M)

NUMERIC = ("orders", "deliv", "completion", "on_time_gmv")
COLUMNS = ("method", "city", "seed", "ep", *NUMERIC, "instance_seed", "instance")


def fields(body: str) -> dict[str, str]:
    out = dict(re.findall(r"([a-z_]+)=(-?[0-9.]+)", body))
    found = re.search(r"instance=(\S+)", body)
    if found:
        out["instance"] = found.group(1)
    return out


def rows_of(path: Path, method: str, city: str, seed: int) -> list[dict]:
    text = path.read_text(errors="replace")
    rows = []
    for match in TRAIN.finditer(text):
        got = fields(match.group("body"))
        # the log's own seed= is the instance seed, cycling over the train set
        instance_seed = got.pop("seed", "")
        rows.append({
            "method": method, "city": city, "seed": seed, "ep": int(match.group("ep")),
            **{k: got.get(k, "") for k in NUMERIC},
            "instance_seed": instance_seed, "instance": got.get("instance", ""),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="cross-validate each log's eval mean against the records table 3 reads")
    args = ap.parse_args()

    rows, missing, mismatched = [], [], []
    for method, _label in AE_METHODS:
        pattern, root, suffix = SOURCES[method]
        # CITIES holds (label, instance city, records city); the logs are named
        # with the label and the records with the third field
        for city, _instance, rec in CITIES:
            logs = {s: Path(pattern.format(city=city, seed=s)) for s in SEEDS}
            for seed, path in logs.items():
                if not path.is_file():
                    missing.append(str(path))
                    continue
                got = rows_of(path, method, city, seed)
                if not got:
                    missing.append(f"{path} (no [train] lines)")
                rows.extend(got)
            if args.check:
                # the eval the records hold for this cell, against the logs' own
                got_rec = per_seed(rec, method, "on_time_gmv", root, suffix)
                # per_seed returns {seed: value}; the values are the comparison
                recorded = sorted(got_rec.values() if hasattr(got_rec, "values") else got_rec)
                logged = sorted(
                    float(m.group("gmv"))
                    for p in logs.values() if p.is_file()
                    for m in [EVAL.search(p.read_text(errors="replace"))] if m
                )
                pairs = list(zip(recorded, logged))
                off = [(a, b) for a, b in pairs if abs(a - b) > max(1.0, 0.005 * a)]
                if off or len(recorded) != len(logged):
                    mismatched.append(
                        f"{method:20s} {city:12s} records={[round(x,1) for x in recorded]} "
                        f"logs={[round(x,1) for x in logged]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    cells = {(r["method"], r["city"], r["seed"]) for r in rows}
    print(f"{OUT}: {len(rows)} rows, {len(cells)} cells "
          f"({len(AE_METHODS)} methods x {len(CITIES)} cities x {len(SEEDS)} seeds "
          f"= {len(AE_METHODS)*len(CITIES)*len(SEEDS)})")
    eps = {r["ep"] for r in rows}
    print(f"  episodes {min(eps)}..{max(eps)}, "
          f"{len(rows)//max(1,len(cells))} per cell on average")
    for name, bad in (("missing", missing), ("eval mismatch", mismatched)):
        if bad:
            print(f"  {len(bad)} {name}:")
            for line in bad[:12]:
                print(f"    {line}")
        elif name == "eval mismatch" and args.check:
            print("  every log's eval mean matches the records table 3 reads")


if __name__ == "__main__":
    main()
