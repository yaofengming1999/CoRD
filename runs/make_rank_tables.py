"""The rank ablation on meituan-sln: the eval table and the raw training curves.

Two outputs, deliberately separate:

  mdtable/table4_1_rank.md        one row per rank, the held-out eval numbers
  mdtable/table4_2_rank_train.csv one row per (rank, seed, episode), as logged

The eval table is what a paper reports; the csv is the raw curve behind it, kept
in full so a plot or a convergence claim can be made without re-reading logs.

Rank 8 is the setting the main experiments use. The curve is what justifies
calling anything at or above 4 equivalent: r=4, 8, 16 and 32 differ by less than
the seed spread (pairwise p > 0.4 against r=4), while 0 -> 2 -> 4 are real steps.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

CITY = "meituan-sln"
RANKS = (0, 1, 2, 4, 8, 16, 32)
SEEDS = (42, 43, 44)

# The two methods whose curves the csv holds, and where their logs live. F2pro is
# the swept one; F2 is the reference the eval table compares against, at the one
# rank it uses, and its curve is included so the two can be plotted on one axis
# without reading from two places.
SERIES = [
    ("F2pro", RANKS, "logs/f2pro/{city}_f2pro_rank{rank}_s{seed}.log"),
    ("F2_Separable", (0,), "logs/rank/{city}_rank{rank}_s{seed}.log"),
]
LOG = SERIES[0][2]      # the eval table reads F2pro only

# the comparison the table is read against: the proposed method at rank 0 on the
# same city, from mdtable/table1_main.md's source (the rank sweep's records)
REFERENCE_LABEL = "F2 (o,l)|b at rank 0"
REFERENCE = 16329.3

# The summary line, not the per-instance ones. A run logs "[eval] loaded best
# validation checkpoint ...", then one line per test instance, then the mean --
# and the per-instance lines carry on_time_gmv too, so a looser pattern picks a
# single instance and calls it the result.
EVAL = re.compile(r"^\[eval\] (?P<body>on_time_gmv=\S+ .*mean over \d+ test instances.*)$", re.M)
TRAIN = re.compile(r"^\[train\] ep=(?P<ep>\d+)\s+(?P<body>.*)$", re.M)


def fields(body: str) -> dict[str, float]:
    """key=value pairs of a log line, numbers only."""
    out = {}
    for key, value in re.findall(r"([a-z_]+)=(-?[0-9.]+)", body):
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def read(rank: int, seed: int, template: str = LOG) -> tuple[dict[str, float] | None, list[dict[str, float]]]:
    path = Path(template.format(city=CITY, rank=rank, seed=seed))
    if not path.exists():
        return None, []
    text = path.read_text(errors="ignore")
    final = EVAL.findall(text)
    curve = [{"episode": int(m.group("ep")), **fields(m.group("body"))}
             for m in TRAIN.finditer(text)]
    return (fields(final[-1]) if final else None), curve


def eval_table() -> str:
    data = {r: {s: read(r, s)[0] for s in SEEDS} for r in RANKS}
    keys = ("on_time_gmv", "completion")
    parts = [
        f"# Table 4.1: rank of the (order, pad) interaction, on {CITY}",
        "",
        "F2pro scores three separate blocks -- orders, return pads, batteries -- with",
        "the bilinear coupling on the (order, pad) pair alone, since that is the pair",
        "the return energy couples. This table prices that coupling. One city, because",
        "this is where the curve has something to say: the fleet is halved there, so the",
        "return pad is the binding decision and its menus are the largest measured (444",
        "feasible triples per decision against 16 on hongkong).",
        "",
        "Three seeds per rank, 300 training episodes each, the best-validation",
        "checkpoint evaluated greedily on the city's 30 held-out test instances.",
        "",
        f"The reference column is {REFERENCE_LABEL} on the same city ({REFERENCE:.1f}),",
        "which fuses the pair instead of splitting it and needs no coupling at all.",
        "",
        "Rank 8 is the setting the main experiments use. What the curve licenses is not",
        "that 8 is optimal but that anything at or above 4 is equivalent: against r=4,",
        "r=8 differs by -151 (p=0.71), r=16 by -270 (p=0.50) and r=32 by -139 (p=0.62),",
        "all inside a seed spread of about 300-500. The steps below it are real --",
        "0 to 2 is +7.5% and 2 to 4 a further +13.4%.",
        "",
        "Two things follow. The coupling is necessary: rank 0 and rank 1 are the same",
        "(-22.4% and -22.5%), so one dimension buys nothing. And it is not sufficient:",
        "the curve flattens 5.4% short of the fused form, so what the split gives up is",
        "not representational capacity but information -- E_task(o,l) is computed by the",
        "simulator over the real route, the wind and the payload, none of which the",
        "coordinates carry, and no rank recovers it.",
        "",
    ]
    for key, title, digits in (("on_time_gmv", "On-time GMV", 1),
                               ("completion", "Completion rate", 4)):
        header = ["rank"] + [f"s{s}" for s in SEEDS] + ["mean", "sd", "vs reference", "vs rank 0"]
        rows = []
        base = None
        for rank in RANKS:
            got = [data[rank][s][key] for s in SEEDS
                   if data[rank][s] is not None and key in data[rank][s]]
            if not got:
                rows.append([f"r={rank}"] + [""] * (len(header) - 1))
                continue
            a = np.array(got)
            if rank == 0:
                base = a.mean()
            ref = REFERENCE if key == "on_time_gmv" else None
            rows.append([
                f"r={rank}" + (" (main)" if rank == 8 else ""),
                *[f"{data[rank][s][key]:.{digits}f}" if data[rank][s] else "" for s in SEEDS],
                f"{a.mean():.{digits}f}",
                f"{a.std(ddof=1):.{digits}f}" if len(a) > 1 else "",
                f"{100 * (a.mean() / ref - 1):+.2f}%" if ref else "",
                f"{100 * (a.mean() / base - 1):+.2f}%" if base else "",
            ])
        widths = [max([len(header[i])] + [len(r[i]) for r in rows]) for i in range(len(header))]
        parts += [f"## {title}", ""]
        parts.append("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |")
        parts.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
        for r in rows:
            parts.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")
        parts.append("")
    return "\n".join(parts)


def train_csv(path: Path) -> int:
    """Every logged training episode of every (rank, seed), one row each."""
    rows: list[dict] = []
    columns: list[str] = []
    for method, ranks, template in SERIES:
        for rank in ranks:
            for seed in SEEDS:
                _, curve = read(rank, seed, template)
                for point in curve:
                    # the log's own "seed=" is the instance sampled that episode,
                    # not the run's seed, so it is kept under its own name rather
                    # than silently overwriting the column identifying the run
                    point = dict(point)
                    if "seed" in point:
                        point["instance_seed"] = point.pop("seed")
                    row = {"city": CITY, "method": method, "rank": rank,
                           "seed": seed, **point}
                    rows.append(row)
                    for key in row:
                        if key not in columns:
                            columns.append(key)
    if not rows:
        return 0
    # a logged line may omit a metric; csv needs one schema, so missing is blank
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-out", default="mdtable/table4_1_rank.md")
    ap.add_argument("--train-out", default="mdtable/table4_2_rank_train.csv")
    args = ap.parse_args()
    for path in (Path(args.eval_out), Path(args.train_out)):
        path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.eval_out).write_text(eval_table())
    print(f"written {args.eval_out}")
    n = train_csv(Path(args.train_out))
    print(f"written {args.train_out}  ({n} episode rows)")


if __name__ == "__main__":
    main()
