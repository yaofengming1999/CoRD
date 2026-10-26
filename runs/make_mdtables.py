"""Dump the results as raw Markdown tables, one row per method, every seed listed.

No rounding decisions, no aggregation beyond what the runner already did (each
number is already a mean over that city's held-out test instances). Presentation
is left to whoever reads the file.

    python runs/make_mdtables.py            # writes mdtable/table1_main.md
    python runs/make_mdtables.py --check    # list missing cells and stop
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import yaml

try:
    from yaml import CSafeLoader as _Loader
except ImportError:
    from yaml import SafeLoader as _Loader

SEEDS = (42, 43, 44)

# (label, instance directory, records directory)
CITIES = [
    ("hongkong", "hongkong", "hongkong"),
    ("london", "london", "london"),
    ("lade_cq", "lade_cq", "lade_cq"),
    ("lade_sh", "lade_sh", "lade_sh"),
    ("meituan-ssn", "meituan-ssn", "meituan_ssn"),
    ("meituan-ssa", "meituan-ssa", "meituan_ssa"),
    ("meituan-sln", "meituan-sln", "meituan_sln"),
    ("meituan-san", "meituan-san", "meituan_san"),
    ("meituan-lsn", "meituan-lsn", "meituan_lsn"),
    ("meituan-asn", "meituan-asn", "meituan_asn"),
]

# meituan-peak carries four times the base order volume on the same network, so it
# is the stress column of table 1 rather than a table of its own. Cells for methods
# that have not run it yet are left blank.
CITIES_WITH_PEAK = CITIES + [("meituan-peak", "meituan-peak", "meituan_peak")]

# Table 2's cities: the five base scenarios, without the meituan variants. The
# variants move one resource of the base meituan set (drones or batteries +/-50%,
# adverse weather); they belong to table 1, where the question is how a method
# holds up as a resource tightens. The agent-structure axis is a property of the
# learner, so five independent scenarios answer it, and running it at rank 0
# across all ten would have cost 150 runs instead of 75.
AS_CITIES = [
    ("hongkong", "hongkong", "hongkong"),
    ("london", "london", "london"),
    ("lade_cq", "lade_cq", "lade_cq"),
    ("lade_sh", "lade_sh", "lade_sh"),
    ("meituan-ssn", "meituan-ssn", "meituan_ssn"),
]
# Every row of table 2 is trained with the actor's bilinear coupling removed
# (rank 0), which table 4 shows costs nothing, so the six rows differ only in the
# agent structure and none of them carries a term the others lack.
AS_RANK_SUFFIX = "_rank0_"

# Table 1: what the proposed method is compared against.
# Table 2: the agent-structure axis -- all four keep the proposed menu, features and
# separable actor, and differ only in how the decision stream is cut for PPO, so a
# difference between them is attributable to that and to nothing else.
AS_METHODS = [
    ("AS1_MAPPO", "AS1 MAPPO (synchronous, team reward)"),
    ("AS4_SyncOwnReward", "AS4 MAPPO (synchronous, own-order reward)"),
    ("AS2_SingleDronePPO", "AS2 SingleDronePPO (one trajectory per episode)"),
    ("AS3_GlobalDronesPPO", "AS3 GlobalDronesPPO (one joint transition per step)"),
    ("AS5_TeamReward", "AS5 SerializedMARL (sequential, team reward)"),
    ("F2_Separable", "F2 (proposed, sequential, own-order reward)"),
]

# Table 3: the action-expansion axis -- all four keep the proposed method's
# serialized credit assignment (one trajectory per drone, own-order rewards) and
# differ only in how one dispatch is formed and scored.
AE_METHODS = [
    ("F3_FullCombination", "F3 FullCombination (one row per triple)"),
    ("F1_StagedOLB", "F1 AutoregressiveOLB (staged o -> l -> b)"),
    ("F0_TwoStage", "F0 TwoStage (two separate learned agents)"),
    ("F2_Separable", "F2 (proposed, factored (o,l)|b, rank 0)"),
]


METHODS = [
    ("R1_urgentLower_nearest", "R1 urgent, nearest"),
    ("R2_random_random", "R2 random"),
    ("R3_fifoLower_backlogHeu", "R3 FIFO, lowest feasible"),
    ("R4_fifoFull_backlogHeu", "R4 FIFO, full charge"),
    ("R5_hungarianGmv_backlogHeu", "R5 Hungarian on-time GMV"),
    ("L1_RBG", "L1 RBG"),
    ("L2_MSRA", "L2 MSRA"),
    ("L3_MAPDP", "L3 MAPDP"),
    ("L4_MoEPointer", "L4 MoE-Pointer"),
    ("F2_Separable", "F2 (proposed, rank 0)"),
    ("F2pro", "F2pro (o | l | b, rank 4)"),
]

METRICS = [
    ("on_time_gmv", "On-time GMV", 1),
    ("on_time_order_rate", "On-time fulfilment rate  |T|/|G|", 4),
    ("completion_rate", "Completion rate  |D|/|G|", 4),
]


# Table 5: zero-shot transfer off meituan-ssn. The five variants are symlink
# overlays on the base meituan instances, so the order stream and the network are
# identical to the source and exactly one resource moves; peak is a different
# generator window (4x volume), so it is a shift in demand rather than in supply.
TRANSFER_SOURCE = "meituan-ssn"
TRANSFER_TARGETS = [
    ("meituan-ssa", "meituan_ssa", "adverse weather, 8 m/s wind"),
    ("meituan-sln", "meituan_sln", "drones -50%"),
    ("meituan-san", "meituan_san", "drones +50%"),
    ("meituan-lsn", "meituan_lsn", "batteries -50%"),
    ("meituan-asn", "meituan_asn", "batteries +50%"),
    ("meituan-peak", "meituan_peak", "4x order volume"),
]
# Table 5's rows. Three methods, and only the transfer column is comparable
# across them.
#
# Why there is no "trained on target" column. L2 and F2 were trained on every
# city, so a locally trained reference exists for them; AS4 was trained on the
# five base scenarios only (table 2's set), so on the meituan variants it has
# none. A column that is populated for two rows and empty for the third invites
# exactly the comparison it cannot support, so the table reports what every row
# has: the score after transfer, and the gap to the proposed method.
#
# R5 and AS1 are not here. R5 is a rule and carries no weights, so transfer is
# not defined for it. AS1 answers a question table 2 already answered in
# distribution, and the pair that matters out of distribution is AS4 against F2:
# both pay a drone for the orders it flew, and they differ in the decode --
# simultaneous with a rule resolving conflicts, against sequential where each
# drone sees what the previous ones took.

# Table 5 reports on-time GMV only. The transfer pass records that and a delivered
# count, and a count is not comparable across these targets -- peak generates four
# times the orders -- so the two rates the other tables carry have no basis here.
# Reporting the one metric the pass measured is honest; two empty blocks would not be.
TRANSFER_METRICS = [("on_time_gmv", "On-time GMV", 1)]

TRANSFER_METHODS = [
    ("L2_MSRA", "L2 MSRA (strongest baseline)"),
    ("AS4_SyncOwnReward", "AS4 (simultaneous decode, own-order reward)"),
    ("F2_Separable", "F2 (proposed, rank 0)"),
    ("F2pro", "F2pro (o | l | b, rank 4)"),
]
# One csv per method: they read different checkpoint stores, and eval_pass takes
# one store per call.
TRANSFER_GLOB = "records_transfer/*.csv"

# Table 6: what a decision costs, and the size it was measured at. The F series
# differs in how one dispatch is formed, which is exactly what sets how many rows
# the actor has to score, so a time is meaningless without the size beside it:
#   F3  one row per feasible (order, return pad, battery) triple
#   F2  the factored menu, N_ol + N_b rows, never the product
#   F0  the same factored form with the return pad collapsed
#   F1  three stages, each a smaller set, scored one after another
# Several csv files, because the two methods' checkpoints live in different
# records roots: the proposed method is evaluated at rank 0, which the rank sweep
# wrote to records_rank, while the full-expansion control comes from the main
# records. eval_pass takes one root per call, so each method is one call.
EFFICIENCY_GLOB = "records_efficiency/*.csv"
EFFICIENCY_METHODS = [
    ("F3_FullCombination", "F3 FullCombination (one row per triple)"),
    ("F1_StagedOLB", "F1 AutoregressiveOLB (three sequential stages)"),
    ("F2_Separable", "F2 (proposed, factored (o,l) | b, rank 0)"),
    ("F2pro", "F2pro (o | l | b, rank 4)"),
]

# Table 4 asks one question: is the actor's bilinear coupling needed at all? The
# score of a decision, as a matrix over (order, return pad) rows and battery
# columns, is g*1' + 1*h' + UV' -- two rank-one terms whose one side is constant
# plus a rank-r interaction -- so r=0 is not "no low-rank structure" but "the
# score matrix is exactly rank two". The physics says that is the right rank:
# with one battery capacity the only (o,l)-by-battery quantity the scorer needs
# is the energy margin, charge_b - 0.2C - E_task(o,l), which is a row effect plus
# a column effect and nothing more, and the feasibility threshold is applied by
# the menu rather than learned. So the two rows below are the whole ablation.
RANKS = (0, 8)                     # 8 is the proposed setting, read from the main records
RANK_CITIES = [(label, inst, rec) for label, inst, rec in CITIES]   # every city but peak
RANK_ROOT = "records_rank"
# For the keep ratio each row needs its own trained-on-target baseline, and the
# three rows do not share a records root or a rank: (root, run-name suffix).
TRANSFER_TRAINED = {
    "L2_MSRA": ("records", ""),
    "AS4_SyncOwnReward": (RANK_ROOT, "_rank0_"),
    "F2_Separable": (RANK_ROOT, "_rank0_"),
    "F2pro": (RANK_ROOT, "_rank8_"),
}

# Where each method's cells come from: (records root, run-name suffix). An empty
# suffix means the main sweep; "_rank0_" means the rank sweep. The proposed row is
# reported at rank 0 -- table 4 shows the bilinear coupling costs nothing on any of
# the ten cities and shrinks the seed spread -- so its cells come from the rank
# sweep, not from the main records, which hold the rank-8 runs.
#
# F0 is here for the same reason, one table down. It is the only row of table 3
# besides F2 that carries a rank at all, because it reuses the separable actor on
# a collapsed (order, battery) menu, and the formal sweep took the yaml default of
# 8. Comparing it against F2 at rank 0 would move two knobs at once, so table 3
# reads the rank-0 re-run instead (zbashrun/f0_rank0.sh, 10 cities x 3 seeds); its
# saved order_battery checkpoint records interaction_rank 0 and holds neither
# ol_factor nor b_factor, the two projections the rank-8 actor carries.
SOURCE = {"F2_Separable": (RANK_ROOT, "_rank0_"),
          "F2pro": (RANK_ROOT, "_rank8_"),
          "F0_TwoStage": (RANK_ROOT, "_rank0_")}



def per_seed(rec_city: str, algo: str, key: str, root: str = "records",
             suffix: str = "") -> dict[int, float]:
    """{seed: value} for one cell.

    A trained method's held-out numbers are the eval phase of its training run;
    a rule-based method has no training phase and was run on the test split
    directly, so its numbers are that run's own summary. Both are means over the
    same held-out instances. Where a cell has more than one run per seed (a
    superseded specification), the newest directory wins.
    """
    found: dict[int, float] = {}
    for pattern, name in (
        (f"{root}/train/{rec_city}/drone_only/fixed_step/{algo}/*{suffix}*/", "eval_summary.json"),
        (f"{root}/test/{rec_city}/drone_only/fixed_step/{algo}/*{suffix}*/", "summary.json"),
    ):
        hits = {}
        for d in sorted(glob.glob(pattern)):           # oldest first, newest wins
            path = os.path.join(d, name)
            if not os.path.exists(path):
                continue
            m = re.search(r"seed(\d+)", os.path.basename(d.rstrip("/")))
            if not m:
                continue
            try:
                value = json.load(open(path)).get(key + "_mean")
            except Exception:
                continue
            if value is not None:
                hits[int(m.group(1))] = float(value)
        if hits:
            found = hits
            break
    return found


def ceiling(city_dir: str) -> float:
    """Total value of every generated order, averaged over the test instances."""
    totals = []
    for inst in sorted(glob.glob(f"data/instances_drone/{city_dir}/test/*/")):
        orders = yaml.load(open(inst + "orders.yaml"), Loader=_Loader)
        totals.append(sum(float(o["value"]) for o in orders))
    return float(np.mean(totals)) if totals else float("nan")


def md(rows: list[list[str]], header: list[str]) -> str:
    # a pipe inside a cell is the column separator, so it has to be escaped or the
    # row silently shifts: "factored (o,l)|b" split the proposed method's row in two
    esc = lambda c: c.replace("|", r"\|")
    header = [esc(h) for h in header]
    rows = [[esc(c) for c in r] for r in rows]
    widths = [max([len(header[i])] + [len(r[i]) for r in rows]) for i in range(len(header))]
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        out.append("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


def build(check: bool = False, methods=None, title: str = None,
          relative_to: str = None, intro: list[str] = None, cities=None,
          rank_suffix: str = "") -> str:
    methods = methods or METHODS
    cities = cities if cities is not None else CITIES
    data = {}
    missing = []
    for key, _, _ in METRICS:
        for algo, _ in methods:
            for label, _, rec in cities:
                if rank_suffix:                    # table 2: every row at rank 0
                    got = per_seed(rec, algo, key, RANK_ROOT, rank_suffix)
                elif algo in SOURCE:                 # table 1: the proposed row
                    got = per_seed(rec, algo, key, *SOURCE[algo])
                else:
                    got = per_seed(rec, algo, key)
                data[(key, algo, label)] = got
                for s in SEEDS:
                    if s not in got:
                        missing.append(f"{key}/{algo}/{label}/s{s}")
    if check:
        return (f"{len(missing)} missing cells\n" + "\n".join("  " + m for m in missing) + "\n"
                if missing else "complete\n")
    incomplete = sorted({m.split("/")[1] + " on " + m.split("/")[2] for m in missing})

    parts = [
        f"# {title or 'Table 1: every method on the ten cities'}",
        "",
        "Protocol: 300 training episodes, validation every 50, the best-validation",
        "checkpoint evaluated greedily on the city's held-out test instances (30, or 28",
        "and 25 for LaDe Chongqing and Shanghai), candidate pruning off."
        + (" Rule-based methods are not trained and were run on the same test split"
           if methods is METHODS else ""),
        "directly." if methods is METHODS else "All methods here are trained.",
        "",
        "Every value below is already a mean over that city's test instances; the three",
        "seeds are listed separately and nothing is averaged across cities. Values are",
        "unrounded apart from the digits shown.",
        "",
        "`ceiling` is the total value of all generated orders, the most on-time GMV could",
        "possibly be. All cities price orders with the same value model, so a city's GMV",
        "differs from another's by order volume, not by scale.",
        "",
        "The proposed row is the factored method at rank 0: no bilinear coupling between",
        "the (order, return pad) block and the battery block. Table 4 is the evidence --",
        "dropping the coupling costs nothing measurable on any of the ten cities (median",
        "+0.36% on-time GMV, worst -0.30%, every negative inside its own seed spread) and",
        "shrinks the seed spread itself, by an order of magnitude on two cities. These",
        "cells are therefore the rank-0 runs, not the rank-8 runs the earlier sweeps",
        "produced.",
        "",
        "F2pro is the other instantiation of the same decomposition. It splits the",
        "order from the return pad as well, so the rows a decision needs fall from",
        "N_o*N_l + N_b to N_o + N_l + N_b -- linear in each factor, which is what the",
        "combinatorial action space asks for, and on these cities it cuts the worst",
        "case of a decision from about 2600 rows to under 170. The split costs the",
        "precomputed E_task(o, l), which the interaction term then has to recover, so",
        "unlike the row above it needs a rank: 8, the largest the sln sweep measured.",
        "The two rows price that trade-off against each other.",
        "",
    ]
    if intro:
        parts = parts[:2] + intro + [""] + parts[2:]
    if incomplete:
        parts += ["Cells left blank are still running: " + ", ".join(incomplete) + ".", ""]
    cl = [f"{ceiling(cd):.1f}" for _, cd, _ in cities]
    parts.append("## Ceiling: total order value per city")
    parts.append("")
    parts.append(md([["total order value"] + cl], ["metric"] + [c for c, _, _ in cities]))
    parts.append("")

    for key, title_m, digits in METRICS:
        parts.append(f"## {title_m}")
        parts.append("")
        header = ["method"] + [f"{c} s{s}" for c, _, _ in cities for s in SEEDS]
        rows = []
        for algo, label in methods:
            cells = []
            for c, _, _ in cities:
                got = data[(key, algo, c)]
                for s in SEEDS:
                    cells.append(f"{got[s]:.{digits}f}" if s in got else "")
            rows.append([label] + cells)
        parts.append(md(rows, header))
        parts.append("")
        parts.append(f"### {title_m} -- mean and sd over the three seeds")
        parts.append("")
        header2 = ["method"] + [f"{c}" for c, _, _ in cities]
        rows2 = []
        for algo, label in methods:
            cells = []
            for c, _, _ in cities:
                v = [data[(key, algo, c)][s] for s in SEEDS if s in data[(key, algo, c)]]
                cells.append(f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}"
                             if len(v) == 3 else "")
            rows2.append([label] + cells)
        parts.append(md(rows2, header2))
        parts.append("")
        if relative_to:
            parts.append(f"### {title_m} -- relative to the reference, per city (%)")
            parts.append("")
            rows3 = []
            for algo, label in methods:
                if algo == relative_to:
                    continue
                cells = []
                for c, _, _ in cities:
                    mine = [data[(key, algo, c)][sd] for sd in SEEDS if sd in data[(key, algo, c)]]
                    ref = [data[(key, relative_to, c)][sd] for sd in SEEDS
                           if sd in data[(key, relative_to, c)]]
                    cells.append(f"{100 * (np.mean(mine) / np.mean(ref) - 1):+.2f}"
                                 if len(mine) == 3 and len(ref) == 3 else "")
                rows3.append([label] + cells)
            parts.append(md(rows3, ["method"] + [c for c, _, _ in cities]))
            parts.append("")
    return "\n".join(parts)


def build_rank_table() -> str:
    """Table 4: the rank of the bilinear coupling.

    Every row is the proposed method with one number changed, so the row at r=8 is
    the same run reported in table 1 and r=0 is the purely additive score: the
    (o,l) block and the battery block contribute independently and nothing couples
    them. Reading down a column therefore prices the coupling, and the step from
    r=0 to r=1 prices having any coupling at all.
    """
    data = {}
    for key, _, _ in METRICS:
        for r in RANKS:
            for label, _, rec in RANK_CITIES:
                # r=8 comes from the main records: those are the runs tables 1-3
                # report, and re-running them only to match batches was judged not
                # worth the GPU time once r=0 landed within noise of them. The
                # header says what that costs. The trailing underscore matters on
                # the sweep side: "_rank1" alone would also match _rank16.
                root, suffix = (("records", "") if r == 8 else (RANK_ROOT, f"_rank{r}_"))
                data[(key, r, label)] = per_seed(rec, "F2_Separable", key, root, suffix)

    parts = [
        "# Table 4: is the bilinear coupling needed?",
        "",
        "Two rows, one knob. r=8 is the setting tables 1-3 report, read from those",
        "same runs; r=0 drops the bilinear term, so the actor scores a decision as",
        "g(order, return pad) + h(battery) and nothing couples the two blocks.",
        "",
        "One caveat on the comparison: the r=0 runs were trained on 2026-10-02/03,",
        "while the r=8 runs they are compared against are from 2026-09-21 and 09-28,",
        "so a difference of well under a percent could come from the code moving in",
        "between rather than from the rank. The claim this table supports is the",
        "negative one -- that dropping the term costs nothing measurable -- which",
        "that confound cannot manufacture: it would have to coincidentally cancel a",
        "real effect. A same-batch r=8 was planned and dropped as not worth the GPU",
        "time once r=0 came in at parity.",
        "",
        "r=0 is not the absence of low-rank structure. Written as a matrix over the",
        "(order, return pad) rows and the battery columns, the score is",
        "",
        "    Z = g 1' + 1 h' + U V'",
        "",
        "two rank-one terms whose one side is constant, plus a rank-r interaction, so",
        "rank(Z) <= r + 2. r=0 means the score matrix is exactly rank two.",
        "",
        "Rank two is what the problem contains. Every battery here holds 720 Wh, so",
        "the reserve 0.2C is a constant and the only (o,l)-by-battery quantity the",
        "scorer needs is the energy margin",
        "",
        "    margin = charge_b - 0.2C - E_task(o,l),",
        "",
        "a battery effect plus an (order, return pad) effect and no product of the",
        "two. The threshold margin >= 0, the one place a nonlinear function of it",
        "would be needed, is applied when the menu is built: an infeasible triple",
        "never reaches the actor. An earlier offline study on logged menus agreed,",
        "putting rank's contribution to held-out argmax agreement at most 0.02.",
        "",
        "Actor parameters: 54,405 at r=0, 58,517 at r=8 (+514 per unit of rank), and",
        "the critic is factor_v, which takes no rank, so it is identical in both rows.",
        "",
        "meituan-peak is left out: it is the stress column of table 1, not an ablation",
        "axis, and r=8 has not finished there.",
        "",
    ]
    for key, title_m, digits in METRICS:
        parts.append(f"## {title_m}")
        parts.append("")
        header = ["rank"] + [f"{c} s{sd}" for c, _, _ in RANK_CITIES for sd in SEEDS]
        rows = []
        for r in RANKS:
            cells = []
            for c, _, _ in RANK_CITIES:
                got = data[(key, r, c)]
                cells += [f"{got[sd]:.{digits}f}" if sd in got else "" for sd in SEEDS]
            rows.append([f"r={r}" + (" (proposed)" if r == 8 else " (additive)")] + cells)
        parts.append(md(rows, header))
        parts.append("")
        parts.append(f"### {title_m} -- mean and sd, and the change from r=8")
        parts.append("")
        header2 = ["rank"] + [f"{c}{w}" for c, _, _ in RANK_CITIES for w in ("", " vs r=8 (%)")]
        rows2 = []
        for r in RANKS:
            cells = []
            for c, _, _ in RANK_CITIES:
                v = [data[(key, r, c)][sd] for sd in SEEDS if sd in data[(key, r, c)]]
                ref = [data[(key, 8, c)][sd] for sd in SEEDS if sd in data[(key, 8, c)]]
                cells.append(f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}"
                             if len(v) == 3 else "")
                cells.append(f"{100 * (np.mean(v) / np.mean(ref) - 1):+.2f}"
                             if len(v) == 3 and len(ref) == 3 and r != 8 else
                             ("--" if r == 8 else ""))
            rows2.append([f"r={r}" + (" (proposed)" if r == 8 else " (additive)")] + cells)
        parts.append(md(rows2, header2))
        parts.append("")
    return "\n".join(parts)


def transfer_cells(key: str) -> dict:
    """{(algo, target): {seed: value}} from the transfer csv, averaged per episode.

    eval_pass.py writes one row per (city, algo, seed, instance); a cell's number is
    the mean over that city's test instances, which is what the training runs' own
    eval phase reports, so a transfer number and a trained number are comparable.
    """
    out: dict = {}
    import collections
    import glob as _glob
    bucket = collections.defaultdict(list)
    rows = []
    for path in sorted(_glob.glob(TRANSFER_GLOB)):
        with open(path) as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        return out
    for row in rows:
        # only the transferred rows. A pass that also evaluated a city on its own
        # checkpoint writes those to the same directory, and they are not transfer.
        if row.get("source_city") != TRANSFER_SOURCE or key not in row:
            continue
        # eval_pass names the column "algorithm"; "algo" was a guess and would
        # have raised on the first row
        bucket[(row["algorithm"], row["city"], int(row["seed"]))].append(float(row[key]))
    for (algo, city, seed), values in bucket.items():
        out.setdefault((algo, city), {})[seed] = float(np.mean(values))
    return out


def build_transfer_table() -> str:
    """Table 5: a model trained on one scenario, run on a perturbed one."""
    got = {key: transfer_cells(key) for key, _, _ in TRANSFER_METRICS}
    targets = [(c, rec) for c, rec, _ in TRANSFER_TARGETS]
    parts = [
        "# Table 5: zero-shot transfer off " + TRANSFER_SOURCE,
        "",
        "Every row is the checkpoint trained on " + TRANSFER_SOURCE + " and evaluated,",
        "without any further training, on a perturbed scenario. The five variants are",
        "symlink overlays on the base meituan instances: the order stream, the network",
        "and the launchpads are identical to the source and exactly one resource moves,",
        "so a drop is attributable to that resource. meituan-peak is a different",
        "generator window carrying four times the volume, so it shifts demand rather",
        "than supply and is the hardest of the six.",
        "",
        "`keep` is the transfer score over the same method trained on that target: the",
        "share of a locally-trained policy that the transferred weights hold on to.",
        "keep > 1 means the perturbation is easier than the source, not that transfer",
        "helped.",
        "",
        "Each row needs its own trained baseline, and they were not all there to begin",
        "with: L2 had all six, F2 and F2pro had five (nobody had trained the learned",
        "methods on peak), and AS4 had none, since table 2 deliberately used the five",
        "base scenarios only. Those 8 missing baselines are being trained; a keep cell",
        "is blank until its baseline exists, never filled with a substitute.",
        "",
        "AS4 against F2 is the pair to read. Both pay a drone for the orders it flew,",
        "and they differ in the decode: AS4 has every idle drone of a step choose from",
        "the same pre-commitment menu with conflicts resolved by rule, while F2 decodes",
        "them in sequence, each seeing what the previous ones took. Table 2 prices that",
        "difference in distribution at -2.60% on-time GMV; this table asks whether it",
        "also changes how the policy holds up when a resource moves.",
        "",
    ]
    parts.append("| scenario | perturbation |")
    parts.append("|----------|--------------|")
    for c, _, what in TRANSFER_TARGETS:
        parts.append(f"| {c} | {what} |")
    parts.append("")

    for key, title, digits in TRANSFER_METRICS:
        cells = got[key]
        parts.append(f"## {title}")
        parts.append("")
        header = ["method"] + [f"{c} s{sd}" for c, _ in targets for sd in SEEDS]
        rows = []
        for algo, label in TRANSFER_METHODS:
            row = []
            for c, _ in targets:
                seen = cells.get((algo, c), {})
                row += [f"{seen[sd]:.{digits}f}" if sd in seen else "" for sd in SEEDS]
            rows.append([label] + row)
        parts.append(md(rows, header))
        parts.append("")

        parts.append(f"### {title} -- transfer, trained on target, and what transfer keeps")
        parts.append("")
        header3 = ["method"] + [f"{c} {w}" for c, _ in targets
                               for w in ("transfer", "trained", "keep")]
        rows3 = []
        for algo, label in TRANSFER_METHODS:
            row = []
            for c, rec in targets:
                seen = cells.get((algo, c), {})
                moved = [seen[sd] for sd in SEEDS if sd in seen]
                root, suffix = TRANSFER_TRAINED.get(algo, ("records", ""))
                local = per_seed(rec, algo, key, root, suffix)
                home = [local[sd] for sd in SEEDS if sd in local]
                row.append(f"{np.mean(moved):.{digits}f}" if len(moved) == 3 else "")
                row.append(f"{np.mean(home):.{digits}f}" if len(home) == 3 else "")
                row.append(f"{np.mean(moved) / np.mean(home):.3f}"
                           if len(moved) == 3 and len(home) == 3 else "")
            rows3.append([label] + row)
        parts.append(md(rows3, header3))
        parts.append("")

        parts.append(f"### {title} -- mean and sd over the three seeds")
        parts.append("")
        rows2 = []
        for algo, label in TRANSFER_METHODS:
            row = []
            for c, _ in targets:
                seen = cells.get((algo, c), {})
                v = [seen[sd] for sd in SEEDS if sd in seen]
                row.append(f"{np.mean(v):.{digits}f} ± {np.std(v, ddof=1):.{digits}f}"
                           if len(v) == 3 else "")
            rows2.append([label] + row)
        parts.append(md(rows2, ["method"] + [c for c, _ in targets]))
        parts.append("")

        parts.append(f"### {title} -- relative to the proposed method, per scenario (%)")
        parts.append("")
        rows3 = []
        for algo, label in TRANSFER_METHODS:
            if algo == "F2_Separable":
                continue
            row = []
            for c, _ in targets:
                mine = [v for sd in SEEDS
                        for v in [cells.get((algo, c), {}).get(sd)] if v is not None]
                ref = [v for sd in SEEDS
                       for v in [cells.get(("F2_Separable", c), {}).get(sd)] if v is not None]
                row.append(f"{100 * (np.mean(mine) / np.mean(ref) - 1):+.2f}"
                           if len(mine) == 3 and len(ref) == 3 else "")
            rows3.append([label] + row)
        parts.append(md(rows3, ["method"] + [c for c, _ in targets]))
        parts.append("")
    return "\n".join(parts)


def build_efficiency_table() -> str:
    """Table 6: decision time for the F series, each next to the size it was measured at."""
    import collections
    import glob as _glob
    paths = sorted(_glob.glob(EFFICIENCY_GLOB))
    if not paths:
        return ("# Table 6: decision time\n\n"
                f"No csv matches `{EFFICIENCY_GLOB}` yet. Produce it with, for the\n"
                "proposed method at rank 0 and then the full-expansion control:\n\n"
                "    python runs/eval_pass.py --algos F2_Separable --seeds 42 \\\n"
                "        --records-root records_rank --run-suffix _rank0 --device cuda \\\n"
                "        --out records_efficiency/f2_rank0.csv\n"
                "    python runs/eval_pass.py --algos F3_FullCombination --seeds 42 \\\n"
                "        --records-root records --device cuda \\\n"
                "        --out records_efficiency/f3.csv\n")

    rows = collections.defaultdict(list)
    cities: list[str] = []
    for path in paths:
        with open(path) as handle:
            for row in csv.DictReader(handle):
                rows[(row["algorithm"], row["city"])].append(row)
                if row["city"] not in cities:
                    cities.append(row["city"])
    order = [label for label, _, _ in CITIES]
    cities = [c for c in order if c in cities] + [c for c in cities if c not in order]

    def cell(algo, city, what):
        got = rows.get((algo, city))
        if not got:
            return ""
        total = lambda k: sum(float(r[k]) for r in got)
        actions = max(1.0, total("actions"))
        groups = max(1.0, total("decision_groups"))
        if what == "ms_per_action":
            return f"{1000.0 * total('policy_time_s') / actions:.2f}"
        if what == "ms_per_decision":
            return f"{1000.0 * total('policy_time_s') / groups:.3f}"
        if what == "rows_per_decision":
            return f"{total('feature_rows') / groups:.1f}"
        if what == "triples_per_decision":
            # F1 never forms the triple product, so its candidate counter holds
            # the stage-1 ORDER menu (1.7-7.2 per decision) and not a triple
            # count. Printed, that column would read as a two-triple decision.
            if algo == "F1_StagedOLB":
                return ""
            return f"{total('candidates') / groups:.1f}"
        return ""

    parts = [
        "# Table 6: decision cost, and the size it was measured at",
        "",
        "Four methods, one axis: how a decision's candidates are presented to the",
        "actor. F3 expands the feasible (order, return pad, battery) triples into one",
        "row each. F2 scores the (order, return pad) rows and the battery rows once",
        "each and gathers the grid. F2pro splits the order from the return pad too, so",
        "it scores three blocks. F1 never forms the product at all: it decodes",
        "o -> l -> b in three sequential stages, so it scores the fewest rows of the",
        "four and pays three forward passes per decision to do it. Tables 1 and 3",
        "settle what they score; this table is only what they cost.",
        "",
        "One warning before the rows. F1's rows/decision is NOT the same quantity as",
        "the other three. The counter sums its three stages, each of which scores a",
        "menu built against the previous stage's choice, so the number says how many",
        "rows passed through an actor and not how large a single scored set was. The",
        "ms columns are comparable across all four; the row column is comparable only",
        "among F3, F2 and F2pro, which score one set each.",
        "",
        "Read it as two findings that point opposite ways.",
        "",
        "The rows follow the arithmetic exactly. F2 replaces N_o*N_l*N_b with",
        "N_o*N_l + N_b, so its ratio tends to N_b once N_b << N_o*N_l -- and a pad",
        "offers two or three selectable batteries. That ceiling is visible: F2 sits at",
        "a median 1.59x and does not grow with the menu (1.87x on the smallest menu",
        "here, 1.92x on the largest). F2pro replaces it with N_o + N_l + N_b, linear",
        "in each factor rather than quadratic in two, and its ratio does grow with the",
        "menu -- ordered by triples per decision it runs 1.65, 1.39, 2.19, 2.05, 2.13,",
        "2.32, 2.74, 3.01, 5.09, 6.45x. On meituan-sln the actor is handed 69.7 rows",
        "where F3 is handed 449.8.",
        "",
        "At the small end the two factorisations cross, and F2pro is the worse of the",
        "two: 13.8 rows against F2's 8.6 on lade_cq, 10.5 against 9.3 on lade_sh. This",
        "is the arithmetic, not a defect -- when a factor degenerates to a singleton,",
        "a single order against ten pads say, N_o + N_l = 11 exceeds N_o*N_l = 10.",
        "Splitting a factor only pays once the product is large enough",
        "to pay for the extra block.",
        "",
        "The wall-clock does not follow, and this is the result to be honest about.",
        "The median saving is 1.12x for F2 and 1.06x for F2pro per dispatch issued,",
        "and the ratio *falls* as the menu grows -- exactly where the row saving",
        "rises. On the two largest menus F2pro is level with F3 and then behind it",
        "(1.02x on meituan-asn, 0.94x on meituan-sln), while scoring 5.09x and 6.45x",
        "fewer rows there. Cutting what the actor scores is not cutting what a",
        "decision costs.",
        "",
        "Against F2 rather than F3 it is sharper, and this is the comparison the",
        "paper actually has to make, since F2 is the proposed method and F2pro the",
        "variant. F2pro is slower than F2 on all eleven cities, in both columns:",
        "a median 1.12x per decision and 1.12x per dispatch, from 1.02x on",
        "meituan-asn to 1.23x on grubhub. So the two orderings are reversed for",
        "F2pro -- fewest rows scored of the three, slowest of the three in time --",
        "and the slowdown does not shrink as the row saving grows. Ordered by",
        "triples per decision it is 1.20, 1.12, 1.20, 1.05, 1.16, 1.13, 1.07, 1.03,",
        "1.02, 1.23, 1.09x: no trend. Three blocks instead of two costs a roughly",
        "constant eighth of a decision, and cutting up to 6.45x of the rows never",
        "pays it back at these instance sizes.",
        "",
        "One shared cost is measurable and explains the direction. The energy calls",
        "per decision barely move between the three methods -- a 4.8% spread across",
        "them on the median city, 17.5% at worst -- while they grow sixfold with the",
        "menu, 46 per decision on lade_sh to 336 on meituan-sln. The factorisation",
        "changes how candidates are scored, not how many are built, tested for",
        "feasibility, or assembled and sampled over as a product grid at decode. That",
        "shared work grows with the triples while the saving grows with the rows, so",
        "the two cancel. policy_time_s is logged as one figure and not split into",
        "scoring and decode, so this is the account consistent with the measurements",
        "rather than a measured decomposition.",
        "",
        "F1 settles the question the other three only point at, and it does so in",
        "the opposite direction from the row counts. It scores 13.0-62.8 rows a",
        "decision against F3's 18.5-389.4 -- fewer on all six cities, and the fewest",
        "of the four on meituan-sln and meituan-asn, the two largest menus -- and it",
        "is the slowest of the four on every city, 1.93-3.98x F2 per decision and",
        "1.88-3.62x per dispatch. On meituan-sln it scores 6.2x fewer rows than F3",
        "and takes 2.15x longer (8.818 ms against 4.110). The reason is not the rows:",
        "a staged decode launches three dependent forward passes per decision, each",
        "waiting on the previous stage's choice, so none of the three can be batched",
        "with another. F2 scores more rows than F1 on five of the six cities and is",
        "between two and four times faster on all six. Cutting what the",
        "actor scores is not cutting what a decision costs; serialising the decode to",
        "cut it further costs more than the cut is worth.",
        "",
        "So the factorisation flattens the growth of what the actor scores from",
        "quadratic to linear in the menu, and at these instance sizes that does not",
        "convert into time. The case for it rests on table 4 -- it constrains the",
        "score matrix to the rank the problem has -- and on the parameter count, not",
        "on this table.",
        "",
        "  rows/decision     what the actor was actually given -- one row per triple",
        "                    for F3, N_ol + N_b for F2, N_o + N_l + N_b for F2pro.",
        "                    This is the only quantity the factorisation changes.",
        "  triples/decision  the menu the decision faced, counted identically for all",
        "                    three, so it says how big the problem was. The three do",
        "                    not face identical menus: different policies reach",
        "                    different states, so this column also shows how far the",
        "                    trajectories diverge.",
        "  ms/decision       wall-clock inside the policy per decision group.",
        "  ms/action         the same time per dispatch actually issued, which is what",
        "                    an operator feels; the two differ because a decision can",
        "                    end in defer or stop.",
        "",
        "F2 is evaluated at rank 0 and F2pro at rank 4, each the setting table 4",
        "selects for it. F3 has no bilinear term, so no rank applies to it and its",
        "checkpoints are the ones tables 1 and 3 report.",
        "",
        "Measured by a serial single-process pass over the saved checkpoints with no",
        "training running: a time measured while six jobs share the GPU is not a",
        "property of the method. The policy accounts for 52-74% of wall time across",
        "these runs, and on a given city the methods agree on that share to",
        "within 4.5 points, so the comparison is not an artefact of how much of the",
        "loop each one covers. One seed, since the",
        "cost follows the architecture and the menu rather than the weights.",
        "",
    ]
    for what, title in (("rows_per_decision", "Rows the actor scored, per decision"),
                        ("triples_per_decision", "Feasible triples the decision faced"),
                        ("ms_per_decision", "Policy time per decision (ms)"),
                        ("ms_per_action", "Policy time per dispatch issued (ms)")):
        parts.append(f"## {title}")
        parts.append("")
        body = [[label] + [cell(algo, c, what) for c in cities]
                for algo, label in EFFICIENCY_METHODS]
        parts.append(md(body, ["method"] + cities))
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--table", type=int, default=1, choices=(1, 2, 3, 4, 5, 6))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.table == 1:
        out_default = "mdtable/table1_main.md"
        kw = dict(cities=CITIES_WITH_PEAK)
    elif args.table == 3:
        out_default = "mdtable/table3_action_expansion.md"
        kw = dict(
            methods=AE_METHODS,
            title="Table 3: action expansion",
            relative_to="F2_Separable",
            intro=[
                "All four keep the proposed method's serialized credit assignment -- one GAE",
                "trajectory per physical drone, carrying only the rewards of the orders that",
                "drone flew -- and the same PPO hyperparameters and budget. They differ only in",
                "how one dispatch is formed and scored:",
                "",
                "  F3   expands the feasible (order, return pad, battery) triples into one row",
                "       each and scores them with a full-capacity actor and a row critic. The",
                "       control for the factorisation: identical decode, trajectories and",
                "       training, only the actor's treatment of the triple differs.",
                "  F1   never forms the product. Three sequential stages per decision: an",
                "       existence-only feasibility test per order against the highest-charge",
                "       local battery, then the return targets of the one chosen order, then",
                "       the battery, whose energy requirement is by then a single number.",
                "  F0   two learned agents that never share a menu. The dispatch agent picks",
                "       (order, battery) and observes the order and battery blocks only -- no",
                "       launchpad state, and not the minimum round-trip energy either, so it",
                "       knows which pairs are legal but not what the return costs. A separate",
                "       agent picks the pad once the drone is at the customer endpoint.",
                "  F2   scores the factored (o,l) | b menu with the separable actor.",
                "",
                "On the rank each row carries. Every row that carries one carries 0, so the",
                "axis this table varies is the action expansion and nothing else. F1 and F3",
                "have no rank at all -- F1 scores three stages with three separate actors and",
                "F3 scores one row per triple with a full-capacity actor, and neither builds a",
                "bilinear term -- so their numbers are the same at any rank and need no",
                "re-run. F2 is reported at rank 0, the setting table 4 selects. F0 does carry",
                "a rank, since it reuses the separable actor on a collapsed (order, battery)",
                "menu, and the formal sweep took the yaml default of 8; its cells here are the",
                "rank-0 re-run of all ten cities and three seeds instead, whose saved dispatch",
                "checkpoint holds neither of the two projections the rank-8 actor carries.",
                "Dropping the coupling moved F0 up by 0.18% on average, between -0.78% and",
                "+0.73% and upward on eight of the ten cities -- the same scale and the same",
                "eight-of-ten pattern table 4 reports for F2. So the rank-8 row this replaces",
                "was marginally flattering to F0, not punitive, and the gap the table shows",
                "narrows very slightly rather than widening.",
            ],
        )
    else:
        out_default = "mdtable/table2_agent_structure.md"
        kw = dict(
            methods=AS_METHODS,
            cities=AS_CITIES,
            rank_suffix=AS_RANK_SUFFIX,
            title="Table 2: agent structure (rank 0)",
            relative_to="F2_Separable",
            intro=[
                "All four share the proposed method's menu construction, feature layout and",
                "separable actor, and are trained with the same PPO hyperparameters and budget.",
                "They differ only in who the agent is and how the decision stream is cut for the",
                "update, so a difference between rows is attributable to that alone:",
                "",
                "  AS1  synchronous -- every idle drone of a step decides from the same state",
                "       without seeing the others, the step becomes one joint action, a central",
                "       critic on the global state gives the value, and conflicts on the same",
                "       order or battery are resolved by rule afterwards.",
                "  AS2  drones still decide in sequence, but the whole episode is one GAE",
                "       trajectory carrying the team reward.",
                "  AS3  the fleet is one agent: drones decode sequentially as in the proposed",
                "       method, but a step's decisions are packed into one joint transition with",
                "       one value, one advantage and one PPO ratio, valued by the same global",
                "       state critic AS1 uses.",
                "  F2   one GAE trajectory per physical drone, carrying only the rewards of the",
                "       orders that drone flew.",
                "",
                "All six are trained with the actor's bilinear coupling removed (rank 0),",
                "which table 4 shows costs nothing on these cities, so no row carries a term",
                "the others lack. The five cities are the base scenarios; the meituan",
                "variants perturb one resource of the base set and belong to table 1.",
                "",
                "AS4 and AS5 move one knob each off AS1 and F2, so the two axes separate:",
                "AS1 vs AS4 is the credit under simultaneous decode (team vs own-order), and",
                "F2 vs AS5 is the same question under sequential decode. AS5 is not AS2: the",
                "cut stays per-drone and only the reward stream changes, where AS2 moves both.",
                "AS4 also replaces AS1's team V(s) with V(s, drone), which an own-order reward",
                "requires -- a team-sized baseline under an own-order reward is not a choice.",
            ],
        )
    if args.table == 6:
        out = Path(args.out or "mdtable/table6_efficiency.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_efficiency_table())
        print(f"written {out}")
        return
    if args.table == 5:
        out = Path(args.out or "mdtable/table5_transfer.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_transfer_table())
        print(f"written {out}")
        return
    if args.table == 4:
        out = Path(args.out or "mdtable/table4_rank.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_rank_table())
        print(f"written {out}")
        return
    text = build(check=args.check, **kw)
    if args.check:
        print(text, end="")
        return
    out = Path(args.out or out_default)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"written {out}  ({len(text.splitlines())} lines)")


if __name__ == "__main__":
    main()
