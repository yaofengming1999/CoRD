"""Fill the mdtable2 frames from the 30-column runs, cell by cell.

Patching cells rather than regenerating files: the frames carry prose that says
what each table is for and which of its rows are carried over from the
44-column set, and that prose is the part that keeps the two configurations
from being mixed. A regenerator would have to hold it, and would eventually
drop it.

One run can land in several tables -- F2 on meituan-ssn is a row of table 1.1,
of table 1.2, of table 2 and of table 3, and F2pro at rank 4 is also table
4.1's r=4 row -- so the mapping is from (algorithm, rank) to a set of
(file, row) targets, and the run is written to every one of them.

The seed a run carries comes from its directory name, not from a flag: the
three seeds are interleaved in one queue, so at any moment some cities have all
three and others have one, and a flag would have to be right about which.

The mean-and-sd cells are recomputed from every seed present and overwritten on
each pass, so they track the sweep as it lands. They say the mean alone while
there is one seed. What a single run has is a spread over test instances, which
is a different quantity, and writing it where a seed spread belongs is how a
one-seed result gets read as three.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "mdtable2"

# (algorithm, critic override or None) -> [(file, row label), ...]
#
# The second slot used to be an unused "rank or None" and is now the
# ``--critic-type`` override a run carries, because AS4 is trained twice on
# meituan-ssn -- once with its own V(s, drone) and once with F2's factored
# critic -- and the two are the same (algorithm, city, rank). Without the
# critic in the key the collector refused both as one ambiguous cell, which is
# how this was found rather than silently averaged.
ROWS: dict[tuple[str, str | None], list[tuple[str, str]]] = {
    ("F2_Separable", None): [
        ("table1_1_base.md", "F2 (proposed, rank 0)"),
        ("table1_2_variants.md", "F2 (proposed, rank 0)"),
        ("table1_3_variants25.md", "F2 (proposed, rank 0)"),
        ("table1_4_charge.md", "F2 (proposed, rank 0)"),
        ("table2_agent_structure.md", "F2 (proposed, sequential, own-order reward)"),
        ("table3_action_expansion.md", "F2 (proposed, factored (o,l) \\| b, rank 0)"),
    ],
    ("F2pro", None): [
        ("table1_1_base.md", "F2pro (o \\| l \\| b, rank 4)"),
        ("table1_2_variants.md", "F2pro (o \\| l \\| b, rank 4)"),
        ("table1_3_variants25.md", "F2pro (o \\| l \\| b, rank 4)"),
        ("table1_4_charge.md", "F2pro (o \\| l \\| b, rank 4)"),
    ],
    ("AS1_MAPPO", None): [("table2_agent_structure.md", "AS1 MAPPO (synchronous, team reward)")],
    ("AS2_SingleDronePPO", None): [("table2_agent_structure.md", "AS2 SingleDronePPO (one trajectory per episode)")],
    ("AS3_GlobalDronesPPO", None): [("table2_agent_structure.md", "AS3 GlobalDronesPPO (one joint transition per step)")],
    ("AS4_SyncOwnReward", None): [("table2_agent_structure.md", "AS4 MAPPO (synchronous, own-order reward)")],
    ("AS4_SyncOwnReward", "factor_v"): [("table2_agent_structure.md", "AS4' MAPPO (synchronous, own-order reward, critic = F2's)")],
    ("AS5_TeamReward", None): [("table2_agent_structure.md", "AS5 SerializedMARL (sequential, team reward)")],
    ("F3_FullCombination", None): [("table3_action_expansion.md", "F3 FullCombination (one row per triple)")],
    ("F1_StagedOLB", None): [("table3_action_expansion.md", "F1 AutoregressiveOLB (staged o -> l -> b)")],
    ("F1b_IndependentOLB", None): [("table3_action_expansion.md", "F1b IndependentOLB (o, l, b drawn independently)")],
    ("F0_TwoStage", None): [("table3_action_expansion.md", "F0 TwoStage (two separate learned agents)")],
    # The four learned baselines. In records_f30 these exist only on the two
    # cities table 1.1 had no baseline at all for; the carried-over cells of the
    # other five cities are 44-column runs marked dagger and no group here
    # reaches them, so the mean cells, which are rewritten every pass, cannot
    # overwrite a carryover.
    # Both tables, because the five meituan variant cities are table 1.2's and the
    # L3/L4 re-run covers them. A row that has no group under records_f30 for a
    # given city is simply not reached, so listing both targets cannot disturb the
    # L1 and L2 carryovers: those two are not re-run, their per-seed cells are
    # filled and write-once, and no L1/L2 group exists for a table 1.2 city.
    # The five rule baselines. They train nothing, so their run is one pass over
    # every test instance and lands under records_f30/test/ with its aggregate in
    # summary.json rather than eval_summary.json -- see finished().
    ("R1_urgentLower_nearest", None): [("table1_1_base.md", "R1 urgent, nearest"),
                                       ("table1_2_variants.md", "R1 urgent, nearest"),
                                       ("table1_3_variants25.md", "R1 urgent, nearest"),
                                       ("table1_4_charge.md", "R1 urgent, nearest")],
    ("R2_random_random", None): [("table1_1_base.md", "R2 random"),
                                 ("table1_2_variants.md", "R2 random"),
                                 ("table1_3_variants25.md", "R2 random"),
                                 ("table1_4_charge.md", "R2 random")],
    ("R3_fifoLower_backlogHeu", None): [("table1_1_base.md", "R3 FIFO, lowest feasible"),
                                        ("table1_2_variants.md", "R3 FIFO, lowest feasible"),
                                        ("table1_3_variants25.md", "R3 FIFO, lowest feasible"),
                                        ("table1_4_charge.md", "R3 FIFO, lowest feasible")],
    ("R4_fifoFull_backlogHeu", None): [("table1_1_base.md", "R4 FIFO, full charge"),
                                       ("table1_2_variants.md", "R4 FIFO, full charge"),
                                       ("table1_3_variants25.md", "R4 FIFO, full charge"),
                                       ("table1_4_charge.md", "R4 FIFO, full charge")],
    ("R5_hungarianGmv_backlogHeu", None): [("table1_1_base.md", "R5 Hungarian on-time GMV"),
                                           ("table1_2_variants.md", "R5 Hungarian on-time GMV"),
                                           ("table1_3_variants25.md", "R5 Hungarian on-time GMV"),
                                           ("table1_4_charge.md", "R5 Hungarian on-time GMV")],
    ("L1_RBG", None): [("table1_1_base.md", "L1 RBG"),
                       ("table1_2_variants.md", "L1 RBG")],
    ("L2_MSRA", None): [("table1_1_base.md", "L2 MSRA"),
                        ("table1_2_variants.md", "L2 MSRA"),
                        ("table1_3_variants25.md", "L2 MSRA"),
                        ("table1_4_charge.md", "L2 MSRA")],
    ("L3_MAPDP", None): [("table1_1_base.md", "L3 MAPDP"),
                         ("table1_2_variants.md", "L3 MAPDP")],
    ("L4_MoEPointer", None): [("table1_1_base.md", "L4 MoE-Pointer"),
                              ("table1_2_variants.md", "L4 MoE-Pointer")],
}

# the metric sections, and the eval_summary key each reads
METRICS = [("On-time GMV", "on_time_gmv_mean", "{:.1f}"),
           ("On-time fulfilment rate  |T|/|G|", "on_time_order_rate_mean", "{:.4f}"),
           ("Completion rate  |D|/|G|", "completion_rate_mean", "{:.4f}")]

CITY_DIR = {"meituan-ssn": "meituan_ssn", "meituan-sln": "meituan_sln",
            "meituan-san": "meituan_san", "meituan-lsn": "meituan_lsn",
            "meituan-asn": "meituan_asn", "lade_cq": "lade_cq", "lade_sh": "lade_sh",
            "hongkong": "hongkong", "london": "london", "grubhub": "grubhub",
            "shenzhen": "shenzhen",
            # The two scenarios added for table 0's five-scenario set. Their
            # instance directories are the two-letter codes, so city and
            # directory coincide and DIR_CITY inverts cleanly.
            "se": "se", "sg": "sg",
            # se20: Stockholm's order stream on a 20-drone fleet, between se's
            # 10 (where the problem degenerates into a pure ordering rule and
            # R1 beats R5) and the 40 it was first generated with (where the
            # rules already take 94.7% of the ceiling).
            "se20": "se20",
            # table 1.3: the same four moves at +/-25%
            "meituan-sln25": "meituan_sln25", "meituan-san25": "meituan_san25",
            "meituan-lsn25": "meituan_lsn25", "meituan-asn25": "meituan_asn25",
            }
DIR_CITY = {v: k for k, v in CITY_DIR.items()}

# Table 1.4's charge arms are NOT cities. battery_charge_seconds is a scenario
# constant, so every arm runs on meituan-ssn's instances and lands in the same
# records directory; only the resolved config tells them apart. finished()
# relabels an off-base arm as one of these pseudo-cities, which is what puts the
# charge into the cell key -- without it the four arms form one
# (algo, city, rank, critic) group and the collector refuses them all as
# ambiguous.
#
# They are deliberately NOT in CITY_DIR. CITY_DIR is inverted into DIR_CITY to
# turn a records directory back into a city name, and three keys all mapping to
# "meituan_ssn" made that inversion pick the last of them, so every real ssn run
# came back labelled "charge90" and `--cities meituan-ssn` matched 0 of 446 runs.
CHARGE_CITIES = {"charge15", "charge60", "charge90"}

# SCENARIO_DEFAULT's battery charge constant. A run at this value is a run of
# the scenario the other tables report, so it keeps its own city name.
BASE_CHARGE_SECONDS = 1800

RANK_ROWS = {0: "r=0", 2: "r=2", 4: "r=4 (main)", 8: "r=8"}
CURVE_CITIES = ("meituan-ssn", "meituan-sln", "shenzhen")

# The rank each ROWS row claims. Only a run at that rank may be written to it.
# Without this gate every F2pro run on the two rank-curve cities -- r=0, r=2 and
# r=8 included -- also counted as table 1.1's and 1.2's "rank 4" row, and the
# mean cells, which are rewritten every pass, ended up holding r=8.
MAIN_RANK = {"F2_Separable": 0, "F2pro": 4}


def finished(records: Path) -> list[dict]:
    """Every run under records/ that produced a held-out aggregate.

    Two layouts, because the rule baselines do not train. A learned run writes
    train/<city>/.../<run>/eval_summary.json, the aggregate over the held-out
    instances after training. A rule run is launched with --is-train false and
    --eval-episodes 0, so the whole run IS the held-out pass: it writes
    test/<city>/.../<run>/summary.json and no eval_summary.json at all. The two
    files carry the same aggregate keys (on_time_gmv_mean and friends) over the
    same instance set, so one reader handles both once it looks in both places.
    """
    out = []
    paths = (sorted(records.glob("train/*/*/*/*/*/eval_summary.json"))
             + sorted(records.glob("test/*/*/*/*/*/summary.json")))
    for summary in paths:
        run = summary.parent
        city_dir, algo = run.parts[-5], run.parts[-2]
        cfg = run / "resolved_config.json"
        rank = None
        critic = None
        dd: dict = {}
        if cfg.exists():
            dd = json.loads(cfg.read_text()).get("drone_decision", {})
            key = {"F2_Separable": "order_battery_d2_separable_ppo",
                   "F2pro": "order_battery_f2pro_ppo"}.get(algo)
            # --interaction-rank lands on the TOP-LEVEL drone_decision key, and the
            # policy reads that in preference to its own section, so the section
            # still shows the yaml default. Reading the section alone labelled
            # every rank-sweep run as rank 4 and left table 4.1's other rows empty.
            if "interaction_rank" in dd and dd["interaction_rank"] is not None:
                rank = int(dd["interaction_rank"])
            elif key and key in dd:
                rank = int(dd[key]["interaction_rank"])
        # --critic-type lands on the top-level key the same way --interaction-rank
        # does. Only an explicit override counts: every per-policy section carries
        # its own default, so reading those would label every run.
        critic = dd.get("critic_type")
        # The charge arm, read from the merged config rather than the run name.
        charge = None
        if cfg.exists():
            charge = (json.loads(cfg.read_text()).get("service_times", {})
                      .get("battery_charge_seconds"))
        data = json.loads(summary.read_text())
        if "on_time_gmv_mean" not in data:
            continue
        seed = re.search(r"_seed(\d+)$", run.name)
        city = DIR_CITY.get(city_dir, city_dir)
        if charge is not None and int(charge) != BASE_CHARGE_SECONDS:
            city = f"charge{int(charge) // 60}"
        out.append({"city": city, "algo": algo,
                    "rank": rank, "critic": critic, "name": run.name, "metrics": data,
                    "seed": seed.group(1) if seed else "?",
                    "episodes": data.get("episodes")})
    return out


def patch(path: Path, section: str, row_label: str, column: str, value: str,
          scope: str | None = None, overwrite: bool = False) -> bool:
    """Write one cell of one markdown table. Returns False if it was not found.

    ``scope`` is an enclosing heading that must have been seen first -- table 4.1
    repeats "On-time GMV" once per city, so the city heading is what tells the
    two apart. ``overwrite`` is for the cells that are recomputed every pass:
    a per-seed cell is written once and never moved, a mean is not.
    """
    lines = path.read_text().splitlines()
    in_section, in_scope, header, changed = False, scope is None, None, False
    for i, line in enumerate(lines):
        if line.startswith("#"):
            title = line.strip("# ").strip()
            if scope is not None and not line.startswith("###"):
                in_scope = title.startswith(scope)
            in_section = in_scope and (title == section or title.startswith(section + " --"))
            header = None
            continue
        if not in_section or not line.startswith("|"):
            continue
        # a row label can contain an escaped pipe -- "F2pro (o \| l \| b)" -- so
        # splitting on every pipe would cut the label into three cells and shift
        # every column after it. Only unescaped pipes are separators.
        cells = re.split(r"(?<!\\)\|", line)
        names = [c.strip() for c in cells[1:-1]]
        if names and names[0] in ("method", "rank", "metric"):
            header = names
            continue
        if header is None or names[0] != row_label or column not in header:
            continue
        j = header.index(column)
        if not overwrite and cells[1 + j].strip() not in ("--", ""):
            return False                      # never overwrite a filled cell
        width = len(cells[1 + j])
        cells[1 + j] = f" {value} ".ljust(max(width, len(value) + 2))
        lines[i] = "|".join(cells)
        changed = True
        break
    if changed:
        path.write_text("\n".join(lines) + "\n")
    return changed


def agg(values: list[float], fmt: str) -> tuple[str, str]:
    """The combined cell and the separate sd cell, for n seeds.

    One seed gets the mean alone and an empty sd: the sd over one run does not
    exist, and the instance spread that does is a different quantity.
    """
    m = fmt.format(mean(values))
    if len(values) < 2:
        return m, "--"
    s = fmt.format(stdev(values))
    return f"{m} \u00b1 {s}", s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="records_f30")
    ap.add_argument("--dry-run", action="store_true")
    # Fill one city's column and leave the rest alone. The mean cells are
    # recomputed from every seed present and rewritten on each pass, so a pass
    # run while another city is mid-sweep replaces that city's three-seed cell
    # with a one-seed number. Restricting the pass is how a finished column
    # lands without disturbing one still in flight.
    ap.add_argument("--cities", nargs="+", default=None,
                    help="only fill these cities (default: every city found)")
    args = ap.parse_args()

    runs = finished(ROOT / args.records)
    if args.cities:
        keep = set(args.cities)
        unknown = sorted(keep - set(CITY_DIR) - CHARGE_CITIES)
        if unknown:
            raise SystemExit(f"unknown city/cities {unknown}; "
                             f"known: {sorted(set(CITY_DIR) | CHARGE_CITIES)}")
        before = len(runs)
        runs = [r for r in runs if r["city"] in keep]
        print(f"--cities {' '.join(sorted(keep))}: {len(runs)} of {before} runs\n")
    print(f"{len(runs)} finished runs under {args.records}\n")

    # Two runs of the same seed that land in the same cell with different
    # numbers mean the cell is ambiguous, and whichever the glob happened to
    # reach first would win in silence. Refuse instead. Different seeds in one
    # group are the point, not a clash.
    groups: dict[tuple, list] = {}
    for r in runs:
        groups.setdefault((r["algo"], r["city"], r["rank"], r["critic"]), []).append(r)
    clashes = {}
    for k, v in groups.items():
        by_seed: dict[str, set] = {}
        for x in v:
            by_seed.setdefault(x["seed"], set()).add(round(x["metrics"]["on_time_gmv_mean"], 3))
        if any(len(s) > 1 for s in by_seed.values()):
            clashes[k] = v
    if clashes:
        for (algo, city, rank, critic), v in clashes.items():
            print(f"  AMBIGUOUS {algo} {city} rank={rank} critic={critic}:")
            for x in v:
                print(f"      seed {x['seed']}  {x['metrics']['on_time_gmv_mean']:10.1f}  {x['name']}")
        raise SystemExit("two runs of one seed map to one cell; name or re-run them before filling")

    written = 0

    # the per-seed cells, written once each
    for r in runs:
        targets = ([] if r["rank"] != MAIN_RANK.get(r["algo"], r["rank"])
                   else list(ROWS.get((r["algo"], r["critic"]), [])))
        # F2pro's rank curve: every F2pro run on the two rank-curve cities is a
        # row of table 4.1, the default-rank ones included -- which is why stage
        # A's r=4 runs do not have to be trained again for it.
        curve = (r["algo"] == "F2pro" and r["rank"] in RANK_ROWS
                 and r["city"] in CURVE_CITIES)
        for metric, field, fmt in METRICS:
            value = fmt.format(r["metrics"][field])
            for fname, row in targets:
                path = TABLES / fname
                if not path.exists():
                    continue
                column = f"{r['city']} s{r['seed']}"
                if args.dry_run:
                    print(f"  would set {fname} / {metric[:26]:26s} / {row[:24]:24s} / {column} = {value}")
                elif patch(path, metric, row, column, value):
                    written += 1
            if curve and metric in ("On-time GMV", "Completion rate  |D|/|G|"):
                sec = "On-time GMV" if metric == "On-time GMV" else "Completion rate"
                if args.dry_run:
                    print(f"  would set table4_1_rank.md / {r['city']} / {sec:16s} / "
                          f"{RANK_ROWS[r['rank']]:11s} / s{r['seed']} = {value}")
                elif patch(TABLES / "table4_1_rank.md", sec, RANK_ROWS[r["rank"]],
                           f"s{r['seed']}", value, scope=r["city"]):
                    written += 1
        print(f"  {r['algo']:22s} {r['city']:14s} rank={str(r['rank']):4s} "
              f"critic={str(r['critic']):9s} s{r['seed']} "
              f"gmv={r['metrics']['on_time_gmv_mean']:9.1f} "
              f"compl={r['metrics']['completion_rate_mean']:.3f} "
              f"(n={r['episodes']} instances, instance sd {r['metrics']['on_time_gmv_std']:.1f})")

    # the mean-and-sd cells, recomputed from every seed present
    print()
    for (algo, city, rank, critic), v in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        seeds = sorted({x["seed"] for x in v})
        curve = algo == "F2pro" and rank in RANK_ROWS and city in CURVE_CITIES
        main = rank == MAIN_RANK.get(algo, rank)
        for metric, field, fmt in METRICS:
            vals = [x["metrics"][field] for x in v]
            combined, sd = agg(vals, fmt)
            for fname, row in (ROWS.get((algo, critic), []) if main else []):
                path = TABLES / fname
                if not path.exists():
                    continue
                sec = f"{metric} -- mean and sd over the three seeds"
                if args.dry_run:
                    print(f"  would set {fname} / {sec[:34]:34s} / {row[:24]:24s} / {city} = {combined}")
                elif patch(path, sec, row, city, combined, overwrite=True):
                    written += 1
            if curve and metric in ("On-time GMV", "Completion rate  |D|/|G|"):
                sec = "On-time GMV" if metric == "On-time GMV" else "Completion rate"
                for col, val in (("mean", fmt.format(mean(vals))), ("sd", sd)):
                    if args.dry_run:
                        print(f"  would set table4_1_rank.md / {city} / {sec:16s} / "
                              f"{RANK_ROWS[rank]:11s} / {col} = {val}")
                    elif patch(TABLES / "table4_1_rank.md", sec, RANK_ROWS[rank],
                               col, val, scope=city, overwrite=True):
                        written += 1
        print(f"  mean over {len(seeds)} seed(s) {','.join(seeds):11s} "
              f"{algo:22s} {city:14s} rank={str(rank):4s} critic={str(critic):9s} "
              f"gmv={mean([x['metrics']['on_time_gmv_mean'] for x in v]):9.1f}")

    print(f"\ncells written: {written}")


if __name__ == "__main__":
    main()
