"""Fill table 0's menu-size block from runs/menu_size.py's csv.

Kept apart from collect_f30.py because the two read different things: that one
reads a run's held-out evaluation, this one reads the shape of the decisions a
policy met. The quantity here is not a score and does not belong in a metric
section.
"""

from __future__ import annotations

import argparse
import csv
import collections
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "mdtable2" / "table0_cities.md"
ROWS = {"n_triples": "N, feasible triples",
        "rows_f2": "N_ol + N_b, what F2 scores",
        "rows_f2pro": "N_o + N_l + N_b, what F2pro scores"}


def patch(path: Path, row_label: str, column: str, value: str) -> bool:
    lines = path.read_text().splitlines()
    header, done = None, False
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = re.split(r"(?<!\\)\|", line)
        names = [c.strip() for c in cells[1:-1]]
        if names and names[0] == "quantity":
            header = names
            continue
        if header is None or names[0] != row_label or column not in header:
            continue
        j = header.index(column)
        if cells[1 + j].strip() not in ("--", ""):
            return False
        cells[1 + j] = f" {value} "
        lines[i] = "|".join(cells)
        done = True
        break
    if done:
        path.write_text("\n".join(lines) + "\n")
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="mdtable2/menu_size.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(ROOT / args.csv) as fh:
        for r in csv.DictReader(fh):
            for field in ROWS:
                per[r["city"]][field].append(int(r[field]))
    written = 0
    print(f"{'city':14s}{'decisions':>10s}{'N med/p90/max':>22s}{'F2':>18s}{'F2pro':>16s}"
          f"{'F2 x':>7s}{'pro x':>7s}")
    for city, fields in per.items():
        cell = {}
        for field, label in ROWS.items():
            v = np.asarray(fields[field])
            cell[field] = f"{np.median(v):.0f} / {np.percentile(v, 90):.0f} / {v.max():.0f}"
            if not args.dry_run and patch(TABLE, label, city, cell[field]):
                written += 1
        n = np.asarray(fields["n_triples"])
        f2 = np.asarray(fields["rows_f2"]); pro = np.asarray(fields["rows_f2pro"])
        # the factor is a ratio of medians, not a median of ratios: it is what the
        # mean cost of a decision falls by, which is what the claim is about
        rf2, rpro = np.median(n) / np.median(f2), np.median(n) / np.median(pro)
        for label, value in (("N / (N_ol + N_b), F2", f"{rf2:.1f}x"),
                             ("N / (N_o + N_l + N_b), F2pro", f"{rpro:.1f}x")):
            if not args.dry_run and patch(TABLE, label, city, value):
                written += 1
        print(f"{city:14s}{len(n):10d}{cell['n_triples']:>22s}"
              f"{cell['rows_f2']:>18s}{cell['rows_f2pro']:>16s}{rf2:6.1f}x{rpro:6.1f}x")
    print(f"\ncells written: {written}")


if __name__ == "__main__":
    main()
