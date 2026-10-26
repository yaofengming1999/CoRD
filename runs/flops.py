"""Multiply-accumulate cost of one dispatch decision, per city and per method.

Counted from the layer widths the code actually builds, against the row counts
runs/menu_size.py measured, so this is the same decisions as table 0 priced
rather than a separate experiment.

Convention: a Linear(m, n) costs 2mn -- one multiply and one add per weight.
Biases and ReLU are O(n) and left out; they are under a percent of any row here.
Everything is the forward pass of the actor only: the critic and the backward
pass are a training cost, while what the factorisation claims to reduce is the
cost of taking one decision.

One caveat the table states rather than hides. F3 differs from F2 twice over:
it scores one row per triple AND its per-row network is deeper (four input
projections and four score MLPs against one projection and one linear head).
So the "F3 as built" column is what the baseline costs, and the "N rows at F2's
per-row cost" column isolates the factorisation from the head depth.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

H = 128                      # hidden_dim, the yaml default for every method here


def lin(m: int, n: int) -> int:
    return 2 * m * n


def mlp(m: int) -> int:
    """projection(m): Linear(m,H) -> ReLU -> Linear(H,H) -> ReLU."""
    return lin(m, H) + lin(H, H)


# ---- per-decision fixed cost, and per-row cost, read off the modules ----
CONTEXT_DIM = 12             # featmask lossless
O_DIM, L_DIM, B_DIM = 7, 10, 3
OL_DIM = 15
FLAT_KIND, FLAT_ORDER, FLAT_BATTERY, FLAT_RETURN = 3, 5, 5, 13

COST = {
    # SLOTS-S: three blocks, each a projection plus a scalar head, and the two
    # rank heads on the order and pad blocks
    "slots_s": {
        "once": mlp(CONTEXT_DIM) + lin(H, 3),
        "o": mlp(O_DIM) + lin(2 * H, 1),
        "l": mlp(L_DIM) + lin(2 * H, 1),
        "b": mlp(B_DIM) + lin(2 * H, 1),
        "rank_o": lambda r: lin(2 * H, r),
        "rank_l": lambda r: lin(2 * H, r),
        "entry": lambda r: 2 * r + 4,
    },
    # SLOTS-G: the (o,l) pair is one row, the battery the other block
    "slots_g": {
        "once": mlp(CONTEXT_DIM) + lin(H, 3),
        "ol": mlp(OL_DIM) + lin(2 * H, 1),
        "b": mlp(B_DIM) + lin(2 * H, 1),
        "entry": 4,
    },
    # F3: one row per triple, and a deeper per-row network
    "f3": {
        "once": mlp(CONTEXT_DIM),
        "row": (mlp(FLAT_KIND) + mlp(FLAT_ORDER) + mlp(FLAT_BATTERY) + mlp(FLAT_RETURN)
                + mlp(3 * H) + mlp(4 * H) + mlp(5 * H) + mlp(5 * H) + lin(H, 1)),
    },
}


def per_decision(n_o, n_l, n_b, n_ol, n, rank_s=4):
    s = COST["slots_s"]
    slots_s = (s["once"] + n_o * (s["o"] + s["rank_o"](rank_s))
               + n_l * (s["l"] + s["rank_l"](rank_s)) + n_b * s["b"]
               + n * s["entry"](rank_s))
    g = COST["slots_g"]
    slots_g = g["once"] + n_ol * g["ol"] + n_b * g["b"] + n * g["entry"]
    f3 = COST["f3"]["once"] + n * COST["f3"]["row"]
    # the factorisation alone: N rows priced at SLOTS-G's per-row cost
    flat_at_g = g["once"] + n * g["ol"]
    return slots_s, slots_g, f3, flat_at_g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="mdtable2/menu_size.csv")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--out-csv", default="mdtable2/flops.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    per = collections.defaultdict(list)
    with open(root / args.csv) as fh:
        for r in csv.DictReader(fh):
            per[r["city"]].append(per_decision(
                int(r["n_o"]), int(r["n_l"]), int(r["n_b"]), int(r["n_ol"]),
                int(r["n_triples"]), rank_s=args.rank))

    print(f"per-row cost: SLOTS-G {COST['slots_g']['ol'] / 1e3:.1f}k  "
          f"SLOTS-S order {(COST['slots_s']['o'] + COST['slots_s']['rank_o'](args.rank)) / 1e3:.1f}k  "
          f"F3 {COST['f3']['row'] / 1e3:.1f}k MAC-pairs\n")
    print(f"{'city':14s}{'SLOTS-S med':>13s}{'SLOTS-G med':>13s}{'F3 med':>12s}"
          f"{'F3/S med':>10s}{'F3 worst':>12s}{'F3/S worst':>12s}")
    out = []
    for city, v in per.items():
        a = np.asarray(v, dtype=float)
        med = np.median(a, axis=0)
        p90 = np.percentile(a, 90, axis=0)
        worst = a.max(axis=0)
        j = a[:, 2].argmax()                       # the decision F3 pays most for
        out.append({"city": city,
                    "slots_s_med": med[0], "slots_g_med": med[1], "f3_med": med[2],
                    "flat_at_g_med": med[3],
                    "slots_s_p90": p90[0], "slots_g_p90": p90[1], "f3_p90": p90[2],
                    "slots_s_worst": a[j, 0], "slots_g_worst": a[j, 1], "f3_worst": a[j, 2],
                    "flat_at_g_worst": a[j, 3]})
        print(f"{city:14s}{med[0]/1e6:11.1f}M{med[1]/1e6:11.1f}M{med[2]/1e6:10.1f}M"
              f"{med[2]/med[0]:9.1f}x{a[j,2]/1e6:10.1f}M{a[j,2]/a[j,0]:11.1f}x")
    with open(root / args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)
    print(f"\nwritten {args.out_csv}")


if __name__ == "__main__":
    main()
