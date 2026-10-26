"""How big one dispatch decision is, per city, and what each factorisation leaves.

The menu is built before anything is scored: an entry exists when the simulator
says the triple (order, return pad, battery) is flyable, which makes the count a
property of the instance and the feasibility test rather than of the weights.
What the weights change is which states get visited, so the policy is named in
the output and the numbers should be read as "the menus this policy meets".

Three counts per decision, and they are the three rows of table 0:

    N          feasible (order, return pad, battery) triples -- what an actor
               that scores one row per action has to read
    N_ol + N_b what F2 scores: the (order, pad) pairs once each, the batteries
               once each
    N_o + N_l + N_b  what F2pro scores: each block once

Run from the repository root:

    python runs/menu_size.py --cities hongkong london --instances 3 --out menu_size.csv
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runs.eval_pass import build, instance_ids   # noqa: E402

FIELDS = ["city", "policy", "instance", "decision", "t", "launchpad_id",
          "n_o", "n_l", "n_b", "n_ol", "n_triples", "rows_f2", "rows_f2pro"]


@contextmanager
def capturing(policy, sink, meta):
    menu = policy.menu
    real = menu.candidates
    counter = [0]

    def candidates(state, launchpad, *a, **k):
        out = real(state, launchpad, *a, **k)
        counter[0] += 1
        fm = getattr(menu, "factor_menu", None)
        dispatch = [c for c in out if c.kind == "dispatch" and c.battery_id is not None]
        if dispatch:
            n_o = len({c.order_id for c in dispatch})
            n_l = len({c.return_launchpad_id for c in dispatch})
            n_b = len({c.battery_id for c in dispatch})
            n_ol = int(fm.n_ol) if fm is not None else len({(c.order_id, c.return_launchpad_id)
                                                            for c in dispatch})
            sink.append({**meta, "decision": counter[0], "t": int(state.get("time", 0)),
                         "launchpad_id": str(launchpad.get("launchpad_id")),
                         "n_o": n_o, "n_l": n_l, "n_b": n_b, "n_ol": n_ol,
                         "n_triples": len(dispatch),
                         "rows_f2": n_ol + n_b, "rows_f2pro": n_o + n_l + n_b})
        return out

    menu.candidates = candidates
    try:
        yield
    finally:
        menu.candidates = real


def run_city(city, *, seed, instances, device, instance_root, algo):
    config, env, dispatcher = build(city, algo, seed, device=device,
                                    instance_root=instance_root)
    policy = dispatcher.order_battery_policy
    policy.training = False
    rows = []
    for instance_id in instance_ids(instance_root, city)[:instances]:
        config["instance"]["fixed_id"] = instance_id
        state = env.reset(seed=seed)
        with capturing(policy, rows, {"city": city, "policy": algo, "instance": instance_id}):
            done = False
            while not done:
                env.apply_actions(dispatcher.plan(state))
                state, _, done, _ = env.step()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--algorithm", default="F2_Separable")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument("--instance-root", default="data/instances_drone")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for city in args.cities:
            rows = run_city(city, seed=args.seed, instances=args.instances,
                            device=args.device, instance_root=args.instance_root,
                            algo=args.algorithm)
            writer.writerows(rows)
            fh.flush()
            n = np.asarray([r["n_triples"] for r in rows])
            f2 = np.asarray([r["rows_f2"] for r in rows])
            pro = np.asarray([r["rows_f2pro"] for r in rows])
            print(f"  {city:14s} {len(rows):6d} decisions   "
                  f"triples med={np.median(n):7.0f} p90={np.percentile(n, 90):8.0f} max={n.max():8.0f}   "
                  f"F2 med={np.median(f2):5.0f}  F2pro med={np.median(pro):4.0f}", flush=True)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
