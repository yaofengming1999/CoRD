"""Per-city order geometry: how far a delivery flies and how long it is allowed.

Both are properties of the instances, which is why they belong in table 0 beside
the fleet and the map rather than in a results table. The distance is the loaded
leg the simulator actually flies -- departure pad to the order's destination
endpoint, after both have been snapped to their clusters -- not the raw
pickup-to-dropoff distance of the source record, since no drone flies that.

The flight time is that distance at the fixed ground speed, so it carries no
extra information; it is reported because minutes are what the deadline is in
and the comparison is the point.

The deadline window separates the cities by provenance. The Meituan family
carries the source table's own delivery_patience, so its window is real and
spread (30 to 95 minutes). Every other city's source had no promised time, so
the window is the shared synthesis, uniform on 20 to 60 minutes. A table that
did not say which was which would invite reading a difference in lateness as a
difference in policy.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path

import numpy as np
import yaml

SPEED_MPS = 13.8888888889          # the fixed ground speed of every drone
R_KM = 6371.0088
# which cities carry a real deadline from their source table
REAL_DEADLINE = {"meituan-ssn", "meituan-sln", "meituan-san", "meituan-lsn", "meituan-asn"}


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1 = a
    la2, lo2 = b
    h = (math.sin(math.radians(la2 - la1) / 2) ** 2
         + math.cos(math.radians(la1)) * math.cos(math.radians(la2))
         * math.sin(math.radians(lo2 - lo1) / 2) ** 2)
    return 2 * R_KM * math.asin(math.sqrt(h))


def city(name: str, root: str, instances: int) -> dict:
    dist, window = [], []
    for d in sorted(glob.glob(f"{root}/{name}/test/*/"))[:instances]:
        pads = {l["launchpad_id"]: (l["current_location"]["lat"], l["current_location"]["lon"])
                for l in yaml.safe_load(open(d + "launchpads.yaml"))}
        for o in yaml.safe_load(open(d + "orders.yaml")):
            dist.append(haversine_km(pads[o["origin_launchpad_id"]],
                                     (o["destination"]["lat"], o["destination"]["lon"])))
            window.append((o["deadline"] - o["created_at"]) / 60.0)
    dist, window = np.asarray(dist), np.asarray(window)
    fly = dist * 1000.0 / SPEED_MPS / 60.0
    return {"city": name, "orders": len(dist),
            "dist_med": np.median(dist), "dist_p90": np.percentile(dist, 90),
            "dist_max": dist.max(),
            "fly_med": np.median(fly), "fly_p90": np.percentile(fly, 90),
            "window_med": np.median(window), "window_min": window.min(),
            "window_max": window.max(),
            # the headroom a delivery actually has: the allowance less the flight
            "slack_med": np.median(window - fly),
            "deadline_source": "source table" if name in REAL_DEADLINE else "uniform 20-60"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", nargs="+", required=True)
    ap.add_argument("--instance-root", default="data/instances_drone")
    ap.add_argument("--instances", type=int, default=5)
    ap.add_argument("--out", default="mdtable2/order_geometry.csv")
    args = ap.parse_args()

    rows = [city(c, args.instance_root, args.instances) for c in args.cities]
    out = Path(__file__).resolve().parents[1] / args.out
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"{'city':14s}{'n':>7s}{'dist km med/p90/max':>24s}{'fly min':>9s}"
          f"{'deadline min':>16s}{'slack':>7s}  source")
    for r in rows:
        print(f"{r['city']:14s}{r['orders']:7d}{r['dist_med']:8.2f}/{r['dist_p90']:6.2f}"
              f"/{r['dist_max']:6.2f}{r['fly_med']:9.1f}"
              f"{r['window_med']:8.0f} [{r['window_min']:.0f},{r['window_max']:.0f}]"
              f"{r['slack_med']:7.1f}  {r['deadline_source']}")
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
