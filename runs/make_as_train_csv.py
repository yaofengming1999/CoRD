"""Export table 2's raw training curves from the saved run logs.

The output has one row per (method, city, seed, episode), matching the rank
training export while keeping the six table-2 methods together at rank 0.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METHODS = (
    "AS1_MAPPO",
    "AS4_SyncOwnReward",
    "AS2_SingleDronePPO",
    "AS3_GlobalDronesPPO",
    "AS5_TeamReward",
    "F2_Separable",
)
DEFAULT_CITIES = ("hongkong", "meituan-ssn")
SEEDS = (42, 43, 44)
TRAIN = re.compile(r"^\[train\] ep=(?P<episode>\d+)\s+(?P<body>.*)$", re.M)
FIELD = re.compile(r"(?P<key>[a-z_]+)=(?P<value>\S+)")
COLUMNS = (
    "city", "method", "rank", "seed", "episode", "orders", "deliv",
    "completion", "on_time_gmv", "instance_seed", "instance",
)


def log_path(root: Path, city: str, method: str, seed: int) -> Path:
    if method == "F2_Separable":
        return root / "rank" / f"{city}_rank0_s{seed}.log"
    return root / "as_rank0" / f"{city}_{method}_rank0_s{seed}.log"


def rows_of(path: Path, city: str, method: str, seed: int) -> list[dict[str, object]]:
    rows = []
    for match in TRAIN.finditer(path.read_text(errors="replace")):
        values = {item.group("key"): item.group("value") for item in FIELD.finditer(match.group("body"))}
        rows.append({
            "city": city,
            "method": method,
            "rank": 0,
            "seed": seed,
            "episode": int(match.group("episode")),
            "orders": values.get("orders", ""),
            "deliv": values.get("deliv", ""),
            "completion": values.get("completion", ""),
            "on_time_gmv": values.get("on_time_gmv", ""),
            "instance_seed": values.get("seed", ""),
            "instance": values.get("instance", ""),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--output", type=Path,
                        default=Path("mdtable/table2_agent_structure_train.csv"))
    parser.add_argument("--cities", nargs="+", default=list(DEFAULT_CITIES),
                        help="dataset names; HK is accepted as an alias for hongkong")
    parser.add_argument("--check", action="store_true",
                        help="fail if any expected log or episode is missing")
    args = parser.parse_args()

    cities = tuple("hongkong" if city.lower() in {"hk", "hongkong"} else city
                   for city in args.cities)
    rows = []
    missing = []
    for city in cities:
        for method in METHODS:
            for seed in SEEDS:
                path = log_path(args.log_root, city, method, seed)
                if not path.is_file():
                    missing.append(str(path))
                    continue
                rows.extend(rows_of(path, city, method, seed))

    expected = {(city, method, seed, episode)
                for city in cities for method in METHODS for seed in SEEDS
                for episode in range(300)}
    got = {(row["city"], row["method"], row["seed"], row["episode"])
           for row in rows}
    duplicate_count = len(rows) - len(got)
    missing_rows = expected - got
    if args.check and (missing or missing_rows or duplicate_count):
        if missing:
            print("missing logs:")
            print("\n".join(f"  {path}" for path in missing))
        if missing_rows:
            print(f"missing episode rows: {len(missing_rows)}")
        if duplicate_count:
            print(f"duplicate episode rows: {duplicate_count}")
        raise SystemExit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    cells = {(row["city"], row["method"], row["seed"]) for row in rows}
    print(f"written {args.output} ({len(rows)} episode rows, {len(cells)} cells)")
    if missing:
        print(f"missing logs: {len(missing)}")


if __name__ == "__main__":
    main()
