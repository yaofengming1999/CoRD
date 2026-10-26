#!/usr/bin/env python
"""Render an interactive HTML replay from a saved episode recording."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from visualization import load_episode_record, render_episode_html


def _slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"


def resolve_record_dir(
    explicit: str | None,
    city: str | None,
    split: str,
    dispatcher: str,
    time_advancement: str | None,
    records_root: str,
) -> Path:
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_absolute() else (_REPO_ROOT / path)

    records_base = _REPO_ROOT / records_root
    if city is None:
        city = resolve_latest_city(records_base, split, dispatcher, time_advancement)

    base_candidates = _record_parent_candidates(
        records_base,
        city,
        split,
        dispatcher,
        time_advancement,
    )
    existing = [base for base in base_candidates if base.exists()]
    if not existing:
        raise FileNotFoundError(
            "No recordings found under any of: "
            + ", ".join(str(base) for base in base_candidates)
        )

    candidates = [
        path
        for base in existing
        for path in base.iterdir()
        if path.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No recording directories found under any of: "
            + ", ".join(str(base) for base in existing)
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_latest_city(
    records_base: Path,
    split: str,
    dispatcher: str,
    time_advancement: str | None,
) -> str:
    candidates: list[tuple[float, Path]] = []
    if records_base.exists():
        for city_dir in records_base.iterdir():
            if not city_dir.is_dir():
                continue
            run_parents = [
                path for path in _record_parent_candidates(
                    records_base,
                    city_dir.name,
                    split,
                    dispatcher,
                    time_advancement,
                )
                if path.exists()
            ]
            run_dirs = [
                path
                for run_parent in run_parents
                for path in run_parent.iterdir()
                if path.is_dir()
            ]
            if run_dirs:
                latest_run = max(run_dirs, key=lambda path: path.stat().st_mtime)
                candidates.append((latest_run.stat().st_mtime, city_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No city-level recordings found under {records_base}/{split}/<city>/.../{dispatcher}"
        )
    return max(candidates, key=lambda item: item[0])[1].name


def _record_parent_candidates(
    records_base: Path,
    city: str,
    split: str,
    dispatcher: str,
    time_advancement: str | None,
) -> list[Path]:
    split_dir = records_base / _slug(split) / city / "drone_only"
    if time_advancement is not None:
        return [split_dir / time_advancement / dispatcher]
    if split_dir.exists():
        return [
            time_dir / dispatcher
            for time_dir in split_dir.iterdir()
            if time_dir.is_dir()
        ]
    return [split_dir / "<time_advancement>" / dispatcher]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an interactive HTML replay from a saved episode recording")
    parser.add_argument("--record-dir", default=None, help="Recording directory to replay")
    parser.add_argument("--city", default=None, help="City key used when --record-dir is omitted; defaults to latest matching city")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Record split used when --record-dir is omitted")
    parser.add_argument(
        "--dispatcher",
        default="R1_urgentLower_nearest",
        help="Algorithm key used when --record-dir is omitted",
    )
    parser.add_argument("--time-advancement", default=None, help="Time advancement key used when --record-dir is omitted")
    parser.add_argument("--records-root", default="records", help="Top-level recording directory")
    parser.add_argument("--output", default=None, help="Output HTML path; defaults to <record-dir>/replay.html")
    parser.add_argument("--fps", type=int, default=4, help="Initial playback speed in frames per second")
    parser.add_argument("--skip-frames", type=int, default=1, help="Sample every N recorded frames before building the replay")
    parser.add_argument("--trail-length", type=int, default=12, help="Vehicle trail length in frames")
    args = parser.parse_args()

    record_dir = resolve_record_dir(
        args.record_dir,
        args.city,
        args.split,
        args.dispatcher,
        args.time_advancement,
        args.records_root,
    )
    record = load_episode_record(record_dir)
    if args.output is None:
        output_path = record_dir / "replay.html"
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = _REPO_ROOT / output_path

    saved_to = render_episode_html(
        record,
        output_path=output_path,
        fps=args.fps,
        skip_frames=args.skip_frames,
        trail_length=args.trail_length,
    )
    print(f"Loaded recording from: {record_dir}")
    print(f"Saved replay to: {saved_to}")


if __name__ == "__main__":
    main()
