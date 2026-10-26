from __future__ import annotations

import json
import subprocess
from argparse import ArgumentParser, Namespace
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def git_info(repo_root: Path | None = None) -> dict[str, Any]:
    """Commit hash + dirty flag for the repo, so every wandb run carries a
    code-version stamp. Two runs are only safely comparable if this matches
    (or the diff is understood) -- see the 2026-07-07 cr10 pre/post road-network
    -fix mixup this was added to prevent."""
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5, check=True
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except Exception:
        return {"commit": None, "dirty": None}


def add_wandb_args(parser: ArgumentParser, *, default_project: str = "drones") -> None:
    parser.set_defaults(_wandb_default_project=default_project)
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "online", "offline"],
        default=None,
        help="Weights & Biases logging mode. Overrides config.wandb.mode.",
    )
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="", help="Comma-separated WandB tags")
    parser.add_argument("--wandb-notes", default=None)


class WandbLogger:
    def __init__(self, run: Any | None):
        self.run = run
        if self.run is not None:
            for metric_glob in ("train_*", "valid/*"):
                self.run.define_metric(metric_glob, summary="none")
            self.run.define_metric("eval/instance", hidden=True, summary="none")
            self.run.define_metric(
                "eval_*",
                step_metric="eval/instance",
                summary="none",
            )

    @property
    def enabled(self) -> bool:
        return self.run is not None

    def log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(_flatten_for_wandb(data), step=step)

    def summary_update(self, data: dict[str, Any]) -> None:
        if self.run is None:
            return
        self.run.summary.update(_flatten_for_wandb(data))

    def log_artifact(self, path: Path, *, name: str, artifact_type: str) -> None:
        if self.run is None or not path.exists():
            return
        import wandb

        artifact = wandb.Artifact(name=_artifact_name(name), type=artifact_type)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def init_wandb(
    args: Namespace,
    *,
    config: dict[str, Any],
    output_dir: Path | None,
    job_type: str,
    extra_config: dict[str, Any] | None = None,
    default_name: str | None = None,
    default_group: str | None = None,
    default_tags: list[str] | None = None,
) -> WandbLogger:
    wandb_cfg = config.get("wandb", {})
    if not isinstance(wandb_cfg, dict):
        wandb_cfg = {}
    mode = str(getattr(args, "wandb_mode", None) or wandb_cfg.get("mode", "disabled"))
    if mode == "disabled":
        return WandbLogger(None)

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB logging was requested but wandb is not installed. "
            "Install it in the drones environment with: pip install wandb"
        ) from exc

    configured_tags = [
        tag.strip()
        for tag in str(getattr(args, "wandb_tags", None) or wandb_cfg.get("tags", "") or "").split(",")
        if tag.strip()
    ]
    tags = _dedupe_tags([*(default_tags or []), *configured_tags])
    run_config = {
        "cli": vars(args),
        "runtime": {
            "output_dir": str(output_dir) if output_dir is not None else None,
            "job_type": job_type,
        },
        "config": config,
        "git": git_info(),
    }
    if extra_config:
        run_config["extra"] = extra_config

    run = wandb.init(
        project=(
            getattr(args, "wandb_project", None)
            or wandb_cfg.get("project")
            or getattr(args, "_wandb_default_project", "drones")
        ),
        entity=getattr(args, "wandb_entity", None) or wandb_cfg.get("entity"),
        name=getattr(args, "wandb_name", None) or wandb_cfg.get("name") or default_name,
        group=getattr(args, "wandb_group", None) or wandb_cfg.get("group") or default_group,
        tags=tags,
        notes=getattr(args, "wandb_notes", None) or wandb_cfg.get("notes"),
        mode=mode,
        job_type=job_type,
        config=_jsonable(run_config),
        dir=str(output_dir) if output_dir is not None else None,
    )
    return WandbLogger(run)


def _flatten_for_wandb(data: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_for_wandb(value, prefix=f"{name}/"))
        elif _is_scalar(value):
            flat[name] = value
        elif value is not None:
            flat[name] = str(value)
    return flat


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        cleaned = str(tag).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(v) for v in value]
        return str(value)


def _artifact_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
