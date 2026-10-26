from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def config_dir() -> Path:
    """Absolute path to drones/yamls/, regardless of CWD."""
    # config.py lives at drones/simulator/utils/config.py
    # parents: utils -> simulator -> drones
    return Path(__file__).resolve().parent.parent.parent / "yamls"


def simulator_config_dir() -> Path:
    return config_dir() / "simulator"


def algorithm_config_dir() -> Path:
    return config_dir() / "algorithm"


def resolve_simulator_config_path(name_or_path: str) -> Path:
    """Resolve a simulator/scenario config name to an absolute path.

    Lookup order:
      1) literal path (relative to CWD or absolute)
      2) yamls/simulator/<name>
      3) yamls/simulator/<name>.yaml
      4) yamls/simulator/scenario_<name>.yaml
      5) legacy yamls/<name>.yaml paths
    """
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()

    base = simulator_config_dir()
    candidates = [
        base / name_or_path,
        base / f"{name_or_path}.yaml",
        base / f"scenario_{name_or_path}.yaml",
        config_dir() / name_or_path,
        config_dir() / f"{name_or_path}.yaml",
        config_dir() / f"scenario_{name_or_path}.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Config not found: '{name_or_path}'. Tried: "
        + ", ".join(str(c) for c in [p, *candidates])
    )


def resolve_algorithm_config_path(name_or_path: str) -> Path:
    """Resolve an algorithm config name to an absolute path."""
    p = Path(name_or_path)
    if p.is_file():
        return p.resolve()

    base = algorithm_config_dir()
    candidates = [
        base / name_or_path,
        base / f"{name_or_path}.yaml",
        base / f"algorithm_{name_or_path}.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    raise FileNotFoundError(
        f"Algorithm config not found: '{name_or_path}'. Tried: "
        + ", ".join(str(c) for c in [p, *candidates])
    )


def resolve_config_path(name_or_path: str) -> Path:
    """Backward-compatible alias for simulator config resolution."""
    return resolve_simulator_config_path(name_or_path)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    simulator_path: Optional[str] = None,
    algorithm_path: Optional[str] = None,
    dispatcher: Optional[str] = None,
) -> dict:
    """Load the built-in simulator defaults and algorithm YAMLs.

    Canonical layout:
      - built-in ``SCENARIO_DEFAULT`` from the instance generator
      - optional simulator YAML override
      - yamls/algorithm/*.yaml

    The returned dict remains merged for the existing simulator and dispatcher
    APIs.
    """
    # Lazy import avoids making the lightweight config module import the full
    # instance-generation stack during module initialization.
    from data.instance_generator_synthetic.run_generator import SCENARIO_DEFAULT

    simulator_source = "scenario_default"
    cfg = deepcopy(SCENARIO_DEFAULT)

    if simulator_path not in {None, "scenario_default"}:
        simulator_override = resolve_simulator_config_path(simulator_path)
        cfg = _deep_merge(cfg, _load_yaml(simulator_override))
        simulator_source = str(simulator_override)

    algorithm_base = resolve_algorithm_config_path("default")
    cfg = _deep_merge(cfg, _load_yaml(algorithm_base))

    selected_algorithm = algorithm_path
    if selected_algorithm is None and dispatcher is not None:
        try:
            selected_algorithm = str(resolve_algorithm_config_path(dispatcher))
        except FileNotFoundError:
            selected_algorithm = None

    if selected_algorithm is not None:
        algorithm_override = resolve_algorithm_config_path(selected_algorithm)
        if algorithm_override != algorithm_base:
            cfg = _deep_merge(cfg, _load_yaml(algorithm_override))

    cfg.setdefault("_config_sources", {})
    cfg["_config_sources"] = {
        "simulator_default": "scenario_default",
        "simulator": simulator_source,
        "algorithm_default": str(algorithm_base),
        "algorithm": (
            str(resolve_algorithm_config_path(selected_algorithm))
            if selected_algorithm is not None
            else str(algorithm_base)
        ),
    }
    return cfg
