"""Small records shared by the F2 policy and its MenuBuilder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class _JointPlanCandidate:
    kind: str
    order_id: str | None
    battery_id: str | None
    return_launchpad_id: str | None
    features: np.ndarray | None  # None when the decision is described by a FactorMenu


@dataclass(frozen=True)
class _JointPlanDecision:
    time: int
    sample: Any
    kind: str
    launchpad_id: str
    drone_id: str | None = None
    order_id: str | None = None
