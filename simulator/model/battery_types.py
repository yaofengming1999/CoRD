"""The heterogeneous battery ladder: how many pack types exist and what each one is.

A pack type is three numbers that cannot move independently, because they are all
set by how many cells are in the stack:

    capacity_Wh          energy stored
    mass_kg              what the airframe has to carry to carry that energy
    full_charge_seconds  time to fill it from empty at a fixed charger power

So the ladder is generated from one rung count rather than configured three times.
``n_types=1`` reproduces the homogeneous fleet the earlier instances had, exactly:
one rung, at the baseline, with the baseline mass and charge time.

Rungs are symmetric about the baseline in steps of ``capacity_step_Wh``. An odd
count includes the baseline rung; an even count omits it and keeps the pairs, so
the mean capacity of the ladder is the baseline either way and changing
``n_types`` does not silently change how much energy the fleet owns in total.

    n_types=1   720
    n_types=2   540  900
    n_types=3   540  720  900
    n_types=4   360  540  900  1080
    n_types=5   360  540  720  900  1080

Mass moves in a fixed *fraction* per rung rather than a fixed number of kg
(``mass_step_fraction``, 25% by default), because a pack's specific energy is
roughly constant across sizes of one chemistry: twice the cells is twice the mass
and twice the energy. Charge time moves in proportion to capacity, which is what
a fixed charger power gives -- a 900 Wh pack at the power that fills 720 Wh in
1800 s needs 2250 s.

Nothing here reads a config file; ``from_config`` adapts a scenario mapping onto
it so the defaults live in exactly one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

BASE_CAPACITY_WH = 720.0        # the capacity every earlier instance used
CAPACITY_STEP_WH = 180.0        # one rung
BASE_MASS_KG = 4.0              # baseline pack mass, inside the 15 kg empty mass
MASS_STEP_FRACTION = 0.25       # one rung, as a fraction of the baseline mass
BASE_FULL_CHARGE_SECONDS = 1800  # baseline pack, empty to full


@dataclass(frozen=True)
class BatteryType:
    """One rung of the ladder."""

    type_id: str
    capacity_Wh: float
    mass_kg: float
    full_charge_seconds: int
    rung: int  # signed offset from the baseline, in capacity steps

    @property
    def charge_power_W(self) -> float:
        """The charger power this rung implies, for checking the ladder is coherent."""
        return 3600.0 * self.capacity_Wh / max(1, self.full_charge_seconds)


def _rungs(n_types: int) -> list[int]:
    """Signed step offsets, symmetric about 0, ascending.

    Odd counts keep the centre rung; even counts drop it so the offsets still sum
    to zero and the ladder's mean capacity stays at the baseline.
    """
    if n_types < 1:
        raise ValueError(f"n_battery_types must be >= 1, got {n_types}")
    if n_types % 2 == 1:
        half = (n_types - 1) // 2
        return list(range(-half, half + 1))
    half = n_types // 2
    return [k for k in range(-half, half + 1) if k != 0]


def _max_types(base_capacity_Wh: float, capacity_step_Wh: float,
               mass_step_fraction: float) -> int:
    """Largest ``n_types`` whose every rung keeps capacity and mass positive."""
    by_capacity = math.ceil(base_capacity_Wh / capacity_step_Wh) - 1
    by_mass = (math.ceil(1.0 / mass_step_fraction) - 1) if mass_step_fraction > 0 else 10**9
    return max(1, 2 * min(by_capacity, by_mass) + 1)


def battery_type_ladder(
    n_types: int,
    *,
    base_capacity_Wh: float = BASE_CAPACITY_WH,
    capacity_step_Wh: float = CAPACITY_STEP_WH,
    base_mass_kg: float = BASE_MASS_KG,
    mass_step_fraction: float = MASS_STEP_FRACTION,
    base_full_charge_seconds: int = BASE_FULL_CHARGE_SECONDS,
) -> list[BatteryType]:
    """The ladder, ascending in capacity.

    Raises rather than clipping if the requested count would need a non-positive
    capacity or mass: a silently clipped ladder would have fewer distinct types
    than asked for, and every downstream count would be wrong.
    """
    if base_capacity_Wh <= 0.0 or capacity_step_Wh <= 0.0:
        raise ValueError("base_capacity_Wh and capacity_step_Wh must be positive")
    if base_mass_kg <= 0.0:
        raise ValueError("base_mass_kg must be positive")
    if not 0.0 <= mass_step_fraction < 1.0:
        raise ValueError("mass_step_fraction must be in [0, 1)")
    if base_full_charge_seconds < 1:
        raise ValueError("base_full_charge_seconds must be >= 1")

    out: list[BatteryType] = []
    for rung in _rungs(n_types):
        capacity = base_capacity_Wh + rung * capacity_step_Wh
        mass = base_mass_kg * (1.0 + rung * mass_step_fraction)
        if capacity <= 0.0 or mass <= 0.0:
            raise ValueError(
                f"n_battery_types={n_types} needs a rung at {capacity:.0f} Wh / {mass:.2f} kg; "
                f"with base {base_capacity_Wh:.0f} Wh, step {capacity_step_Wh:.0f} Wh and "
                f"mass step {mass_step_fraction:.0%} the ladder only reaches "
                f"{_max_types(base_capacity_Wh, capacity_step_Wh, mass_step_fraction)} types"
            )
        # same charger power on every rung, so time is proportional to capacity
        seconds = max(1, round(base_full_charge_seconds * capacity / base_capacity_Wh))
        out.append(
            BatteryType(
                type_id=f"bt{capacity:.0f}",
                capacity_Wh=float(capacity),
                mass_kg=float(mass),
                full_charge_seconds=int(seconds),
                rung=int(rung),
            )
        )
    return out


def from_config(config: dict, n_types: int) -> list[BatteryType]:
    """``battery_type_ladder`` with the overrides a scenario config may carry.

    Reads ``batteries.heterogeneity`` and falls back to this module's constants,
    so a scenario that says nothing gets the documented ladder.
    """
    spec = (config.get("batteries", {}) or {}).get("heterogeneity", {}) or {}
    base_seconds = (
        (config.get("service_times", {}) or {}).get("battery_charge_seconds")
        or BASE_FULL_CHARGE_SECONDS
    )
    return battery_type_ladder(
        n_types,
        base_capacity_Wh=float(spec.get("base_capacity_Wh", BASE_CAPACITY_WH)),
        capacity_step_Wh=float(spec.get("capacity_step_Wh", CAPACITY_STEP_WH)),
        base_mass_kg=float(spec.get("base_mass_kg", BASE_MASS_KG)),
        mass_step_fraction=float(spec.get("mass_step_fraction", MASS_STEP_FRACTION)),
        base_full_charge_seconds=int(spec.get("base_full_charge_seconds", base_seconds)),
    )
