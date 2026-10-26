from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


JOULES_PER_WH = 3600.0


@dataclass(frozen=True)
class DroneEnergyParams:
    mode: str = "paper"
    speed_mps: float = 50.0 / 3.6
    mass_power_intercept_W: float = 3000.0
    mass_power_slope_W_per_kg: float = 70.0
    wind_linear_coeff: float = 0.5
    wind_beta1: float = -0.0002131
    wind_beta2: float = 0.01722
    wind_beta3: float = -0.5592
    wind_constant: float = 9.595
    # How the drone's airspeed is derived from ground speed + wind for alpha_w.
    #   "vector": wind triangle |v_air|^2 = v_g^2 + w^2 - 2 v_g w cos(theta)
    #             (tailwind cheaper, headwind costlier, crosswind slightly costlier)
    #   "legacy_sin": the pre-2026-09-17 formula v_g - w sin(theta), kept only to
    #             reproduce that behaviour; it made head/tail wind inert and
    #             crosswind *cheaper*, so it is not physical.
    wind_airspeed_model: str = "vector"
    # Clamp airspeed to the range where the wind power polynomial is calibrated
    # and monotone, so extreme winds cannot push it into an unphysical regime.
    wind_airspeed_cap_mps: float = 30.0
    max_payload_kg: float = 10.0
    # The empty mass the power intercept was calibrated at: 3.0 kW for a 15 kg
    # vehicle *including* its battery. Mass above or below this is charged at the
    # payload slope, which is what lets a heavier pack cost power. A drone that
    # reports exactly this mass is charged nothing extra, so every homogeneous
    # instance reproduces its earlier energies bit for bit.
    empty_mass_ref_kg: float = 15.0
    battery_capacity_Wh: float = 720.0
    reserve_ratio: float = 0.20
    safety_margin_Wh: float = 0.0


class DroneEnergyModel:
    """Task-level UAV energy estimator.

    Main model:
        E = alpha_m(payload_kg) * alpha_w(wind) * flight_time_s / 3600

    The active benchmark calibration uses 3.0 kW for the 15 kg empty vehicle
    plus battery and adds 70 W per kg of payload, reaching 3.7 kW at 10 kg. The
    empty mass is therefore represented by the intercept, not added again.

    This follows Gao et al. Eq. (2)-(5), adapted to the simulator's
    reposition / loaded / return flight legs.
    """

    def __init__(self, config: dict[str, Any]):
        self.params = self._params_from_config(config)
        if self.params.mode != "paper":
            raise ValueError(f"Unsupported UAV energy model mode: {self.params.mode!r}. Use 'paper'.")
        if self.params.wind_airspeed_model not in {"vector", "legacy_sin"}:
            raise ValueError("energy_model.uav.wind_airspeed_model must be 'vector' or 'legacy_sin'")

    def estimate_leg_energy(
        self,
        distance_m: float,
        payload_mass_kg: float = 0.0,
        empty_mass_kg: float | None = None,
        weather_factor: float = 1.0,
        wind_speed_mps: float = 0.0,
        wind_theta_deg: float = 0.0,
    ) -> dict[str, Any]:
        distance = max(0.0, float(distance_m))
        payload = max(0.0, float(payload_mass_kg or 0.0))
        ground_speed = self._effective_speed_mps(weather_factor)
        alpha_m_W = self.alpha_m_W(payload, empty_mass_kg)
        alpha_w = self.alpha_w(float(wind_speed_mps or 0.0), float(wind_theta_deg or 0.0), ground_speed)
        airspeed = self.airspeed_mps(float(wind_speed_mps or 0.0), float(wind_theta_deg or 0.0), ground_speed)

        if distance == 0.0:
            return self._leg_result(0.0, payload, 0.0, alpha_m_W, alpha_w, wind_speed_mps, wind_theta_deg, airspeed, ground_speed)

        time_s = distance / ground_speed
        energy_Wh = alpha_m_W * alpha_w * time_s / JOULES_PER_WH
        return self._leg_result(energy_Wh, payload, time_s, alpha_m_W, alpha_w, wind_speed_mps, wind_theta_deg, airspeed, ground_speed)

    def estimate_task_energy(
        self,
        distance_loaded_m: float,
        distance_return_m: float,
        payload_mass_kg: float,
        battery_remaining_Wh: float | None,
        battery_capacity_Wh: float | None,
        distance_reposition_m: float = 0.0,
        max_payload_kg: float | None = None,
        weather_factor: float = 1.0,
        weather_factor_reposition: float | None = None,
        weather_factor_loaded: float | None = None,
        weather_factor_return: float | None = None,
        wind_speed_reposition_mps: float = 0.0,
        wind_theta_reposition_deg: float = 0.0,
        wind_speed_loaded_mps: float = 0.0,
        wind_theta_loaded_deg: float = 0.0,
        wind_speed_return_mps: float = 0.0,
        wind_theta_return_deg: float = 0.0,
        empty_mass_kg: float | None = None,
        safe_return_energy_Wh: float = 0.0,
    ) -> dict[str, Any]:
        battery_capacity = (
            self.default_battery_capacity_Wh
            if battery_capacity_Wh is None
            else float(battery_capacity_Wh)
        )
        battery_remaining = (
            battery_capacity
            if battery_remaining_Wh is None
            else float(battery_remaining_Wh)
        )
        max_payload = self.default_max_payload_kg if max_payload_kg is None else float(max_payload_kg)
        payload = max(0.0, float(payload_mass_kg or 0.0))
        reposition_factor = weather_factor if weather_factor_reposition is None else weather_factor_reposition
        loaded_factor = weather_factor if weather_factor_loaded is None else weather_factor_loaded
        return_factor = weather_factor if weather_factor_return is None else weather_factor_return

        reposition = self.estimate_leg_energy(
            distance_reposition_m,
            0.0,
            empty_mass_kg=empty_mass_kg,
            weather_factor=reposition_factor,
            wind_speed_mps=wind_speed_reposition_mps,
            wind_theta_deg=wind_theta_reposition_deg,
        )
        loaded = self.estimate_leg_energy(
            distance_loaded_m,
            payload,
            empty_mass_kg=empty_mass_kg,
            weather_factor=loaded_factor,
            wind_speed_mps=wind_speed_loaded_mps,
            wind_theta_deg=wind_theta_loaded_deg,
        )
        returning = self.estimate_leg_energy(
            distance_return_m,
            0.0,
            empty_mass_kg=empty_mass_kg,
            weather_factor=return_factor,
            wind_speed_mps=wind_speed_return_mps,
            wind_theta_deg=wind_theta_return_deg,
        )

        task_energy = reposition["E_leg_Wh"] + loaded["E_leg_Wh"] + returning["E_leg_Wh"]
        reserve = self.reserve_energy_Wh(battery_capacity, safe_return_energy_Wh)
        battery_after = battery_remaining - task_energy
        payload_feasible = payload <= max_payload
        energy_feasible = battery_after >= reserve
        capacity_denominator = battery_capacity if battery_capacity > 0 else 1.0

        return {
            "is_feasible": payload_feasible and energy_feasible,
            "payload_feasible": payload_feasible,
            "energy_feasible": energy_feasible,
            "E_task_Wh": task_energy,
            "E_reposition_Wh": reposition["E_leg_Wh"],
            "E_loaded_Wh": loaded["E_leg_Wh"],
            "E_return_Wh": returning["E_leg_Wh"],
            "B_after_Wh": battery_after,
            "B_reserve_Wh": reserve,
            "energy_ratio": task_energy / capacity_denominator,
            "battery_after_ratio": battery_after / capacity_denominator,
            "reserve_after_ratio": (battery_after - reserve) / capacity_denominator,
            "payload_ratio": payload / max_payload if max_payload > 0 else float("inf"),
            "time_reposition_s": reposition["time_s"],
            "time_loaded_s": loaded["time_s"],
            "time_return_s": returning["time_s"],
            "distance_reposition_m": max(0.0, float(distance_reposition_m)),
            "distance_loaded_m": max(0.0, float(distance_loaded_m)),
            "distance_return_m": max(0.0, float(distance_return_m)),
            "power_empty_W": self.alpha_m_W(0.0, empty_mass_kg),
            "power_loaded_W": self.alpha_m_W(payload, empty_mass_kg),
            "alpha_m_empty_W": self.alpha_m_W(0.0, empty_mass_kg),
            "alpha_m_loaded_W": self.alpha_m_W(payload, empty_mass_kg),
            "weather_factor_reposition": reposition_factor,
            "weather_factor_loaded": loaded_factor,
            "weather_factor_return": return_factor,
            "leg_breakdown": {
                "reposition": reposition,
                "loaded": loaded,
                "return": returning,
            },
        }

    def alpha_m_W(self, payload_mass_kg: float, empty_mass_kg: float | None = None) -> float:
        """Hover/cruise power for a payload, optionally at a non-reference empty mass.

        The model is linear in mass, so mass the vehicle carries because of a
        heavier battery is charged exactly like mass it carries because of a
        heavier parcel. The excess is signed: a lighter pack costs less power.

        ``None`` or a non-positive empty mass means "the reference", not "zero
        mass" -- some call sites default the field to 0.0 when a drone does not
        declare one, and reading that as a 15 kg saving would silently cut the
        power of every such flight.
        """
        payload = max(0.0, float(payload_mass_kg or 0.0))
        excess = 0.0
        if empty_mass_kg is not None and float(empty_mass_kg) > 0.0:
            excess = float(empty_mass_kg) - self.params.empty_mass_ref_kg
        # Linear in total mass, and the excess is signed, so a lighter pack really
        # does save power. Payload is clamped at zero on its own (a negative parcel
        # is nonsense) but the sum is not, or a light pack's saving would be
        # swallowed whenever the parcel happens to be weightless.
        power = (
            self.params.mass_power_slope_W_per_kg * (payload + excess)
            + self.params.mass_power_intercept_W
        )
        # unreachable for any sane calibration; it only forbids non-positive power
        return max(1.0, power)

    def airspeed_mps(self, wind_speed_mps: float, wind_theta_deg: float, ground_speed_mps: float | None = None) -> float:
        """Airspeed magnitude for a leg flown at fixed ground speed in a steady wind.

        ``wind_theta_deg`` is the angle between the flight heading and the
        direction the wind blows TOWARD (0 = tailwind, 180 = headwind), as
        produced by ``Weather.uav_wind_state``. The paper model keeps ground
        speed (hence leg time) fixed and lets wind act only through alpha_w.
        """
        ground_speed = self.params.speed_mps if ground_speed_mps is None else float(ground_speed_mps)
        wind = max(0.0, float(wind_speed_mps or 0.0))
        theta_rad = math.radians(float(wind_theta_deg or 0.0))
        if self.params.wind_airspeed_model == "legacy_sin":
            return max(1e-9, ground_speed - wind * math.sin(theta_rad))
        airspeed = math.sqrt(max(0.0, ground_speed * ground_speed + wind * wind
                                 - 2.0 * ground_speed * wind * math.cos(theta_rad)))
        return min(max(1e-9, airspeed), float(self.params.wind_airspeed_cap_mps))

    def alpha_w(self, wind_speed_mps: float, wind_theta_deg: float, ground_speed_mps: float | None = None) -> float:
        ground_speed = self.params.speed_mps if ground_speed_mps is None else float(ground_speed_mps)
        airspeed = self.airspeed_mps(wind_speed_mps, wind_theta_deg, ground_speed)
        numerator = self._wind_power_polynomial(airspeed)
        denominator = self._wind_power_polynomial(ground_speed)
        if abs(denominator) < 1e-12:
            return 1.0
        return max(0.0, numerator / denominator)

    def _wind_power_polynomial(self, speed_mps: float) -> float:
        v = float(speed_mps)
        return (
            self.params.wind_linear_coeff * v
            + self.params.wind_beta1 * v**3
            + self.params.wind_beta2 * v**2
            + self.params.wind_beta3 * v
            + self.params.wind_constant
        )

    def reserve_energy_Wh(
        self,
        battery_capacity_Wh: float | None = None,
        safe_return_energy_Wh: float = 0.0,
    ) -> float:
        capacity = (
            self.default_battery_capacity_Wh
            if battery_capacity_Wh is None
            else float(battery_capacity_Wh)
        )
        return max(
            self.params.reserve_ratio * capacity,
            self.params.safety_margin_Wh,
            float(safe_return_energy_Wh),
        )

    @property
    def default_empty_mass_kg(self) -> float:
        return 0.0

    @property
    def default_max_payload_kg(self) -> float:
        return self.params.max_payload_kg

    @property
    def default_battery_capacity_Wh(self) -> float:
        return self.params.battery_capacity_Wh

    def _effective_speed_mps(self, weather_factor: float = 1.0) -> float:
        return max(1e-9, self.params.speed_mps * max(1e-9, float(weather_factor or 1.0)))

    def _leg_result(
        self,
        energy_Wh: float,
        payload_kg: float,
        time_s: float,
        alpha_m_W: float,
        alpha_w: float,
        wind_speed_mps: float,
        wind_theta_deg: float,
        airspeed_mps: float,
        ground_speed_mps: float,
    ) -> dict[str, Any]:
        return {
            "E_leg_Wh": energy_Wh,
            "payload_mass_kg": payload_kg,
            "time_s": time_s,
            "power_W": alpha_m_W,
            "alpha_m_W": alpha_m_W,
            "alpha_w": alpha_w,
            "wind_speed_mps": float(wind_speed_mps or 0.0),
            "wind_theta_deg": float(wind_theta_deg or 0.0),
            "airspeed_mps": airspeed_mps,
            "ground_speed_mps": ground_speed_mps,
            "effective_speed_mps": ground_speed_mps,
        }

    def _params_from_config(self, config: dict[str, Any]) -> DroneEnergyParams:
        uav_cfg = config.get("vehicles", {}).get("uavs", {})
        energy_cfg = config.get("energy_model", {}).get("uav", {})
        defaults = DroneEnergyParams()

        def pick_float(*values: Any, fallback: float) -> float:
            for value in values:
                if value is not None:
                    return float(value)
            return float(fallback)

        return DroneEnergyParams(
            mode=str(energy_cfg.get("mode") or defaults.mode).lower(),
            speed_mps=pick_float(
                energy_cfg.get("speed_mps"),
                uav_cfg.get("speed_mps"),
                fallback=defaults.speed_mps,
            ),
            empty_mass_ref_kg=pick_float(
                energy_cfg.get("empty_mass_ref_kg"), fallback=defaults.empty_mass_ref_kg
            ),
            mass_power_intercept_W=pick_float(
                energy_cfg.get("mass_power_intercept_W"),
                fallback=defaults.mass_power_intercept_W,
            ),
            mass_power_slope_W_per_kg=pick_float(
                energy_cfg.get("mass_power_slope_W_per_kg"),
                fallback=defaults.mass_power_slope_W_per_kg,
            ),
            wind_linear_coeff=pick_float(energy_cfg.get("wind_linear_coeff"), fallback=defaults.wind_linear_coeff),
            wind_beta1=pick_float(energy_cfg.get("wind_beta1"), fallback=defaults.wind_beta1),
            wind_beta2=pick_float(energy_cfg.get("wind_beta2"), fallback=defaults.wind_beta2),
            wind_beta3=pick_float(energy_cfg.get("wind_beta3"), fallback=defaults.wind_beta3),
            wind_constant=pick_float(energy_cfg.get("wind_constant"), fallback=defaults.wind_constant),
            wind_airspeed_model=str(energy_cfg.get("wind_airspeed_model") or defaults.wind_airspeed_model).lower(),
            wind_airspeed_cap_mps=pick_float(energy_cfg.get("wind_airspeed_cap_mps"), fallback=defaults.wind_airspeed_cap_mps),
            max_payload_kg=pick_float(
                energy_cfg.get("max_payload_kg"),
                uav_cfg.get("max_payload_kg"),
                fallback=defaults.max_payload_kg,
            ),
            battery_capacity_Wh=pick_float(
                energy_cfg.get("battery_capacity_Wh"),
                uav_cfg.get("battery_capacity"),
                fallback=defaults.battery_capacity_Wh,
            ),
            reserve_ratio=pick_float(energy_cfg.get("reserve_ratio"), fallback=defaults.reserve_ratio),
            safety_margin_Wh=pick_float(energy_cfg.get("safety_margin_Wh"), fallback=defaults.safety_margin_Wh),
        )
