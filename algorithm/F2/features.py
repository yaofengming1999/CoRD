"""F2 menu and feature construction (``MenuBuilder``).

Everything between the raw simulator state and the tensors the learner sees:
which (order, return launchpad, battery) triplets are feasible at a launchpad,
their features in the factored (o,l)|b form (``FactorMenu``) or as expanded
rows, the 23-dim decision context, the pre-commitment global state used by the
centralised critics, and the small feasibility / pruning helpers those need.
Pure construction: no learning state. The policy owns one instance
(``F2Policy.menu``); AS*/L*/F1 reach the same helpers through it.

Menu semantics (per launchpad p and step t):
  candidates(...)  -> [dispatch (o, l, b) for every feasible triplet] + [defer] + [stop]
                      and, for the olb layout, ``self.factor_menu`` with
                      ol rows [N_ol, 26 raw -> 22], battery rows [N_b + 1, 9 raw -> 5]
                      and the (kind, ol_idx, b_idx) index grid; feasibility is
                      charge_b - E_task(o, l) >= reserve on that grid.
  context(...)     -> 23-dim row: 16 base (time, fleet ratios, pad queue / drones /
                      batteries / supply-demand gap, terminal lock, bias) + 5 joint-plan
                      (this step's committed departures / projected occupancy) + 2 sequence.
  global_state(s)  -> [P, 23] pre-commitment contexts of the pads with a waiting drone.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from algorithm.F2.menu import (
    BATTERY_DIM,
    B_RAW_DIM,
    FLAT_DIM,
    OL_RAW_DIM,
    _RAW_E_TASK,
    _RAW_TARGET,
    ORDER_DIM,
    ORDER_KEEP,
    TARGET_DIM,
    TARGET_KEEP,
    FactorMenu,
    city_frame,
    frame_xy,
    ol_from_raw,
    reform_flat_to_olb,
)
from algorithm.F2 import featmask
from algorithm.common.geometry_utils import distance_m, safe_ratio, urgency
from simulator.model.enums import BatteryStatus, OrderStatus, VehicleStatus
from simulator.model.types import StateDict

from algorithm.F2.policy_types import _JointPlanCandidate


class MenuBuilder:
    """Candidate menus, features and contexts for one policy instance."""

    def __init__(
        self,
        config: dict,
        *,
        layout: str,
        construction: str,
        collapse_return: bool = False,
        return_feature_mode: str = "order_only",
        context_dim: int,
        max_candidates: int,
        pruning: bool,
        urgent_order_multiplier: float,
        return_corridor_km: float,
    ):
        self.config = config
        self.layout = layout                      # "olb" factored | "flat" expanded rows
        self.construction = construction          # "factored" | "triplet" (reference builder)
        # F0: collapse the return axis so one dispatch row is one (order, battery)
        # pair. The l that survives is the energy-minimal one, which is the plan
        # the simulator's own feasibility test uses when an action names no pad.
        self.collapse_return = bool(collapse_return)
        # With collapse_return, what of the collapsed pad l* the dispatch row keeps:
        #   full        everything, as F2's row has it (the pad l* is not the one the
        #               return agent picks on 72% of Hong Kong flights, so its queue
        #               state is wrong more often than right)
        #   route_only  drop l*'s queue state, keep E_task and d_ret -- those are
        #               minima over l, i.e. properties of the ORDER (the cheapest way
        #               to serve it), and they are what the battery choice needs
        #   order_only  drop those too; measured 5% below route_only on Hong Kong,
        #               because without E_task the agent cannot size a battery at all
        if str(return_feature_mode) not in ("full", "route_only", "order_only"):
            raise ValueError(f"unknown return_feature_mode {return_feature_mode!r}")
        self.return_feature_mode = str(return_feature_mode)
        self.context_dim = int(context_dim)
        # ``context_dim`` is the raw width the context assertion checks; the mask
        # then selects which of those columns the networks are given.
        self.feat_mask = featmask.active()
        self.max_candidates = int(max_candidates)
        self.pruning = bool(pruning)
        self.urgent_order_multiplier = float(urgent_order_multiplier)
        self.return_corridor_km = float(return_corridor_km)
        # simulator hooks, bound by runs/utils/experiment_helpers.build_dispatcher
        self.candidate_feasible_fn: Callable[[dict, dict], bool] | None = None
        self.candidate_details_fn: Callable[[dict, dict], dict] | None = None
        self.route_energy_fn: Callable[[dict], dict | None] | None = None      # energy once per (order, return pad)
        self.battery_state_fn: Callable[[dict, dict], dict | None] | None = None  # battery state once per battery
        # per-step / per-episode working state
        self.frame = None                          # city-centred coordinate frame, refreshed per step
        self.factor_menu: FactorMenu | None = None  # set by the factored builder (olb)
        self.feature_rows = 0                      # rows actually scored by the network this episode
        self.active_order_by_launchpad: dict[str, str] = {}  # terminal lock: pad -> order kept for its last drone

    def begin_step(self, state: StateDict) -> None:
        self.frame = city_frame(state)

    def reset_episode(self) -> None:
        self.active_order_by_launchpad = {}
        self.feature_rows = 0
        self.factor_menu = None

    def candidates(
        self,
        state: StateDict,
        launchpad: dict,
        used_orders: set[str],
        used_batteries: set[str],
        current_time: int,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> list[_JointPlanCandidate]:
        if (
            self.construction == "factored"
            and self.route_energy_fn is not None
            and self.battery_state_fn is not None
        ):
            return self._candidates_factored(
                state, launchpad, used_orders, used_batteries, current_time,
                planned_targets, planned_departures,
            )
        return self._candidates_triplet(
            state, launchpad, used_orders, used_batteries, current_time,
            planned_targets, planned_departures,
        )

    def _eligible_menu_inputs(
        self,
        state: StateDict,
        launchpad: dict,
        used_orders: set[str],
        used_batteries: set[str],
        current_time: int,
        planned_departures: dict[str, int],
    ) -> tuple[list[str], list[str], str | None]:
        """Eligible order ids (queue order, pruned) and battery ids at this pad."""
        launchpad_id = str(launchpad.get("launchpad_id"))
        orders = state.get("orders", {})
        batteries = state.get("batteries", {})
        active_order_id = self.active_order_by_launchpad.get(launchpad_id)
        eligible_order_ids = [
            str(order_id)
            for order_id in launchpad.get("order_queue", [])
            if str(order_id) not in used_orders
            and str(order_id) in orders
            and orders[str(order_id)].get("status") == OrderStatus.PENDING.value
            and (active_order_id is None or str(order_id) == active_order_id)
        ]
        eligible_order_ids = self.prune_order_ids(
            state, launchpad, eligible_order_ids, current_time, planned_departures,
        )
        battery_ids = [
            str(battery_id)
            for battery_id in (
                list(launchpad.get("charged_battery_ids", []))
                + list(launchpad.get("charging_battery_ids", []))
            )
            if str(battery_id) not in used_batteries and str(battery_id) in batteries
        ]
        return eligible_order_ids, battery_ids, active_order_id

    def _candidates_factored(
        self,
        state: StateDict,
        launchpad: dict,
        used_orders: set[str],
        used_batteries: set[str],
        current_time: int,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> list[_JointPlanCandidate]:
        """Factored menu: same rows and order as the triplet loop, built from
        per-(order, return pad) and per-battery quantities.

        Per order o:   routes r(o,l) = {E_task, E_return, distances, time, ok}  (one energy call per (o,l))
        Per battery b: charge_b, capacity_b, reserve_b                          (one state call per b)
        (o,l) kept only if E_task(o,l) <= max_b (charge_b - reserve_b)          (exact pad-level bound)
        feasible[o,l,b] = ok(o,l) and charge_b - E_task(o,l) >= reserve_b        (scalar grid test)
        Rows are emitted in the triplet loop's order (o-major, then b, then l by id),
        followed by one defer row per order and the stop row.
        """
        launchpad_id = str(launchpad.get("launchpad_id"))
        orders = state.get("orders", {})
        batteries = state.get("batteries", {})
        launchpads = state.get("launchpads", {})
        eligible_order_ids, battery_ids, active_order_id = self._eligible_menu_inputs(
            state, launchpad, used_orders, used_batteries, current_time, planned_departures,
        )
        if not eligible_order_ids:
            return []
        horizon = max(1.0, float(self.config.get("simulation", {}).get("horizon_seconds", 1)))

        # ---- per-battery block (once per battery) ----
        b_ids: list[str] = []
        b_rows: list[list[float]] = []   # the 6 battery features (flat 10..15)
        b_charge: list[float] = []
        b_cap: list[float] = []
        b_reserve: list[float] = []
        for battery_id in battery_ids:
            battery = batteries[battery_id]
            bstate = self.battery_state_fn(battery, launchpad)
            if bstate is None:
                continue
            charge = float(bstate["charge_Wh"]); capacity = float(bstate["capacity_Wh"])
            completion = battery.get("charge_complete_at")
            b_ids.append(battery_id)
            b_rows.append([
                safe_ratio(charge, capacity),
                charge / 1500.0,
                capacity / 1500.0,
                float(battery.get("status") == BatteryStatus.CHARGING.value),
                0.0 if completion is None else max(0.0, float(completion) - current_time) / horizon,
            ])
            b_charge.append(charge); b_cap.append(capacity); b_reserve.append(float(bstate["reserve_Wh"]))
        n_b = len(b_ids)
        b_rows_arr = np.asarray(b_rows, dtype=np.float64).reshape(n_b, BATTERY_DIM)
        b_charge_arr = np.asarray(b_charge, dtype=np.float64)
        b_cap_arr = np.asarray(b_cap, dtype=np.float64)
        b_res_arr = np.asarray(b_reserve, dtype=np.float64)
        # env ratios use max(1, C); the energy model's reserve_after uses C if C > 0 else 1
        b_cap_env = np.maximum(1.0, b_cap_arr)
        b_cap_model = np.where(b_cap_arr > 0.0, b_cap_arr, 1.0)
        # Exact (o,l) pre-filter: a route is reachable by SOME battery at this pad
        # iff E_task(o,l) <= max_b (charge_b - reserve_b). Routes above that
        # bound would produce an all-False column of the grid, so they are
        # dropped before any per-(o,l) feature work (the fleet's best battery
        # bounds the whole pad).
        max_margin = float(np.max(b_charge_arr - b_res_arr)) if n_b > 0 else float("-inf")

        factored = self.layout == "olb"
        self.factor_menu = None
        if factored and n_b > 0:
            # The (o,l) block is shared across the pad's batteries, so the E_task
            # column in it has to hold one value for all of them. The row path
            # forms E/1500 as f32(E/C) * f32(C/1500) with *this* battery's C, and
            # the two C's cancel algebraically -- the split exists only so the
            # factored path reproduces the row path's float32 rounding bit for bit.
            #
            # With one capacity on the pad that reproduction is exact, so keep it.
            # With several, it cannot be: the row path itself rounds differently
            # per battery, so no single shared value is bit-identical to all of
            # them. Use the mathematically correct shared value, f32(E/1500) --
            # reached through the same expression with C := 1500 -- and accept
            # agreement to float32 round-off (<= 1 ulp) instead of bit-identity.
            if np.all(b_cap_arr == b_cap_arr[0]):
                cap_env_ref = float(b_cap_env[0])
                cap_over_1500_ref = float(b_cap_arr[0]) / 1500.0
            else:
                cap_env_ref, cap_over_1500_ref = 1500.0, 1.0
        else:
            cap_env_ref, cap_over_1500_ref = 1.0, 0.0
        # factor accumulators (olb only)
        ol_raw_rows: list[np.ndarray] = []      # dispatch (o,l) rows, then defer rows, then the zero stop row
        flat_kind: list[list[float]] = []
        flat_ol: list[int] = []
        flat_b: list[int] = []
        n_ol_real = 0
        zero_b_idx = n_b  # extra all-zero battery row for defer/stop entries

        candidates: list[_JointPlanCandidate] = []
        n_orders = len(eligible_order_ids)
        for rank, order_id in enumerate(eligible_order_ids):
            order = orders[order_id]
            routes_info = self.route_energy_fn(order) if n_b > 0 else None
            order7 = np.asarray(self._order_features(order, launchpad, current_time, rank, n_orders, horizon), dtype=np.float64)
            dest_xy = np.asarray(frame_xy(order.get("destination"), self.frame), dtype=np.float64)
            if routes_info is not None:
                # ---- per-(o,l) block (once per return pad) ----
                routes = routes_info["routes"]
                pruned = self.prune_return_plans(order, launchpad, launchpads, routes)
                l_ids: list[str] = []
                l_rows: list[list[float]] = []   # target features (flat 21..32)
                l_e_task: list[float] = []; l_e_ret: list[float] = []
                l_dl: list[float] = []; l_dr: list[float] = []; l_tr: list[float] = []
                l_xy: list[tuple[float, float]] = []
                for return_launchpad_id, route in sorted(pruned.items()):
                    target = launchpads.get(str(return_launchpad_id))
                    if target is None or not route["route_ok"]:
                        continue
                    if route["E_task_Wh"] > max_margin:
                        continue  # no battery on this pad can fly (o, l)
                    if not self.target_has_projected_capacity(
                        launchpad_id, target, planned_targets, planned_departures,
                    ):
                        continue
                    l_ids.append(str(return_launchpad_id))
                    l_rows.append(self.target_features(order, launchpad, target, planned_targets, planned_departures))
                    l_e_task.append(route["E_task_Wh"]); l_e_ret.append(route["E_return_Wh"])
                    l_dl.append(route["distance_loaded_m"]); l_dr.append(route["distance_return_m"]); l_tr.append(route["time_return_s"])
                    l_xy.append(frame_xy(target.get("current_location"), self.frame))
                n_l = len(l_ids)
                if self.collapse_return and n_l > 1:
                    # keep the cheapest return only: the dispatch agent is not the
                    # one choosing l, so the menu must not branch on it. Cheapest
                    # is also the loosest feasibility test, i.e. "some return works".
                    keep = int(np.argmin(np.asarray(l_e_task)))
                    l_ids, l_rows = [l_ids[keep]], [l_rows[keep]]
                    l_e_task, l_e_ret = [l_e_task[keep]], [l_e_ret[keep]]
                    l_dl, l_dr, l_tr = [l_dl[keep]], [l_dr[keep]], [l_tr[keep]]
                    l_xy = [l_xy[keep]]
                    n_l = 1
                if n_l > 0:
                    e_task = np.asarray(l_e_task, dtype=np.float64)     # [L]
                    e_ret = np.asarray(l_e_ret, dtype=np.float64)
                    # feasibility grid [B, L]: charge_b - E_task(o,l) >= reserve_b
                    feasible = (b_charge_arr[:, None] - e_task[None, :]) >= b_res_arr[:, None]
                    if feasible.any():
                        target12 = np.asarray(l_rows, dtype=np.float64)   # [L, 12]
                        tgt_xy = np.asarray(l_xy, dtype=np.float64)       # [L, 2]
                        d_ret = np.asarray(l_dr, dtype=np.float64) / 5000.0
                        t_ret = np.asarray(l_tr, dtype=np.float64) / horizon
                        b_idx, l_idx = np.nonzero(feasible)               # (b, l) order == triplet loop order
                        if factored:
                            # ol raw rows for this order's (o,l) pairs (only those some battery can fly)
                            keep_l = np.unique(l_idx)
                            base = len(ol_raw_rows)
                            l_to_row = {int(l): base + i for i, l in enumerate(keep_l)}
                            for l in keep_l:
                                row = np.concatenate([
                                    order7, dest_xy, target12[l, 1:12], tgt_xy[l],
                                    [e_task[l], e_ret[l], d_ret[l], t_ret[l]],
                                ])
                                if self.collapse_return and self.return_feature_mode != "full":
                                    # F0 chooses (o, b) only, so its row must not
                                    # describe a return pad it does not pick. On Hong
                                    # Kong the return agent lands on the collapsed pad
                                    # l* in 28% of flights and is a mean 1126 m away
                                    # from it otherwise, so l*'s queue state and route
                                    # numbers are wrong far more often than right --
                                    # not merely uninformative. Everything from the
                                    # target block onwards is therefore zeroed, leaving
                                    # the order block. Feasibility is untouched: the
                                    # grid below still uses E_task(o, l*), which is the
                                    # simulator's own test for an action naming no pad.
                                    end = (len(row) if self.return_feature_mode == "order_only"
                                           else _RAW_E_TASK)
                                    row[_RAW_TARGET.start:end] = 0.0
                                ol_raw_rows.append(row)
                            n_ol_real += len(keep_l)
                            for k in range(len(b_idx)):
                                flat_kind.append([1.0, 0.0, 0.0]); flat_ol.append(l_to_row[int(l_idx[k])]); flat_b.append(int(b_idx[k]))
                                candidates.append(_JointPlanCandidate("dispatch", order_id, b_ids[b_idx[k]], l_ids[l_idx[k]], None))
                        else:
                            route3 = np.empty((n_b, n_l, 3), dtype=np.float64)
                            route3[..., 0] = e_task[None, :] / b_cap_env[:, None]
                            route3[..., 1] = ((b_charge_arr[:, None] - e_task[None, :]) - b_res_arr[:, None]) / b_cap_model[:, None]
                            route3[..., 2] = d_ret[None, :]
                            target_kept = target12[:, [0] + [1 + k for k in TARGET_KEEP]]   # [L, 1 + 9]
                            rows_flat = np.concatenate([
                                np.broadcast_to(np.asarray([1.0, 0.0, 0.0]), (n_b, n_l, 3)),
                                np.broadcast_to(order7[ORDER_KEEP], (n_b, n_l, ORDER_DIM)),
                                np.broadcast_to(b_rows_arr[:, None, :], (n_b, n_l, BATTERY_DIM)),
                                route3,
                                np.broadcast_to(target_kept[None, :, :], (n_b, n_l, 1 + TARGET_DIM)),
                            ], axis=-1)
                            rows = rows_flat[feasible].astype(np.float32)     # [K, FLAT_DIM]
                            for k in range(len(rows)):
                                candidates.append(_JointPlanCandidate(
                                    "dispatch", order_id, b_ids[b_idx[k]], l_ids[l_idx[k]], rows[k],
                                ))
            if factored:
                # defer entry: an (o,l) row with the order block only
                ol_raw_rows.append(np.concatenate([order7, dest_xy, np.zeros(OL_RAW_DIM - 9)]))
                flat_kind.append([0.0, 1.0, 0.0]); flat_ol.append(len(ol_raw_rows) - 1); flat_b.append(zero_b_idx)
                candidates.append(_JointPlanCandidate("defer", order_id, None, None, None))
            else:
                candidates.append(_JointPlanCandidate(
                    "defer", order_id, None, None,
                    self._joint_features("defer", order, None, launchpad, None, current_time,
                                         rank, n_orders, {}, {}, planned_targets, planned_departures),
                ))
        if active_order_id is None:
            if factored:
                ol_raw_rows.append(np.zeros(OL_RAW_DIM))
                flat_kind.append([0.0, 0.0, 1.0]); flat_ol.append(len(ol_raw_rows) - 1); flat_b.append(zero_b_idx)
                candidates.append(_JointPlanCandidate("stop", None, None, None, None))
            else:
                candidates.append(_JointPlanCandidate(
                    "stop", None, None, None,
                    self._joint_features("stop", None, None, launchpad, None, current_time,
                                         0, n_orders, {}, {}, planned_targets, planned_departures),
                ))
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"launchpad {launchpad_id} has {len(candidates)} complete PPO plans, "
                f"above max_candidates_per_launchpad={self.max_candidates}"
            )
        if factored and candidates:
            ol_raw = np.stack(ol_raw_rows) if ol_raw_rows else np.zeros((0, OL_RAW_DIM))
            b_raw = np.zeros((n_b + 1, B_RAW_DIM), dtype=np.float64)
            b_raw[:n_b, 0:BATTERY_DIM] = b_rows_arr
            # offsets are relative to BATTERY_DIM, so changing the battery block's
            # width cannot leave the raw tail pointing at the wrong columns
            _c, _ce, _cm, _r = (BATTERY_DIM + k for k in range(4))
            b_raw[:n_b, _c] = b_charge_arr; b_raw[:n_b, _ce] = b_cap_env
            b_raw[:n_b, _cm] = b_cap_model; b_raw[:n_b, _r] = b_res_arr
            b_raw[n_b, _ce] = 1.0; b_raw[n_b, _cm] = 1.0   # zero battery row: safe denominators
            self.factor_menu = FactorMenu(
                ol_raw=ol_raw,
                b_raw=b_raw,
                ol=self.feat_mask.ol(ol_from_raw(ol_raw, cap_env_ref, cap_over_1500_ref)),
                b=self.feat_mask.b(b_raw[:, 0:BATTERY_DIM].astype(np.float32)),
                kind=np.asarray(flat_kind, dtype=np.float32),
                ol_idx=np.asarray(flat_ol, dtype=np.int64),
                b_idx=np.asarray(flat_b, dtype=np.int64),
                n_ol=n_ol_real,
                n_b=n_b,
            )
        return candidates

    def _candidates_triplet(
        self,
        state: StateDict,
        launchpad: dict,
        used_orders: set[str],
        used_batteries: set[str],
        current_time: int,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> list[_JointPlanCandidate]:
        launchpad_id = str(launchpad.get("launchpad_id"))
        orders = state.get("orders", {})
        batteries = state.get("batteries", {})
        launchpads = state.get("launchpads", {})
        active_order_id = self.active_order_by_launchpad.get(launchpad_id)
        eligible_order_ids = [
            str(order_id)
            for order_id in launchpad.get("order_queue", [])
            if str(order_id) not in used_orders
            and str(order_id) in orders
            and orders[str(order_id)].get("status") == OrderStatus.PENDING.value
            and (active_order_id is None or str(order_id) == active_order_id)
        ]
        eligible_order_ids = self.prune_order_ids(
            state,
            launchpad,
            eligible_order_ids,
            current_time,
            planned_departures,
        )
        if not eligible_order_ids:
            return []
        battery_ids = [
            str(battery_id)
            for battery_id in (
                list(launchpad.get("charged_battery_ids", []))
                + list(launchpad.get("charging_battery_ids", []))
            )
            if str(battery_id) not in used_batteries
            and str(battery_id) in batteries
        ]
        candidates: list[_JointPlanCandidate] = []
        for rank, order_id in enumerate(eligible_order_ids):
            order = orders[order_id]
            for battery_id in battery_ids:
                battery = batteries[battery_id]
                if self.candidate_details_fn is None:
                    if not self.candidate_feasible(order, battery, launchpad_id):
                        continue
                    details: dict = {}
                else:
                    details = self.candidate_details_fn(order, battery)
                    if not bool(details.get("feasible", False)):
                        continue
                return_plans = self.prune_return_plans(
                    order,
                    launchpad,
                    launchpads,
                    details.get("return_plans", {}),
                )
                for return_launchpad_id, route_details in sorted(return_plans.items()):
                    target = launchpads.get(str(return_launchpad_id))
                    if target is None or not bool(route_details.get("feasible", False)):
                        continue
                    if not self.target_has_projected_capacity(
                        launchpad_id,
                        target,
                        planned_targets,
                        planned_departures,
                    ):
                        continue
                    candidates.append(_JointPlanCandidate(
                        "dispatch",
                        order_id,
                        battery_id,
                        str(return_launchpad_id),
                        self._joint_features(
                            "dispatch",
                            order,
                            battery,
                            launchpad,
                            target,
                            current_time,
                            rank,
                            len(eligible_order_ids),
                            details,
                            route_details,
                            planned_targets,
                            planned_departures,
                        ),
                    ))
            candidates.append(_JointPlanCandidate(
                "defer",
                order_id,
                None,
                None,
                self._joint_features(
                    "defer",
                    order,
                    None,
                    launchpad,
                    None,
                    current_time,
                    rank,
                    len(eligible_order_ids),
                    {},
                    {},
                    planned_targets,
                    planned_departures,
                ),
            ))
        if active_order_id is None:
            candidates.append(_JointPlanCandidate(
                "stop",
                None,
                None,
                None,
                self._joint_features(
                    "stop",
                    None,
                    None,
                    launchpad,
                    None,
                    current_time,
                    0,
                    len(eligible_order_ids),
                    {},
                    {},
                    planned_targets,
                    planned_departures,
                ),
            ))
        if len(candidates) > self.max_candidates:
            raise ValueError(
                f"launchpad {launchpad_id} has {len(candidates)} complete PPO plans, "
                f"above max_candidates_per_launchpad={self.max_candidates}"
            )
        return candidates

    def filter_for_drone(
        self, state: StateDict, drone_id: str, candidates: list[_JointPlanCandidate]
    ) -> list[_JointPlanCandidate]:
        """Drop dispatch entries this drone cannot carry; keeps the FactorMenu aligned."""
        keep = [
            i for i, candidate in enumerate(candidates)
            if candidate.kind != "dispatch"
            or (
                candidate.order_id is not None
                and self.drone_can_carry(state, drone_id, state["orders"][candidate.order_id])
            )
        ]
        if len(keep) == len(candidates):
            return candidates
        if self.factor_menu is not None and len(self.factor_menu) == len(candidates):
            self.factor_menu = self.factor_menu.subset(keep)
        return [candidates[i] for i in keep]

    def _joint_features(
        self,
        kind: str,
        order: dict | None,
        battery: dict | None,
        departure: dict,
        target: dict | None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        f33 = self._triplet_features(kind, order, battery, departure, target, *args, **kwargs)
        if self.layout == "flat":
            return f33
        dest_xy = frame_xy(order.get("destination"), self.frame) if order is not None else (0.0, 0.0)
        target_xy = frame_xy(target.get("current_location"), self.frame) if target is not None else (0.0, 0.0)
        return reform_flat_to_olb(f33, dest_xy, target_xy)

    @staticmethod
    def _order_features(
        order: dict, departure: dict, current_time: int, queue_rank: int, queue_size: int, horizon: float
    ) -> list[float]:
        """The 7 RAW order features [value, urgency, slack, age, weight, delivery distance,
        queue rank]; the F2 family consumes ORDER_KEEP of them, L2/L3 tokens use all 7."""
        return [
            float(order.get("value", 0.0) or 0.0) / 50.0,
            urgency(order, current_time),
            (float(order.get("deadline", current_time)) - current_time) / horizon,
            max(0.0, current_time - float(order.get("created_at", current_time)))
            / horizon,
            float(order.get("weight", 0.0) or 0.0) / 5.0,
            distance_m(
                departure.get("current_location"),
                order.get("destination"),
            )
            / 5000.0,
            float(queue_rank) / max(1.0, queue_size - 1.0),
        ]

    def _triplet_features(
        self,
        kind: str,
        order: dict | None,
        battery: dict | None,
        departure: dict,
        target: dict | None,
        current_time: int,
        queue_rank: int,
        queue_size: int,
        details: dict,
        route_details: dict,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> np.ndarray:
        """The expanded triplet vector (layout 'flat'): kind | order(5) | battery(6) | route(3) | target(1+9)."""
        horizon = max(
            1.0,
            float(self.config.get("simulation", {}).get("horizon_seconds", 1)),
        )
        kind_features = [
            float(kind == "dispatch"),
            float(kind == "defer"),
            float(kind == "stop"),
        ]
        if order is None:
            order_features = [0.0] * ORDER_DIM
        else:
            raw_order = self._order_features(order, departure, current_time, queue_rank, queue_size, horizon)
            order_features = [raw_order[k] for k in ORDER_KEEP]
        if battery is None:
            battery_features = [0.0] * BATTERY_DIM
        else:
            capacity = float(battery.get("capacity_Wh", 0.0) or 0.0)
            charge = float(
                details.get(
                    "available_charge_Wh",
                    battery.get("current_charge_Wh", 0.0),
                )
                or 0.0
            )
            completion = battery.get("charge_complete_at")
            battery_features = [
                safe_ratio(charge, capacity),
                charge / 1500.0,
                capacity / 1500.0,
                float(battery.get("status") == BatteryStatus.CHARGING.value),
                0.0
                if completion is None
                else max(0.0, float(completion) - current_time) / horizon,
            ]
        route_features = (
            [
                float(route_details.get("task_energy_ratio", 0.0) or 0.0),
                float(route_details.get("reserve_after_ratio", 0.0) or 0.0),
                float(route_details.get("distance_return_m", 0.0) or 0.0) / 5000.0,
            ]
            if target is not None
            else [0.0] * 3
        )
        raw_target = self.target_features(
            order,
            departure,
            target,
            planned_targets,
            planned_departures,
        )
        target_features = [raw_target[0]] + [raw_target[1 + k] for k in TARGET_KEEP]
        features = np.asarray(
            kind_features
            + order_features
            + battery_features
            + route_features
            + target_features,
            dtype=np.float32,
        )
        if features.shape != (FLAT_DIM,):
            raise RuntimeError(f"triplet candidate must have shape ({FLAT_DIM},), got {features.shape}")
        return features

    def target_features(
        self,
        order: dict | None,
        departure: dict,
        target: dict | None,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> list[float]:
        if order is None or target is None:
            return [0.0] * 12
        departure_id = str(departure.get("launchpad_id"))
        target_id = str(target.get("launchpad_id"))
        planned_incoming = int(planned_targets.get(target_id, 0))
        planned_outgoing = int(planned_departures.get(target_id, 0))
        projected = self.projected_target_occupancy(
            departure_id,
            target,
            planned_targets,
            planned_departures,
        )
        backlog = float(len(target.get("order_queue", [])))
        capacity = target.get("drone_capacity")
        return [
            distance_m(order.get("destination"), target.get("current_location")) / 5000.0,
            backlog / 20.0,
            len(target.get("drone_ids", [])) / 20.0,
            float(target.get("reserved_return_slots", 0) or 0) / 20.0,
            planned_incoming / 20.0,
            planned_outgoing / 20.0,
            projected / 20.0,
            (backlog - projected) / 20.0,
            len(target.get("charged_battery_ids", [])) / 20.0,
            len(target.get("charging_battery_ids", [])) / 20.0,
            float(target_id == departure_id),
            projected / max(1.0, float(capacity)) if capacity is not None else 0.0,
        ]

    def context(
        self,
        state: StateDict,
        launchpad: dict,
        used_orders: set[str],
        used_batteries: set[str],
        remaining_drones: int,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
        agent_index: int | None = None,
        agent_count: int | None = None,
    ) -> np.ndarray:
        base = self._base_context(
            state,
            launchpad,
            used_orders,
            used_batteries,
            remaining_drones,
        )
        launchpads = state.get("launchpads", {})
        fleet_size = max(1.0, float(len(state.get("drones", {}))))
        finite = [
            target
            for target in launchpads.values()
            if target.get("drone_capacity") is not None
        ]
        projected_ratios = [
            (
                len(target.get("drone_ids", []))
                + int(target.get("reserved_return_slots", 0) or 0)
                - int(planned_departures.get(str(target.get("launchpad_id")), 0))
                + int(planned_targets.get(str(target.get("launchpad_id")), 0))
            )
            / max(1.0, float(target["drone_capacity"]))
            for target in finite
        ]
        joint = np.asarray([
            sum(planned_targets.values()) / fleet_size,
            sum(count > 0 for count in planned_targets.values())
            / max(1.0, float(len(launchpads))),
            max(planned_targets.values(), default=0) / fleet_size,
            max(projected_ratios, default=0.0),
            float(bool(finite)),
        ], dtype=np.float32)
        fleet_size = max(1, len(state.get("drones", {})))
        count = max(1, int(agent_count if agent_count is not None else fleet_size))
        position = max(0, min(int(agent_index or 0), count - 1))
        sequence_features = np.asarray(
            [position / max(1.0, float(count - 1)), count / float(fleet_size)],
            dtype=np.float32,
        )
        context = np.concatenate([base, joint, sequence_features]).astype(
            np.float32, copy=False
        )
        # MAT variants append previous-action context after this base context.
        if context.shape != (self.context_dim,):
            raise RuntimeError(
                f"joint-plan base context must have shape ({self.context_dim},), "
                f"got {context.shape}"
            )
        return self.feat_mask.context(context)

    def _base_context(
        self,
        state: StateDict,
        launchpad: dict,
        used_orders: set[str],
        used_batteries: set[str],
        remaining_drones: int,
    ) -> np.ndarray:
        horizon = max(1.0, float(self.config.get("simulation", {}).get("horizon_seconds", 1)))
        drones = state.get("drones", {})
        batteries = state.get("batteries", {})
        # Fleet-wide totals are fixed within one step (the state dict is rebuilt
        # every step); count them once and subtract this step's commitments.
        totals = self._step_totals(state)
        pending = totals["pending"] - sum(
            1 for order_id in used_orders
            if (order := state.get("orders", {}).get(order_id)) is not None
            and order.get("status") == OrderStatus.PENDING.value
        )
        idle = totals["idle"]
        moving = totals["moving"]
        docked = totals["docked"]
        planned_dispatches = len(used_orders)
        idle = max(0, idle - planned_dispatches)
        moving += planned_dispatches
        docked = max(0, docked - planned_dispatches)
        effective_queue = sum(
            str(order_id) not in used_orders
            and str(order_id) in state.get("orders", {})
            and state["orders"][str(order_id)].get("status") == OrderStatus.PENDING.value
            for order_id in launchpad.get("order_queue", [])
        )
        charged_batteries = sum(
            str(battery_id) not in used_batteries
            for battery_id in launchpad.get("charged_battery_ids", [])
        )
        charging_batteries = sum(
            str(battery_id) not in used_batteries
            for battery_id in launchpad.get("charging_battery_ids", [])
        )
        fleet_size = max(1.0, float(len(drones)))
        initially_available = self.available_drone_count(state, launchpad)
        effective_docked = max(
            0,
            len(launchpad.get("drone_ids", []))
            - max(0, initially_available - remaining_drones),
        )
        return np.asarray([
            float(state.get("time", 0)) / horizon,
            pending / 50.0,
            idle / max(1.0, len(drones)),
            moving / max(1.0, len(drones)),
            docked / max(1.0, len(drones)),
            (len(batteries) - len(used_batteries)) / 50.0,
            (totals["charged"] - sum(
                1 for battery_id in used_batteries
                if (battery := batteries.get(battery_id)) is not None
                and battery.get("status") == BatteryStatus.CHARGED.value
            )) / max(1.0, len(batteries)),
            (totals["charging"] - sum(
                1 for battery_id in used_batteries
                if (battery := batteries.get(battery_id)) is not None
                and battery.get("status") == BatteryStatus.CHARGING.value
            )) / max(1.0, len(batteries)),
            effective_queue / 20.0,
            remaining_drones / fleet_size,
            effective_docked / fleet_size,
            (effective_queue - remaining_drones) / 20.0,
            charged_batteries / 20.0,
            charging_batteries / 20.0,
            float(str(launchpad.get("launchpad_id")) in self.active_order_by_launchpad),
            1.0,
        ], dtype=np.float32)

    def global_state(self, state: StateDict) -> np.ndarray:
        """Pre-commitment global state of this step for a centralised critic:
        one 23-dim context row per launchpad that has an available drone
        (``[P, C]``). Drones on the same pad share the pad's situation, so the
        state is per pad, not per drone; the row already carries the pad's
        available-drone count. No candidate-level information."""
        pads = [
            (pad_id, pad) for pad_id, pad in state.get("launchpads", {}).items()
            if self.available_drone_count(state, pad) > 0
        ]
        rows = [
            self.context(
                state, launchpad, set(), set(), self.available_drone_count(state, launchpad), {}, {},
                agent_index=pad_index, agent_count=len(pads),
            )
            for pad_index, (_pad_id, launchpad) in enumerate(pads)
        ]
        return (np.stack(rows).astype(np.float32) if rows
                else np.zeros((0, self.feat_mask.context_dim), dtype=np.float32))

    def prune_order_ids(
        self,
        state: StateDict,
        launchpad: dict,
        order_ids: list[str],
        current_time: int,
        planned_departures: dict[str, int] | None = None,
    ) -> list[str]:
        """The only thing pruning does: keep the earliest-deadline
        ``urgent_order_multiplier x (drones this pad can dispatch right now)``
        orders. "Can dispatch right now" is ``available_drone_count`` -- idle, no
        route, no onboard order, no pack mounted -- minus the departures already
        planned this step, floored at 1 so a pad with a full queue and no free
        drone still offers one order to defer on."""
        if not self.pruning or not order_ids:
            return order_ids
        launchpad_id = str(launchpad.get("launchpad_id"))
        planned = int((planned_departures or {}).get(launchpad_id, 0))
        available = max(1, self.available_drone_count(state, launchpad) - planned)
        limit = max(1, math.ceil(self.urgent_order_multiplier * available))
        orders = state.get("orders", {})
        # earliest deadline first; sorted() is stable, so equal-deadline orders
        # retain their queue order
        return sorted(
            order_ids,
            key=lambda order_id: float(orders[order_id].get("deadline", current_time)),
        )[:limit]

    def prune_return_plans(
        self,
        order: dict,
        departure: dict,
        launchpads: dict,
        return_plans: dict,
    ) -> dict:
        """Off by default: ``return_corridor_km <= 0`` keeps every return pad, so
        ``candidate_pruning`` restricts the order dimension alone.

        The corridor was a geometric filter -- pads within this distance of the
        order's own flight segment -- applied *before* the energy tests. It was
        measured to drop a mean 7.7 (meituanC) / 9.2 (shenzhen) energy-feasible
        pads per (order, pad) pair while never removing the cheapest one, so it
        cut real options for a speedup that came almost entirely from the order
        dimension instead. Set a positive value to put it back."""
        if not self.pruning or self.return_corridor_km <= 0.0:
            return return_plans
        destination = order.get("destination")
        origin = departure.get("current_location")
        if not origin or not destination:
            return return_plans
        radius_m = self.return_corridor_km * 1000.0
        return {
            str(target_id): route
            for target_id, route in return_plans.items()
            if (target := launchpads.get(str(target_id))) is not None
            and self._point_to_segment_distance_m(
                target.get("current_location"), origin, destination
            ) <= radius_m
        }

    @staticmethod
    def _point_to_segment_distance_m(
        point: dict | None,
        start: dict | None,
        end: dict | None,
    ) -> float:
        """Approximate point-to-order-route distance in metres."""
        if not point or not start or not end:
            return float("inf")
        lat0 = np.deg2rad(float(start.get("lat", 0.0)))
        scale_x = 111_320.0 * float(np.cos(lat0))
        scale_y = 110_540.0
        x0, y0 = float(start.get("lon", 0.0)) * scale_x, float(start.get("lat", 0.0)) * scale_y
        x1, y1 = float(end.get("lon", 0.0)) * scale_x, float(end.get("lat", 0.0)) * scale_y
        xp, yp = float(point.get("lon", 0.0)) * scale_x, float(point.get("lat", 0.0)) * scale_y
        dx, dy = x1 - x0, y1 - y0
        if dx == 0.0 and dy == 0.0:
            return float(np.hypot(xp - x0, yp - y0))
        t = max(0.0, min(1.0, ((xp - x0) * dx + (yp - y0) * dy) / (dx * dx + dy * dy)))
        return float(np.hypot(xp - (x0 + t * dx), yp - (y0 + t * dy)))

    def candidate_feasible(self, order: dict, battery: dict, launchpad_id: str) -> bool:
        if battery.get("current_vehicle_id") is not None:
            return False
        if str(battery.get("current_launchpad_id")) != launchpad_id:
            return False
        if battery.get("status") not in {BatteryStatus.CHARGED.value, BatteryStatus.CHARGING.value}:
            return False
        return self.candidate_feasible_fn is None or self.candidate_feasible_fn(order, battery)

    def target_has_projected_capacity(
        self,
        departure_id: str,
        target: dict,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> bool:
        capacity = target.get("drone_capacity")
        if capacity is None:
            return True
        return self.projected_target_occupancy(
            departure_id,
            target,
            planned_targets,
            planned_departures,
        ) <= int(capacity)

    @staticmethod
    def projected_target_occupancy(
        departure_id: str,
        target: dict,
        planned_targets: dict[str, int],
        planned_departures: dict[str, int],
    ) -> int:
        target_id = str(target.get("launchpad_id"))
        return (
            len(target.get("drone_ids", []))
            + int(target.get("reserved_return_slots", 0) or 0)
            - int(planned_departures.get(target_id, 0))
            + int(planned_targets.get(target_id, 0))
            - int(target_id == departure_id)
            + 1
        )

    def _step_totals(self, state: StateDict) -> dict[str, int]:
        """Fleet-wide counts of the current step, memoised on the state object."""
        key = (id(state), int(state.get("time", 0)))
        cached = getattr(self, "_step_totals_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        drones = state.get("drones", {})
        batteries = state.get("batteries", {})
        totals = {
            "pending": sum(order.get("status") == OrderStatus.PENDING.value for order in state.get("orders", {}).values()),
            "idle": sum(drone.get("status") == VehicleStatus.IDLE.value for drone in drones.values()),
            "moving": sum(drone.get("status") == VehicleStatus.MOVING.value for drone in drones.values()),
            "docked": sum(drone.get("current_launchpad_id") is not None for drone in drones.values()),
            "charged": sum(battery.get("status") == BatteryStatus.CHARGED.value for battery in batteries.values()),
            "charging": sum(battery.get("status") == BatteryStatus.CHARGING.value for battery in batteries.values()),
        }
        self._step_totals_cache = (key, totals)
        return totals

    def available_drone_count(self, state: StateDict, launchpad: dict) -> int:
        drones = state.get("drones", {})
        return sum(
            1
            for drone_id in launchpad.get("drone_ids", [])
            if drone_id in drones
            and drones[drone_id].get("status") == VehicleStatus.IDLE.value
            and not drones[drone_id].get("route_plan")
            and not drones[drone_id].get("onboard_order_ids")
            and drones[drone_id].get("current_battery_id") is None
        )

    @staticmethod
    def drone_is_available(
        state: StateDict,
        launchpad: dict,
        drone_id: str,
    ) -> bool:
        if drone_id not in {str(item) for item in launchpad.get("drone_ids", [])}:
            return False
        drone = state.get("drones", {}).get(drone_id)
        return bool(
            drone is not None
            and drone.get("status") == VehicleStatus.IDLE.value
            and not drone.get("route_plan")
            and not drone.get("onboard_order_ids")
            and drone.get("current_battery_id") is None
        )

    @staticmethod
    def drone_can_carry(state, drone_id: str, order: dict) -> bool:
        """Can this drone physically take the order: enough free capacity for its size and
        enough payload for its weight. Nothing about energy or routes is checked here.

        Both tests hold trivially in the current scenario -- every drone has capacity 1 and
        is only offered a menu when it is idle and empty, every order has size 1, and the
        payload limit (10 kg) equals the largest weight the generator draws -- so this
        is a guard, not a filter that ever fires. It duplicates the simulator's own test in
        ``SimulatorEnv._available_drone_at_launchpad``, which is the one that actually
        rejects an action; keeping it here only stops the menu from offering something the
        simulator would refuse. Both must change together if a drone ever carries more than
        one order.
        """
        drone = state["drones"][drone_id]
        if float(drone.get("remaining_capacity", 0.0)) < float(order.get("size", 0.0)):
            return False
        max_payload = drone.get("max_payload_kg")
        return max_payload is None or float(max_payload) >= float(order.get("weight", 0.0))

    def clear_terminal_lock(self, state: StateDict, launchpad_id: str, launchpad: dict) -> None:
        active_order_id = self.active_order_by_launchpad.get(launchpad_id)
        if active_order_id is None:
            return
        order = state.get("orders", {}).get(active_order_id)
        if (
            order is None
            or active_order_id not in launchpad.get("order_queue", [])
            or order.get("status") != OrderStatus.PENDING.value
        ):
            self.active_order_by_launchpad.pop(launchpad_id, None)

