"""L1: RBG -- "Ready, Bid, Go!" (Talamali et al., AAMAS 2025).

Planned reproduction. Mechanism (paper): every idle battery/UAV owns an online
linear SVM over (delivery distance, parcel weight, state of charge); f(x) >= 0
means "bid", bid value = f(x)/||w||; the LEAST-confident positive bidder wins
the order; the winner's outcome (success / abort) is fed back as a one-sample
SGD update. Orders are advertised FIFO.

Repo mapping used here:
  * bidders = detachable batteries at the launchpad (one SVM per battery id);
  * the simulator forbids in-flight aborts, so the exact energy-feasibility
    check plays the abort: the winner is vetoed, gets the negative label at
    once and the order is re-auctioned among the remaining bidders (a vetoed
    bid must NOT inherit the order's later outcome under another battery);
  * batteries are homogeneous (no hidden SoH), so the paper's main source of
    bidder heterogeneity is absent -- report, do not over-claim;
  * return stays the backlog heuristic (two-stage, like R1-R5).
This is intentionally a small, dependency-free adaptation rather than a claim
that the repository reproduces the paper's heterogeneous-battery simulator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from algorithm.common.decision_metrics import OrderBatteryDecisionMetrics
from algorithm.common.geometry_utils import distance_m
from simulator.model.enums import BatteryStatus, OrderStatus, VehicleStatus
from simulator.model.types import StateDict, TaskLevelActionDict


class _LinearBidder:
    """Small, dependency-free modified-Huber linear SVM."""

    MEAN = np.asarray((3500.0, 2.75, 50.0), dtype=float)
    STD = np.asarray((1443.38, 1.298, 28.87), dtype=float)

    def __init__(self, *, alpha: float = 0.01, learning_rate: float = 0.01):
        self.alpha = float(alpha)
        self.learning_rate = float(learning_rate)
        self.w = np.zeros(3, dtype=float)
        self.bias = 0.0
        self.steps = 0
        # The paper starts from two physically obvious labels.
        self._update_raw(np.asarray((1000.0, 0.5, 100.0)), 1, decay=False)
        self._update_raw(np.asarray((6000.0, 5.0, 0.0)), 0, decay=False)

    @classmethod
    def from_state(cls, state: dict) -> "_LinearBidder":
        bidder = cls()
        bidder.w = np.asarray(state.get("w", bidder.w), dtype=float)
        bidder.bias = float(state.get("bias", bidder.bias))
        bidder.steps = int(state.get("steps", bidder.steps))
        return bidder

    def state(self) -> dict:
        return {"w": self.w.tolist(), "bias": self.bias, "steps": self.steps}

    def _scaled(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=float) - self.MEAN) / self.STD

    def score(self, features: np.ndarray) -> float:
        return float(np.dot(self.w, self._scaled(features)) + self.bias)

    def bid_value(self, features: np.ndarray) -> float:
        return self.score(features) / max(float(np.linalg.norm(self.w)), 1e-12)

    def update(self, features: np.ndarray, label: int) -> None:
        self._update_raw(features, label, decay=True)

    def _update_raw(self, features: np.ndarray, label: int, *, decay: bool) -> None:
        x = self._scaled(features)
        signed_label = 1.0 if int(label) else -1.0
        margin = signed_label * (float(np.dot(self.w, x)) + self.bias)
        if margin >= -1.0:
            hinge = max(0.0, 1.0 - margin)
            d_loss_d_score = -2.0 * signed_label * hinge
        else:
            d_loss_d_score = -4.0 * signed_label
        eta = self.learning_rate / np.sqrt(1.0 + self.steps) if decay else self.learning_rate
        self.w -= eta * (d_loss_d_score * x + self.alpha * self.w)
        self.bias -= eta * d_loss_d_score
        if decay:
            self.steps += 1


class RBGSVMAuctionOrderBatteryPolicy:
    """RBG: FIFO order advertising with least-confident SVM bidding.

    The real paper learns one bidder per UAV. This simulator assigns a battery
    to an idle UAV only after the order-battery action, so the battery is the
    stable local bidder identity here. Exact simulator feasibility remains a
    safety veto; rejected actions become negative online examples.
    """

    def __init__(self, config: dict, *, seed: int = 0):
        self.config = config
        self.training = False
        self.candidate_feasible_fn = None
        self._bidders: dict[str, _LinearBidder] = {}
        self._attempts: list[tuple[str, str, np.ndarray]] = []
        self._vetoed: list[tuple[str, str, np.ndarray]] = []
        self._metrics = OrderBatteryDecisionMetrics()
        self._positive_bids = 0
        self._seed = int(seed)

    def _bidder(self, battery_id: str) -> _LinearBidder:
        return self._bidders.setdefault(battery_id, _LinearBidder())

    @staticmethod
    def _available_drones(launchpad: dict, drones: dict) -> int:
        return sum(
            1
            for drone_id in launchpad.get("drone_ids", [])
            if drone_id in drones
            and drones[drone_id].get("status") == VehicleStatus.IDLE.value
            and not drones[drone_id].get("route_plan")
            and drones[drone_id].get("current_battery_id") is None
        )

    @staticmethod
    def _available_batteries(launchpad: dict, batteries: dict, used: set[str]) -> list[str]:
        return [
            str(battery_id)
            for battery_id in (
                list(launchpad.get("charged_battery_ids", []))
                + list(launchpad.get("charging_battery_ids", []))
            )
            if str(battery_id) not in used
            and str(battery_id) in batteries
            and batteries[str(battery_id)].get("current_vehicle_id") is None
            and batteries[str(battery_id)].get("status")
            in {BatteryStatus.CHARGED.value, BatteryStatus.CHARGING.value}
        ]

    @staticmethod
    def _features(order: dict, battery: dict, launchpad: dict) -> np.ndarray:
        capacity = float(battery.get("capacity_Wh", 0.0) or 0.0)
        charge = float(battery.get("current_charge_Wh", 0.0) or 0.0)
        soc = 100.0 * charge / capacity if capacity > 0.0 else 0.0
        return np.asarray(
            (
                distance_m(launchpad.get("current_location"), order.get("destination")),
                float(order.get("weight", 0.0) or 0.0),
                soc,
            ),
            dtype=float,
        )

    @staticmethod
    def _id_tiebreaker(battery_id: str) -> int:
        _, _, suffix = str(battery_id).rpartition("_")
        try:
            return int(suffix)
        except ValueError:
            return 0

    def plan(
        self,
        state: StateDict,
        used_orders: set[str],
        used_batteries: set[str],
    ) -> list[TaskLevelActionDict]:
        actions: list[TaskLevelActionDict] = []
        current_time = int(state.get("time", 0))
        orders = state.get("orders", {})
        batteries = state.get("batteries", {})
        drones = state.get("drones", {})
        for launchpad in sorted(
            state.get("launchpads", {}).values(),
            key=lambda item: str(item.get("launchpad_id", "")),
        ):
            available_drones = self._available_drones(launchpad, drones)
            battery_ids = self._available_batteries(launchpad, batteries, used_batteries)
            pending_order_ids = [
                str(order_id)
                for order_id in launchpad.get("order_queue", [])
                if str(order_id) not in used_orders
                and str(order_id) in orders
                and orders[str(order_id)].get("status") == OrderStatus.PENDING.value
            ]
            selected = 0
            for order_id in pending_order_ids:
                if selected >= available_drones or not battery_ids:
                    break
                order = orders[order_id]
                bidders: list[tuple[float, str, np.ndarray]] = []
                for battery_id in battery_ids:
                    features = self._features(order, batteries[battery_id], launchpad)
                    bidder = self._bidder(battery_id)
                    if bidder.score(features) >= 0.0:
                        bidders.append((bidder.bid_value(features), battery_id, features))
                self._positive_bids += len(bidders)
                # Paper: the least-confident positive bidder wins and attempts the
                # mission; an abort is an immediate negative sample and the order is
                # re-advertised. The simulator forbids in-flight aborts, so the exact
                # feasibility check is the abort: veto -> negative label now, re-run
                # the auction among the remaining bidders for the same order.
                while bidders:
                    bid_value, battery_id, features = min(
                        bidders,
                        # Paper tie-break: highest UAV/battery id wins.
                        key=lambda item: (item[0], -self._id_tiebreaker(item[1]), item[1]),
                    )
                    del bid_value
                    if self.candidate_feasible_fn is not None and not self.candidate_feasible_fn(
                        order, batteries[battery_id]
                    ):
                        self._vetoed.append((order_id, battery_id, features))
                        bidders = [item for item in bidders if item[1] != battery_id]
                        continue
                    actions.append({
                        "action_id": f"rbg_{current_time}_{order_id}_{battery_id}",
                        "action_type": "order_battery_dispatch",
                        "order_id": order_id,
                        "battery_id": battery_id,
                        "feasibility": True,
                    })
                    self._attempts.append((order_id, battery_id, features))
                    used_orders.add(order_id)
                    used_batteries.add(battery_id)
                    battery_ids.remove(battery_id)
                    selected += 1
                    break
            self._metrics.record_group(
                order_ids=pending_order_ids,
                battery_ids=[
                    str(battery_id)
                    for battery_id in (
                        list(launchpad.get("charged_battery_ids", []))
                        + list(launchpad.get("charging_battery_ids", []))
                    )
                ],
                selected_count=selected,
                available_drones=available_drones,
            )
        return actions

    def finish_episode(
        self,
        on_time_gmv: float,
        *,
        reward_events: list[dict] | None = None,
        **kwargs,
    ) -> dict[str, float]:
        delivered = {str(event.get("order_id")) for event in (reward_events or [])}
        updates = 0
        if self.training:
            # accepted attempts: label = this dispatch was delivered (the order is
            # locked to this battery, so the order outcome is the attempt outcome)
            for order_id, battery_id, features in self._attempts:
                self._bidder(battery_id).update(features, int(order_id in delivered))
                updates += 1
            # vetoed attempts: the paper's aborted missions, always negative
            for _order_id, battery_id, features in self._vetoed:
                self._bidder(battery_id).update(features, 0)
                updates += 1
        result = {
            "rbg_svm_updates": float(updates),
            "rbg_svm_positive_bids": float(self._positive_bids),
            "rbg_svm_vetoed_bids": float(len(self._vetoed)),
        }
        result.update(self._metrics.summary())
        self.clear_episode()
        return result

    def clear_episode(self) -> None:
        self._attempts.clear()
        self._vetoed.clear()
        self._positive_bids = 0
        self._metrics.clear()

    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"bidders": {key: value.state() for key, value in self._bidders.items()}}),
            encoding="utf-8",
        )

    def load(self, path: str) -> None:
        input_path = Path(path)
        if not input_path.is_file():
            return
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        self._bidders = {
            str(key): _LinearBidder.from_state(value)
            for key, value in payload.get("bidders", {}).items()
        }
