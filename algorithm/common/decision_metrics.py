from __future__ import annotations

class OrderBatteryDecisionMetrics:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.groups = 0
        self.candidates = 0
        self.selected = 0
        self.unique_orders = 0
        self.unique_batteries = 0
        self.available_drones = 0
        self.groups_without_feasible_dispatch = 0
        self.defer_decisions = 0
        self.defer_with_feasible_battery = 0
        self.defer_without_feasible_battery = 0
        self.stop_decisions = 0

    def record_group(
        self,
        *,
        order_ids: list[str],
        battery_ids: list[str],
        selected_count: int,
        available_drones: int,
        selected_kind: str | None = None,
        selected_order_has_feasible_battery: bool | None = None,
    ) -> None:
        self.groups += 1
        self.candidates += len(order_ids)
        self.selected += selected_count
        self.unique_orders += len(set(order_ids))
        self.unique_batteries += len(set(battery_ids))
        self.available_drones += available_drones
        if not order_ids:
            self.groups_without_feasible_dispatch += 1
        if selected_kind == "defer":
            self.defer_decisions += 1
            if selected_order_has_feasible_battery is True:
                self.defer_with_feasible_battery += 1
            elif selected_order_has_feasible_battery is False:
                self.defer_without_feasible_battery += 1
        elif selected_kind == "stop":
            self.stop_decisions += 1

    def summary(self) -> dict[str, float]:
        denominator = max(1, self.groups)
        return {
            "order_battery_decision_groups": float(self.groups),
            "order_battery_candidates": float(self.candidates),
            "order_battery_selected": float(self.selected),
            "order_battery_candidates_per_group": float(self.candidates / denominator),
            "order_battery_selected_per_group": float(self.selected / denominator),
            "order_battery_unique_orders_per_group": float(self.unique_orders / denominator),
            "order_battery_unique_batteries_per_group": float(self.unique_batteries / denominator),
            "order_battery_available_drones_per_group": float(self.available_drones / denominator),
            "order_battery_selection_rate": float(self.selected / max(1, self.unique_orders)),
            "order_battery_groups_without_feasible_dispatch": float(
                self.groups_without_feasible_dispatch
            ),
            "order_battery_defer_decisions": float(self.defer_decisions),
            "order_battery_defer_with_feasible_battery": float(
                self.defer_with_feasible_battery
            ),
            "order_battery_defer_without_feasible_battery": float(
                self.defer_without_feasible_battery
            ),
            "order_battery_defer_rate": float(self.defer_decisions / denominator),
            "order_battery_stop_decisions": float(self.stop_decisions),
            "order_battery_stop_rate": float(self.stop_decisions / denominator),
        }
