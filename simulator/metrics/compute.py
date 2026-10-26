from __future__ import annotations

from ..model.entities import Battery, Drone, Launchpad, Order
from ..model.enums import BatteryStatus, OrderStatus
from ..utils.math_utils import safe_div


def compute_episode_metrics(
    orders: dict[str, Order],
    drones: dict[str, Drone],
    horizon_seconds: int,
    config: dict,
    launchpads: dict[str, Launchpad] | list[Launchpad] | None = None,
    batteries: dict[str, Battery] | list[Battery] | None = None,
) -> dict:
    all_orders = list(orders.values())
    delivered = [order for order in all_orders if order.status == OrderStatus.DELIVERED]
    expired = [order for order in all_orders if order.status == OrderStatus.EXPIRED]
    unfinished = [order for order in all_orders if order.status == OrderStatus.IN_TRANSIT]
    pending = [order for order in all_orders if order.status == OrderStatus.PENDING]
    on_time = [
        order for order in delivered
        if order.delivery_time is not None and order.delivery_time <= order.deadline
    ]
    late = [order for order in delivered if order not in on_time]
    generated = len(all_orders)
    assert len(delivered) + len(expired) + len(unfinished) + len(pending) == generated
    completion_rate = safe_div(len(delivered), generated)
    expired_rate = safe_div(len(expired), generated)
    if generated and not unfinished and not pending:
        # closed episode: completion + expired must account for every order
        assert abs(completion_rate + expired_rate - 1.0) < 1e-9, (completion_rate, expired_rate)

    # Average lateness over ALL delivered orders (on-time deliveries count as 0),
    # so it is not conditional on "given late". Identity: avg_lateness * delivered
    # * lateness_penalty_per_min == delay_penalty_cost.
    avg_lateness = safe_div(
        sum(
            max(0.0, (order.delivery_time - order.deadline) / 60.0)
            for order in delivered
            if order.delivery_time is not None
        ),
        len(delivered),
    )
    avg_fulfillment = safe_div(
        sum((order.delivery_time - order.created_at) / 60.0 for order in delivered if order.delivery_time is not None),
        len(delivered),
    )
    # Average wait over ALL orders: creation -> take-off for dispatched orders;
    # an order that expires waited its whole deadline window without ever being
    # dispatched (creation -> deadline). No survivorship bias from expiries.
    def _wait_min(order) -> float:
        if order.dispatch_time is not None:
            return (order.dispatch_time - order.created_at) / 60.0
        return (order.deadline - order.created_at) / 60.0
    avg_wait = safe_div(sum(_wait_min(order) for order in all_orders), generated)
    # Dispatch responsiveness: creation -> take-off over dispatched orders only
    # (in a closed episode dispatched == delivered). Unlike avg_wait this is not
    # driven by the deadline window of expired orders.
    dispatched = [order for order in all_orders if order.dispatch_time is not None]
    avg_dispatch_wait = safe_div(
        sum((order.dispatch_time - order.created_at) / 60.0 for order in dispatched),
        len(dispatched),
    )

    drone_distance = sum(drone.total_distance_m for drone in drones.values())
    drone_energy = sum(drone.total_energy for drone in drones.values())
    battery_values = (
        list(batteries.values()) if isinstance(batteries, dict) else list(batteries or [])
    )
    gross_value = sum(order.value for order in delivered)
    on_time_gmv = sum(order.value for order in on_time)
    cost_config = config["cost"]
    delay_penalty = sum(
        max(0.0, (order.delivery_time - order.deadline) / 60.0)
        * float(cost_config["lateness_penalty_per_min"])
        for order in delivered
        if order.delivery_time is not None
    )
    launchpad_values = list(launchpads.values()) if isinstance(launchpads, dict) else list(launchpads or [])
    deployment_cost = sum(launchpad.deployment_cost for launchpad in launchpad_values)
    operational_cost = sum(launchpad.op_cost_per_second * launchpad.active_time_s for launchpad in launchpad_values)
    service_cost = sum(launchpad.service_cost_per_event * launchpad.service_events for launchpad in launchpad_values)
    drone_energy_cost = drone_energy * float(
        cost_config.get("drone_energy_cost_per_Wh", 0.0)
    )
    # Full-cycle equivalents: cumulative discharged energy over capacity —
    # the physical cycle count (two 50% discharges = 1 cycle), immune to the
    # shallow-charge / interrupted-charge distortions of an event counter.
    battery_fce = sum(
        battery.total_discharged_Wh / battery.capacity_Wh
        for battery in battery_values
        if battery.capacity_Wh > 0
    )
    battery_total_discharged_Wh = sum(
        battery.total_discharged_Wh for battery in battery_values
    )
    # ---- A_profit relations (exact identities; float tolerance only for summation order) ----
    late_gmv = sum(order.value for order in late)
    net_profit = (
        gross_value - delay_penalty - drone_energy_cost - deployment_cost - operational_cost - service_cost
    )
    _tol = 1e-6 * max(1.0, abs(gross_value))
    assert abs(on_time_gmv + late_gmv - gross_value) <= _tol, (
        f"on_time_gmv + late_gmv != gross_delivered_value ({on_time_gmv} + {late_gmv} != {gross_value})"
    )
    assert on_time_gmv <= gross_value + _tol, (on_time_gmv, gross_value)
    _rate = float(cost_config["lateness_penalty_per_min"])
    assert abs(avg_lateness * len(delivered) * _rate - delay_penalty) <= 1e-6 * max(1.0, abs(delay_penalty)), (
        f"avg_lateness_min * delivered * rate != delay_penalty ({avg_lateness} * {len(delivered)} * {_rate} != {delay_penalty})"
    )
    assert delay_penalty >= 0.0, delay_penalty
    if float(cost_config["lateness_penalty_per_min"]) > 0.0:
        # late means delivery_time > deadline (integer seconds), so any late order carries a positive penalty
        assert (delay_penalty > 0.0) == (len(late) > 0), (delay_penalty, len(late))
    assert abs(
        net_profit - (gross_value - delay_penalty - drone_energy_cost - deployment_cost - operational_cost - service_cost)
    ) <= _tol
    # ---- E1_orders relations ----
    on_time_rate = safe_div(len(on_time), len(delivered))
    on_time_order_rate = safe_div(len(on_time), generated)
    assert abs(on_time_order_rate - on_time_rate * completion_rate) <= 1e-9, (
        f"on_time_order_rate != on_time_rate * completion_rate ({on_time_order_rate} != {on_time_rate} * {completion_rate})"
    )
    if not unfinished and not pending:
        # closed episode: every dispatched order has landed
        assert len(dispatched) == len(delivered), (len(dispatched), len(delivered))

    return {
        "orders_generated": generated,
        "orders_delivered": len(delivered),
        "orders_expired": len(expired),
        "orders_unfinished": len(unfinished),
        "orders_pending": len(pending),
        "completion_rate": completion_rate,
        "expired_rate": expired_rate,
        "on_time_rate": on_time_rate,
        "on_time_order_rate": on_time_order_rate,
        "avg_lateness_min": avg_lateness,
        "avg_fulfillment_time_min": avg_fulfillment,
        "avg_wait_time_min": avg_wait,
        "avg_dispatch_wait_min": avg_dispatch_wait,
        "drone_total_distance_m": drone_distance,
        "avg_distance_per_order_m": safe_div(drone_distance, len(delivered)),
        "drone_total_energy_Wh": drone_energy,
        "drone_energy_cost": drone_energy_cost,
        "battery_total_discharged_Wh": battery_total_discharged_Wh,
        "battery_full_cycle_equivalents": battery_fce,
        "battery_avg_full_cycle_equivalents": safe_div(battery_fce, len(battery_values)),
        "battery_final_charged": sum(
            battery.status == BatteryStatus.CHARGED for battery in battery_values
        ),
        "battery_final_charging": sum(
            battery.status == BatteryStatus.CHARGING for battery in battery_values
        ),
        "battery_final_in_use": sum(
            battery.status == BatteryStatus.IN_USE for battery in battery_values
        ),
        "on_time_gmv": on_time_gmv,
        "gross_delivered_value": gross_value,
        "delay_penalty_cost": delay_penalty,
        "launchpad_deployment_cost": deployment_cost,
        "launchpad_operational_cost": operational_cost,
        "launchpad_service_cost": service_cost,
        "net_profit": net_profit,
        "late_gmv": late_gmv,
    }
