from __future__ import annotations
from typing import Literal, TypedDict


class LocationDict(TypedDict):
    lat: float  # 纬度
    lon: float  # 经度


class RouteStopDict(TypedDict, total=False):
    stop_id: str  # 这个 stop 的唯一标识
    location: LocationDict  # stop 的地理位置
    stop_type: str  # delivery / drone_return
    related_order_id: str | None  # 关联的订单 id
    related_launchpad_id: str | None  # 关联的起降点 id
    related_battery_id: str | None  # 关联的电池 id
    earliest_time: int | None  # 最早允许到达时间
    latest_time: int | None  # 最晚允许到达时间
    planned_arrival_time: int | None  # 计划到达时间
    service_time_seconds: int | None  # 在这个 stop 的服务时间（秒）


class OrderStateDict(TypedDict, total=False):
    order_id: str  # 订单 id
    created_at: int  # 订单创建时间
    origin_launchpad_id: str
    destination: LocationDict
    destination_endpoint_id: str | None
    size: float  # 订单体积/容量需求
    weight: float  # 订单重量
    value: float  # 订单总价值/收益
    deadline: int  # 订单截止时间
    status: str  # 订单当前状态
    assigned_drone_id: str | None  # 分配到的无人机 id
    selected_battery_id: str | None
    return_launchpad_id: str | None  # 返航起降点 id
    base_revenue: float | None  # 基础收入
    distance_revenue: float  # 距离额外收入
    dispatch_time: int | None
    delivery_time: int | None  # 实际送达时间


class VehicleStateDict(TypedDict, total=False):
    vehicle_id: str  # 载具 id
    vehicle_type: str  # always uav
    current_location: LocationDict  # 当前所在位置
    available_at: int  # 下次可用时间
    capacity: float  # 总容量
    remaining_capacity: float  # 当前剩余容量
    speed_mps: float  # 速度（米/秒）
    battery_level: float | None  # 挂载电池的当前电量；停靠时为 None
    battery_capacity: float | None  # 电池总容量
    max_range_m: float | None  # 最大航程（米）
    empty_mass_kg: float | None  # 无人机空载质量（kg）
    max_payload_kg: float | None  # 无人机最大载重（kg）
    current_battery_id: str | None  # UAV 飞行/执行任务时装载的电池 id；停靠时为 None
    current_launchpad_id: str | None  # 当前所在起降点 id
    route_plan: list[RouteStopDict]  # 当前待执行路线/任务序列
    onboard_order_ids: list[str]  # 当前携带的订单 id 列表
    status: str  # 载具当前状态
    current_stop_eta: int | None  # 当前目标 stop 的预计到达时间
    current_leg_started_at: int | None  # 当前 leg 的实际出发时间
    current_leg_target_stop_id: str | None  # 当前 leg 对应的目标 stop
    current_leg_progress_ratio: float  # 当前 leg 的进度比例
    current_leg_distance_m: float  # 当前 leg 的计费/统计距离
    current_leg_travel_time_s: float  # 当前 leg 预计移动时间
    total_distance_m: float  # 累计行驶/飞行距离
    total_energy: float  # 累计能耗


class LaunchpadStateDict(TypedDict, total=False):
    launchpad_id: str  # 起降点 id
    launchpad_type: str  # 起降点类型（当前主要是固定起降点）
    current_location: LocationDict  # 当前所在位置
    drone_capacity: int | None  # None 表示不限制；否则为硬停靠容量
    order_queue: list[str]  # 当前在起降点等待处理的订单队列
    drone_ids: list[str]  # 当前停在该起降点的无人机 id 列表
    charged_battery_ids: list[str]  # 当前可用于换电的满电备用电池
    charging_battery_ids: list[str]  # 当前正在充电的电池
    charged_batteries: int  # 满电备用电池数量
    charging_batteries: int  # 正在充电电池数量
    status: str  # 起降点当前状态
    reserved_return_slots: int  # 已承诺返航或搬迁到本站的无人机数量
    deployment_cost: float  # 部署成本
    op_cost_per_second: float  # 每秒运营成本
    service_cost_per_event: float  # 每次服务事件成本
    active_time_s: float  # 累计活跃时间（秒）
    service_events: int  # 累计服务事件次数


class BatteryStateDict(TypedDict, total=False):
    battery_id: str  # 电池 id
    capacity_Wh: float  # 电池容量
    current_charge_Wh: float  # 当前电量
    status: str  # charged / charging / in_use
    current_launchpad_id: str | None  # 所在 launchpad
    current_vehicle_id: str | None  # 所在 UAV
    charge_started_at: int | None  # 本轮充电开始时间
    charge_start_Wh: float | None  # 本轮充电开始时电量
    charge_complete_at: int | None  # 预计充电完成时间
    total_discharged_Wh: float  # 累计放电能量；/capacity 即等效满循环 (FCE)


class WeatherStateDict(TypedDict, total=False):
    pass


class StepStatsDict(TypedDict):
    time: int  # 当前统计对应的时间
    total_orders: int  # 当前系统中的订单总数
    pending: int  # 仍处于 pending 的订单数
    delivered: int  # 已送达订单数
    idle_vehicles: int  # 空闲载具数量
    total_vehicles: int  # 载具总数


class StateDict(TypedDict):
    time: int  # 当前仿真时间
    orders: dict[str, OrderStateDict]  # 全部订单状态，key 是 order_id
    drones: dict[str, VehicleStateDict]  # 全部 drone 状态，key 是 drone_id
    launchpads: dict[str, LaunchpadStateDict]  # 全部起降点状态，key 是 launchpad_id
    batteries: dict[str, BatteryStateDict]  # 全部电池状态，key 是 battery_id
    weather: WeatherStateDict  # 当前天气状态
    stats: StepStatsDict  # 当前全局统计摘要


class TaskLevelActionDict(TypedDict, total=False):
    action_id: str
    action_type: Literal["order_battery_dispatch", "drone_return", "drone_rebalance"]
    order_id: str
    battery_id: str
    drone_id: str
    target_launchpad_id: str  # return/rebalance target, or dispatch-time committed return
    feasibility: bool  # 该任务动作是否可行
