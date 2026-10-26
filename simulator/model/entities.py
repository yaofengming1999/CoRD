from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .enums import (
    VehicleType,
    OrderStatus,
    VehicleStatus,
    StopType,
    LaunchpadStatus,
    BatteryStatus,
)


@dataclass
class Location:
    lat: float  # 纬度
    lon: float  # 经度
    node_id: Optional[int] = None  # 路网节点 id

    def __repr__(self) -> str:
        return f"Location({self.lat:.5f}, {self.lon:.5f})"


@dataclass
class Stop:
    stop_id: str  # stop 的唯一标识
    location: Location  # stop 的位置
    stop_type: StopType  # stop 类型
    related_order_id: Optional[str] = None  # 关联的订单 id
    related_launchpad_id: Optional[str] = None  # 关联的起降点 id
    related_battery_id: Optional[str] = None  # 关联的电池 id；用于记录任务使用电池或换电目标
    earliest_time: Optional[int] = None  # 最早允许到达时间
    latest_time: Optional[int] = None  # 最晚允许到达时间
    planned_arrival_time: Optional[int] = None  # 计划到达时间
    service_time_seconds: Optional[int] = None  # 服务时间（秒）


@dataclass
class Order:
    order_id: str  # 订单 id
    created_at: int  # 订单创建时间
    origin_launchpad_id: str  # 订单所在的起飞点
    destination: Location  # 配送终点
    size: float  # 订单体积/容量需求
    weight: float  # 订单重量
    value: float  # 订单价值/收益
    deadline: int  # 截止时间
    destination_endpoint_id: Optional[str] = None  # instance 中的配送终点 id
    status: OrderStatus = OrderStatus.PENDING  # 当前订单状态
    assigned_drone_id: Optional[str] = None  # 分配到的无人机 id
    selected_battery_id: Optional[str] = None  # 任务选择的电池 id
    return_launchpad_id: Optional[str] = None  # 返航起降点 id
    base_revenue: Optional[float] = None  # 基础收入
    distance_revenue: float = 0.0  # 距离额外收入
    weight_revenue: float = 0.0  # 重量额外收入
    dispatch_time: Optional[int] = None  # 电池匹配并起飞的时间
    delivery_time: Optional[int] = None  # 实际送达时间


@dataclass
class Drone:
    """Drone is the simulator's only vehicle type."""

    vehicle_id: str  # 无人机 id
    current_location: Location  # 当前所在位置
    available_at: int  # 下次可用时间
    capacity: float
    remaining_capacity: float
    speed_mps: float  # 速度（米/秒）
    route_plan: list[Stop] = field(default_factory=list)  # 当前待执行路线/任务序列
    onboard_order_ids: list[str] = field(default_factory=list)  # 当前携带的订单 id 列表
    status: VehicleStatus = VehicleStatus.IDLE  # 当前状态
    current_stop_eta: Optional[int] = None  # 当前目标 stop 的预计到达时间
    current_leg_started_at: Optional[int] = None  # 当前 leg 的实际出发时间
    current_leg_target_stop_id: Optional[str] = None  # 当前 leg 对应的目标 stop
    current_leg_origin: Optional[Location] = None  # 当前 leg 起点
    current_leg_path: list[Location] = field(default_factory=list)  # 当前 leg 几何路径
    current_leg_distance_m: float = 0.0  # 当前 leg 的计费/统计距离
    current_leg_path_distance_m: float = 0.0  # 当前 leg 几何路径长度
    current_leg_travel_time_s: float = 0.0  # 当前 leg 预计移动时间
    current_leg_arrived: bool = False  # 是否已经到达当前 leg 的目标点
    total_distance_m: float = 0.0  # 累计移动距离

    battery_level: Optional[float] = None  # 挂载电池的当前电量；停靠时为 None
    battery_capacity: Optional[float] = None  # 电池总容量
    max_range_m: Optional[float] = None  # 最大航程（米）
    empty_mass_kg: Optional[float] = None  # 无人机空载质量（kg）
    max_payload_kg: Optional[float] = None  # 无人机最大载重（kg）
    current_battery_id: Optional[str] = None  # UAV 飞行/执行任务时装载的电池 id；停靠时为 None
    current_launchpad_id: Optional[str] = None  # 当前所在起降点 id
    total_energy: float = 0.0  # 累计能耗

    @property
    def vehicle_type(self) -> VehicleType:
        return VehicleType.UAV


@dataclass
class Launchpad:
    launchpad_id: str  # 起降点 id
    current_location: Location  # 当前所在位置
    # None means unlimited holding. A finite value is the hard number of
    # docked plus committed-to-land drones allowed at this launchpad.
    drone_capacity: Optional[int] = None
    order_queue: list[str] = field(default_factory=list)  # 当前等待处理的订单队列
    drone_ids: list[str] = field(default_factory=list)  # 当前停靠的无人机 id 列表
    charged_battery_ids: list[str] = field(default_factory=list)  # 当前可直接换上的满电备用电池
    charging_battery_ids: list[str] = field(default_factory=list)  # 当前正在充电的电池
    status: LaunchpadStatus = LaunchpadStatus.STATIONARY  # 当前状态
    reserved_return_slots: int = 0  # 已承诺返航或搬迁到本站的无人机数量
    deployment_cost: float = 0.0  # 部署成本
    op_cost_per_second: float = 0.0  # 每秒运营成本
    service_cost_per_event: float = 0.0  # 每次服务事件成本
    active_time_s: float = 0.0  # 累计活跃时间（秒）
    service_events: int = 0  # 累计服务事件次数

    @property
    def location(self) -> Location:
        return self.current_location

    def has_drone_capacity(self) -> bool:
        return (
            self.drone_capacity is None
            or len(self.drone_ids) + self.reserved_return_slots < self.drone_capacity
        )


@dataclass
class Battery:
    """One physical battery.

    Heterogeneity lives in three fields that used to be global constants:
    ``capacity_Wh``, ``mass_kg`` and ``full_charge_seconds``. A battery pack is
    not free to vary them independently -- a bigger cell stack stores more, masses
    more and takes longer to fill at a given charger power -- so they are issued
    together by ``battery_types.battery_type_ladder`` and stamped here with the
    ``type_id`` they came from.

    ``mass_kg`` and ``full_charge_seconds`` are optional so that an instance
    written before heterogeneity existed still loads: ``None`` means "the
    scenario-wide default", which reproduces the old homogeneous behaviour
    exactly rather than guessing a value.
    """

    battery_id: str  # 电池 id
    capacity_Wh: float  # 电池容量
    current_charge_Wh: float  # 当前电量
    status: BatteryStatus  # 当前电池状态
    current_launchpad_id: Optional[str] = None  # 所在 launchpad；使用中则为 None
    current_vehicle_id: Optional[str] = None  # 所在 UAV；在 launchpad 上则为 None
    charge_started_at: Optional[int] = None  # 本轮充电开始时间；用于计算部分电量
    charge_start_Wh: Optional[float] = None  # 本轮充电开始时电量
    charge_complete_at: Optional[int] = None  # 如果正在充电，预计完成时间
    total_discharged_Wh: float = 0.0  # 累计放电能量；/capacity 即等效满循环 (FCE)
    # --- heterogeneous pack properties; None = fall back to the global default ---
    mass_kg: Optional[float] = None  # 电池自重（kg）；进入能耗模型的等效空载质量
    full_charge_seconds: Optional[int] = None  # 从 0 充满所需秒数；同充电功率下 ∝ 容量
    type_id: Optional[str] = None  # 所属电池型号（battery_type_ladder 的 rung）
