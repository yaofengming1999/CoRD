from enum import Enum


class VehicleType(Enum):
    UAV = "uav"


class OrderStatus(Enum):
    PENDING = "pending"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    EXPIRED = "expired"


class VehicleStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    FLYING_LOADED = "flying_loaded"
    FLYING_EMPTY = "flying_empty"
    WAITING = "waiting"


class StopType(Enum):
    DELIVERY = "delivery"
    DRONE_RETURN = "drone_return"


class LaunchpadStatus(Enum):
    STATIONARY = "stationary"
    SERVING = "serving"


class BatteryStatus(Enum):
    CHARGED = "charged"
    CHARGING = "charging"
    IN_USE = "in_use"
