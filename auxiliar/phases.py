from enum import Enum


class PhaseName(Enum):
    """
    Enumeration that specifies the naming convention of phases
    """
    LIQUID = "liquid"
    VAPOUR = "vapour"
    SOLID = "solid"
    WATER = "water"
    SAND = "sand"
