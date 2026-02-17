from dataclasses import dataclass
from typing import Union

class Action:
    """Marker Base Class"""
    
    pass


@dataclass(frozen=True)
class BuildRoad(Action):
    edge: int


@dataclass(frozen=True)
class BuildSettlement(Action):
    vertex: int


@dataclass(frozen=True)
class BuildCity(Action):
    vertex: int

@dataclass(frozen=True)
class MoveRobber(Action):
    vertex: int

@dataclass(frozen=True)
class EndTurn(Action):
    pass

    
GameAction = Union[
    BuildRoad,
    BuildSettlement,
    BuildCity,
    MoveRobber,
    EndTurn,
]