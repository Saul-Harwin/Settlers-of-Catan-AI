from dataclasses import dataclass
from typing import Union
from map.board import Resource

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
class TradeWithBank(Action):
    give_resource: Resource
    give_amount: int
    receive_resource: Resource
    receive_amount: int = 1

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

