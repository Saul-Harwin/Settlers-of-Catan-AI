import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
from enum import IntEnum

class Resource(IntEnum):
    WOOD  = 0
    BRICKS  = 1
    SHEEP = 2
    WHEAT = 3
    ORE  = 4
    
@dataclass(frozen=True)
class Port:
    id: int
    rate: int                 # 2 or 3
    resource: Optional[Resource]   # None = generic 3:1
    vertices: Tuple[int, int]
    
STANDARD_PORTS: Tuple[Port, ...] = (
    Port(0, 3, None,              (0, 1)),
    Port(1, 2, Resource.WOOD,     (3, 4)),
    Port(2, 3, None,              (14, 15)),
    Port(3, 2, Resource.BRICKS,   (26, 37)),
    Port(4, 2, Resource.ORE,      (45, 46)),
    Port(5, 2, Resource.WHEAT,    (50, 51)),
    Port(6, 2, Resource.SHEEP,    (47, 48)),
    Port(7, 3, None,              (28, 38)),
    Port(8, 3, None,              (7, 17)),
)    

@dataclass(frozen=True)
class Board:
    hex_terrain: np.ndarray
    hex_numbers: np.ndarray
    ports: Tuple[Port, ...]

    vertex_to_port: Dict[int, Port] = field(init=False)

    def __post_init__(self):
        vertex_map: Dict[int, Port] = {}
        for port in self.ports:
            for v in port.vertices:
                vertex_map[v] = port

        object.__setattr__(self, "vertex_to_port", vertex_map)


# ================================
# Trade Rate Utility
# ================================

def get_trade_rate(board: Board, player, resource: Resource) -> int:
    """
    Returns best available trade rate for a given resource.
    Precedence:
        2:1 specific port
        3:1 generic port
        4:1 bank default
    """

    rate = 4

    for vertex in player.settlements | player.cities:
        port = board.vertex_to_port.get(vertex)
        if port is None:
            continue

        if port.resource == resource:
            return 2

        if port.resource is None:
            rate = min(rate, 3)

    return rate
    
    
