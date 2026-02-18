import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class Board:
    """
    Immutable description of a Catan map layout.
    Contains structure only, no player-dependent information.
    """
    
    hex_terrain: np.ndarray  # shape (19,), dtype=np.uint8
    hex_numbers: np.ndarray  # shape (19,), dtype=np.uint8