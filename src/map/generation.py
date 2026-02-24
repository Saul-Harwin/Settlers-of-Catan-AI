import numpy as np
import random
from .board import Board, STANDARD_PORTS


def generate_board(rng: random.Random) -> Board:
    """
    Generate a random, rule-correct Catan board.
    """

    # --- terrain pool ---
    terrain_pool = (
        [0] * 1 +   # desert
        [1] * 4 +   # forest
        [2] * 4 +   # pasture
        [3] * 4 +   # fields
        [4] * 3 +   # hills
        [5] * 3     # mountains
    )

    rng.shuffle(terrain_pool)
    hex_terrain = np.array(terrain_pool, dtype=np.uint8)

    # --- number pool (no 7) ---
    number_pool = (
        [2] * 1 +
        [3] * 2 +
        [4] * 2 +
        [5] * 2 +
        [6] * 2 +
        [8] * 2 +
        [9] * 2 +
        [10] * 2 +
        [11] * 2 +
        [12] * 1
    )

    rng.shuffle(number_pool)

    hex_numbers = np.zeros(19, dtype=np.uint8)

    num_idx = 0
    for i in range(19):
        if hex_terrain[i] == 0:  # desert
            hex_numbers[i] = 0
        else:
            hex_numbers[i] = number_pool[num_idx]
            num_idx += 1

    return Board(
        hex_terrain=hex_terrain,
        hex_numbers=hex_numbers,
        ports=STANDARD_PORTS
    )
