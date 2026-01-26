import random
from map.generation import generate_board
from engine.state import GameState, PlayerState 

def setup(
    seed: int,
    n_players: int = 4,
) -> GameState:
    
    rng = random.Random(seed)
    board = generate_board(rng)
    players = [PlayerState() for _ in range(n_players)]
    desert_hex = int((board.hex_terrain == 0).nonzero()[0][0])

    return GameState(
        board=board,
        players=players,
        robber_hex=desert_hex,
        turn=0,
        rng=rng,
    )