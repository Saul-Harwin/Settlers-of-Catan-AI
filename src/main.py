import random
from map.generation import generate_board
from engine.state import PlayerState, GameState
from tools.randomize_player_state import randomize_player_state
from tools.visualiser import draw
from engine.simulate import simulate_game
import numpy as np

rng = random.Random(234)
board = generate_board(rng)

players = [PlayerState() for _ in range(4)]

<<<<<<< Updated upstream
robber_hex = np.where(board.hex_terrain == 0)[0][0]
print(robber_hex)
=======
robber_hex = rng.randint(0, 18)
>>>>>>> Stashed changes
turn = 0

game_state = GameState(board, players, robber_hex, turn, rng)

<<<<<<< Updated upstream
# simulate 5 turns
game_states = simulate_game(game_state, n_turns=9, visualise=False)


# visualise final state
draw([game_states[-1]])
=======
# simulate n turns
game_state = simulate_game(game_state, n_turns=500, visualise=False)
print(game_state)

draw(game_state)
>>>>>>> Stashed changes
