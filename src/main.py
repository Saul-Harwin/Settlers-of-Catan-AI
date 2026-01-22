import random
from map.generation import generate_board
from engine.state import PlayerState, GameState
from tools.randomize_player_state import randomize_player_state
from tools.visualiser import draw
from engine.simulate import simulate_game

rng = random.Random(32)
board = generate_board(rng)

players = [PlayerState() for _ in range(4)]
# randomly populate settlements/cities/roads for testing
# for p in players:
#     randomize_player_state(p, rng=rng)


robber_hex = rng.randint(0, 18)
turn = 0

game_state = GameState(board, players, robber_hex, turn, rng)
# draw(game_state)

# simulate 5 turns
simulate_game(game_state, n_turns=9, visualise=True)

# visualise final state
# draw(game_state)