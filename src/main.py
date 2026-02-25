import random
import typer
import numpy as np

from map.generation import generate_board
from engine.state import PlayerState, GameState
from tools.randomize_player_state import randomize_player_state
from tools.visualiser import draw
from engine.simulate import simulate_game

app = typer.Typer()


@app.command()
def main(
    n_turns: int = typer.Option(15, help="Number of turns to simulate"),
    visualise: bool = typer.Option(True, help="Whether to visualise the game"),
    seed: int = typer.Option(45, help="Random seed"),
):
    rng = random.Random(seed)
    board = generate_board(rng)

    players = [PlayerState() for _ in range(4)]

    # place robber on first empty hex (just example)
    robber_hex = np.where(board.hex_terrain == 0)[0][0]
    turn = 0

    game_state = GameState(board, players, robber_hex, turn, rng)
    game_state.state_to_tensor()

    # simulate the game
    game_history = simulate_game(game_state, n_turns=n_turns, visualise=visualise, rng=rng)

    # print final game state
    final_state = game_history[-1]
    print(final_state)

    # draw final board
    draw(final_state)


if __name__ == "__main__":
    app()