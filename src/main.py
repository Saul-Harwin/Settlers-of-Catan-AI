import random
import typer
import numpy as np

from map.generation import generate_board, initialise_game
from engine.state import PlayerState, GameState
from tools.randomize_player_state import randomize_player_state
from tools.visualiser import draw
from engine.simulate import simulate_game
from logic.strategies import HeuristicStrategy

app = typer.Typer()


@app.command()
def main(
    n_turns: int = typer.Option(15, help="Number of turns to simulate"),
    visualise: bool = typer.Option(True, help="Whether to visualise the game"),
    seed: int = typer.Option(45, help="Random seed"),
):
    game_state = initialise_game(seed)
    
    strategies = [HeuristicStrategy(), HeuristicStrategy(), HeuristicStrategy(), HeuristicStrategy()]

    # simulate the game
    game_history = simulate_game(strategies=strategies, game_state=game_state, n_turns=n_turns, visualise=visualise, rng=game_state.rng)

    # print final game state
    final_state = game_history[-1]
    print(final_state)

    # draw final board
    draw(final_state)

    

if __name__ == "__main__":
    app()