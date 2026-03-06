import random
import typer
import numpy as np
import torch

from map.generation import generate_board, initialise_game
from map.geometry import TOTAL_ACTIONS

from engine.state import PlayerState, GameState
from engine.simulate import simulate_game

from tools.randomize_player_state import randomize_player_state
from tools.visualiser import draw

from logic.strategies import HeuristicStrategy

from learning.agent import PPOAgent
from learning.env import CatanEnv
from learning.model import ActorCritic



app = typer.Typer()


@app.command()
def main(
    n_turns: int = typer.Option(15, help="Number of turns to simulate"),
    visualise: bool = typer.Option(True, help="Whether to visualise the game"),
    seed: int = typer.Option(45, help="Random seed"),
    model_name: str = typer.Option(None, help="Specifies a Model to use for a PPO Agent")
):

    if model_name != None:
        # Load model 
        # --------------------
        path     = f".\\learning\\settlers_ppo_training\\{model_name}\\"

        # Define the params
        epsilon     = 3e-4      # Learning Rate
        state_dim   = 1027
        action_dim  = TOTAL_ACTIONS

        # # Define the environment
        env = CatanEnv()

        # Recreate the model architecture
        model = ActorCritic(state_dim=state_dim, action_dim=action_dim)

        # Load parameters
        model.load_state_dict(torch.load(f"{path}model.pth"))

        # Set to evaluation mode
        model.eval()    

        # -----------------------
        
        # Define Agent 
        agent = PPOAgent(
            model=model,
            optimiser=torch.optim.Adam(model.parameters(), lr=epsilon),
            debug=True
        )
    
        strategies = [
            agent, 
            HeuristicStrategy(), 
            HeuristicStrategy(), 
            HeuristicStrategy()
        ]
    
    else: 
        strategies = [
            HeuristicStrategy(), 
            HeuristicStrategy(), 
            HeuristicStrategy(), 
            HeuristicStrategy()
        ]
        
    game_state = initialise_game(seed)

    # simulate the game
    game_history = simulate_game(
        strategies=strategies, 
        env=env,
        game_state=game_state, 
        n_turns=n_turns, 
        visualise=visualise, 
        rng=game_state.rng
    )

    # print final game state
    final_state = game_history[-1]
    print(final_state)

    # draw final board
    draw(final_state)

    

if __name__ == "__main__":
    app()