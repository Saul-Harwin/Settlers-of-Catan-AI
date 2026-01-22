from engine.state import GameState
from simulation.strategies import random_strategy
from tools.visualiser import draw, draw_many_states
import matplotlib.pyplot as plt

def simulate_game(game_state: GameState, n_turns: int = 10, visualise: bool = True) -> GameState:
    game_states = []
    
    print(game_state)
    for t in range(n_turns):
        
        for player in game_state.players:
            random_strategy(player, game_state)
        
        print(game_state)
        game_states.append(game_state.copy())
        
        game_state.turn += 1
        
        
    if visualise:
        # for state in game_states:
        # print(state)
        draw(game_states)
        # fig, ax = plt.subplots(figsize=(10, 10))  # adjusted aspect ratio
        # draw_many_states(game_states, ax)
            
    return game_state
