from engine.state import GameState
from logic.strategies import random_strategy
from tools.visualiser import draw, draw_many_states
import matplotlib.pyplot as plt
import random
import numpy as np
from map.geometry import TILE_VERTICES


def simulate_game(game_state: GameState, n_turns: int = 10, visualise: bool = True) -> GameState:
    game_states = []
    
    print(game_state)
    for t in range(n_turns):
        
        dice_roll(game_state, debug=True)
        
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

def dice_roll(state: GameState, debug: bool):
    # Roll the dice
    dice = np.array([random.choice([1, 2, 3, 4, 5, 6]), random.choice([1, 2, 3, 4, 5, 6])])
    roll = np.sum(dice)    
    
    # Get the tiles that correspond to the roll
    desert_hex = np.where(state.board.hex_terrain == 0)[0]
    hexes = np.where(state.board.hex_numbers == roll)[0]
    
    # Add one to the index if that tile is where the dessert is
    hexes[hexes == desert_hex] += 1    
    
    if debug:
        print("--------------------------------------------------")
        print(f"Dice Roll: {dice[0]}+{dice[1]}={roll}")
        print(f"Hexes: {hexes}")
    
    for hex_idx in hexes:
        if hex_idx == desert_hex:
            continue
        
        resource = state.board.hex_terrain[hex_idx]
        vertices = TILE_VERTICES[hex_idx]
        
        for player_idx, player in enumerate(state.players):
            count = sum(v in player.settlements for v in vertices) + (sum(v in player.cities for v in vertices)) * 2
            
            if count > 0:
                state.players[player_idx].resources[resource-1] += count

                if debug:
                    resource_types = ["Desert", "Clay", "Wood", "Sheep", "Wheat", "Rock"]
                    print(f"Player {player_idx} has received {count} {resource_types[resource]}")

    print("--------------------------------------------------")
                
            
    

    
