import random
import numpy as np
from engine.state import PlayerState

def randomize_player_state(player: PlayerState, rng=None):
    """
    Randomly assigns settlements, cities, and roads to a PlayerState for visualisation/testing.
    Does not enforce game rules.
    """
    rng = rng or random.Random()
    
    # random number of settlements/cities/roads
    n_settlements = rng.randint(1, 4)
    n_cities = rng.randint(0, 2)
    n_roads = rng.randint(2, 6)
    
    available_vertices = list(range(54))
    rng.shuffle(available_vertices)
    player.settlements = set(available_vertices[:n_settlements])
    
    remaining_vertices = [v for v in available_vertices if v not in player.settlements]
    rng.shuffle(remaining_vertices)
    player.cities = set(remaining_vertices[:n_cities])
    
    available_edges = list(range(72))
    rng.shuffle(available_edges)
    player.roads = set(available_edges[:n_roads])
