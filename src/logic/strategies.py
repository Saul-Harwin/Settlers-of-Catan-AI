import random
from map.board import Board
from engine.state import GameState, PlayerState
from engine.rules import can_place_settlement, can_place_road
from map.geometry import EDGE_VERTEX_INDICES, TILE_VERTICES, VERTEX_NEIGHBORS
import numpy as np

def random_strategy(player: PlayerState, state: GameState):
    """
    Example minimal strategy: randomly place a road, settlement, or city if possible.
    For testing the simulator.
    """
    # We need to add these two variable to a costs dictionary their own file
    road_cost = np.array([1, 1, 0, 0, 0])
    settlement_cost = np.array([1, 1, 1, 1, 0])
    
    
    # For now, just randomly pick an empty allowed vertex for settlement
    free_vertices = set(range(54)) - set.union(*(p.settlements | p.cities for p in state.players)) 
    allowed_vertices = np.array([])
    
    for vertices in free_vertices:
        if can_place_settlement(player, vertices, state):
            allowed_vertices = np.append(allowed_vertices, vertices)

    if allowed_vertices.size > 0:
        v = random.choice(list(allowed_vertices))
        player.settlements.add(int(v))
        print(f"    Placed settlement at vertex {v}")
        
        if state.turn > 0:
            player.charge(cost=settlement_cost)




    # Randomly place a road (just pick any empty allowed edge)
    if state.turn > 0:
        free_edges = set(range(72)) - set.union(*(p.roads for p in state.players))
    else:
        incident_edges = {
            e for e, (v1, v2) in enumerate(EDGE_VERTEX_INDICES)
            if v1 == v or v2 == v
        }

        occupied_edges = set.union(*(p.roads for p in state.players))
        free_edges = incident_edges - occupied_edges
                         
    allowed_edges = np.array([])

    for edge in free_edges:
        if can_place_road(player, edge, state):
            allowed_edges = np.append(allowed_edges, edge)

    if allowed_edges.size > 0:
        e = random.choice(list(allowed_edges))
        player.roads.add(int(e))
<<<<<<< Updated upstream:src/logic/strategies.py
        print(f"    Placed road at edge {EDGE_VERTEX_INDICES[int(e)]}")
        
        if state.turn > 0:
            player.charge(cost=road_cost)
    
=======
        print(f"Player placed road at edge {EDGE_VERTEX_INDICES[int(e)]}")
        
def random_policy(state: GameState, player_idx, legal_actions):
    return random.choice(legal_actions)
>>>>>>> Stashed changes:src/simulation/strategies.py
