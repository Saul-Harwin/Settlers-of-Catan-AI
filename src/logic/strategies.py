import numpy as np
import random

from engine.state import GameState, PlayerState
from engine.rules import can_place_settlement, can_place_road, legal_actions, is_vertex_connected_to_network
from engine.action import BuildSettlement, BuildRoad, BuildCity, EndTurn

from map.board import Board
from map.geometry import EDGE_VERTEX_INDICES, TILE_VERTICES, VERTEX_NEIGHBORS, PIP_WEIGHT
from map.helpers import adjacent_hexes

from logic.helpers import get_free_vertices

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
        print(f"Player placed road at edge {EDGE_VERTEX_INDICES[int(e)]}")
        
class RandomStrategy:
    def select_action(self, state):
        player_idx = state.get_current_player_idx()
        actions = legal_actions(state, player_idx)
        return random.choice(actions)

class HeuristicStrategy:
    
    def __init__(self):
        self.road_weight = 0.8
        self.settlement_weight = 10.0
        self.city_weight = 12.0
    
    
    def select_action(self, state):
        player_idx = state.get_current_player_idx()
        
        actions = legal_actions(state, player_idx)
        scored  = [(self.score(state, a), a) for a in actions]
        scored.sort(reverse=True, key=lambda x: x[0]) 
        
        return scored[0][1]

    def score(self, state, action) -> float:
        if isinstance(action, BuildSettlement):
            return self.score_settlement(state, action)

        elif isinstance(action, BuildCity):
            return self.score_city(state, action)

        elif isinstance(action, BuildRoad):
            return self.score_road(state, action)

        # elif isinstance(action, MoveRobber):
        #     return self.score_robber(state, action)

        elif isinstance(action, EndTurn):
            return 0.0

        return 0.0
    
    def score_settlement(self, state, action) -> float:
        # player = state.players[state.get_current_player_idx()].copy()
        score = self.score_vertex(state, action.vertex) * self.settlement_weight
        return score 
           
    def score_road(self, state, action) -> float:
        player = state.players[state.get_current_player_idx()]
        board  = state.board
        edge_idx = action.edge

        score = 0.0
        
        v1, v2 =  EDGE_VERTEX_INDICES[edge_idx]
        
        unlock_score = 0.0
        
        for vertex in (v1, v2):
            if self.progress_to_build_settlement(state, player, vertex):
                settlement_value = self.score_vertex(state, vertex)
                unlock_score = max(unlock_score, settlement_value)

        score += self.road_weight * unlock_score
        
        return score
        
    def progress_to_build_settlement(self, state, player, vertex):
        occupied_vertices = set().union(*(p.settlements | p.cities for p in state.players))
        
        if vertex not in get_free_vertices(state):
            return False

        # Distance Rule Satisfied
        if any(n in occupied_vertices for n in VERTEX_NEIGHBORS[vertex]):
            return False

        # Connected to network
        if is_vertex_connected_to_network(vertex, player):
            return False  # Already connected to the network why build another road there?


        return True

    def score_vertex(self, state, vertex: int) -> float:
        board  = state.board

        # Production Value
        hexes = adjacent_hexes(vertex)
        
        resource_types = set()
        score = 0.0
        
        for hex_idx in hexes:
            
            # Check the robber isn't on this hex
            if hex_idx == state.robber_hex:
                continue
            
            number   = board.hex_numbers[hex_idx]
            resource = board.hex_terrain[hex_idx]
            
            score += PIP_WEIGHT.get(number, 0)
            resource_types.add(resource)
            
        return score 

    def score_city(self, state, action) -> float:
        board  = state.board
        vertex = action.vertex
        
        score = 0.0 
        
        # Production Value
        for hex_idx in adjacent_hexes(vertex):
            
            # Check the robber isn't on this hex
            if hex_idx == state.robber_hex:
                continue
            
            number = board.hex_numbers[hex_idx]
            
            score += PIP_WEIGHT.get(number, 0)
            
        # city doubles production → marginal gain equals original settlement pips
        return self.city_weight * score
        
        