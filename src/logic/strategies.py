import numpy as np
import random
from dataclasses import dataclass

# from engine.state import GameState, PlayerState
from engine.rules import can_place_settlement, can_place_road, generate_legal_actions, is_vertex_connected_to_network
from engine.action import BuildSettlement, BuildRoad, BuildCity, EndTurn

from map.board import Board
from map.geometry import EDGE_VERTEX_INDICES, TILE_VERTICES, VERTEX_NEIGHBORS, PIP_WEIGHT
from map.helpers import adjacent_hexes

from logic.helpers import get_free_vertices





class RandomStrategy:
    def select_action(self, state, rng):
        player_idx = state.get_current_player_idx()
        actions = generate_legal_actions(state)
        return rng.choice(actions)







class HeuristicStrategy:
    
    def __init__(self):
        self.road_weight = 0.8
        self.settlement_weight = 10.0
        self.city_weight = 12.0
    

    def select_action(self, state, rng):
        from engine.simulate import apply_action, evaluate_state, EvalWeights  # local import avoids circular import

        best_action = None
        best_score = -float('inf')
        weights = EvalWeights()

        player_idx = state.get_current_player_idx()
        
        for action in generate_legal_actions(state):
            # apply_action already returns a copy, so no need for state.copy()
            hypothetical_state = apply_action(state, player_idx, action)
            score = evaluate_state(hypothetical_state, player_idx, weights)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

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
            return True  # Already connected to the network why build another road there?


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
         
