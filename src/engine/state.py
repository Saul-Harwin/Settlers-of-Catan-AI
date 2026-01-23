from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import random
# from src.map.board import Board

# region State data structure -----------------------------------------------------------
# tiles ---------------------------------------------------------------------------------
"""
    0: Dessert = 1
    1: Clay    = 3
    2: Wood    = 4
    3: Sheep   = 4
    4: Wheat   = 4
    5: Rock    = 3
"""

#  numbers ------------------------------------------------------------------------------
"""
    2:  1
    3:  2
    4:  2
    5:  2
    6:  2
    7:  2
    8:  2
    9:  2
    10: 2
    11: 2
    12: 1
"""

# vertices ------------------------------------------------------------------------------
"""
    0: empty
    1-4: player settlement
    5-8: player city
"""

# edges ---------------------------------------------------------------------------------
""" 
    0:  empty 
    1-4: player_index
"""

# endregion ------------------------------------------------------------------------------

class GameState:
    """
    Complete dynamic game state for one simulation run.
    """
    
    def __init__(
        self,
        board: Board,
        players: list[PlayerState],
        robber_hex: int,
        turn: int,
        rng: random.Random
        ):
        
        self.board = board
        self.players = players 
        self.robber_hex = robber_hex
        self.turn = turn
        self.rng = rng

    def copy(self) -> GameState:
        rng_copy = random.Random()
        rng_copy.setstate(self.rng.getstate())
        
        return GameState(
            board=self.board,
            players=[p.copy() for p in self.players],
            robber_hex=self.robber_hex,
            turn=self.turn,
            rng=rng_copy
        )
        
    def get_vertices(self):
        players = self.players
        # Vertices: 0=empty, 1-4=settlements, 5-8=cities
        vertices = np.zeros(54, dtype=np.uint8)
        for player_idx, player in enumerate(players):
            for v in player.settlements:
                vertices[v] = player_idx + 1        # settlements: 1-4
            for v in player.cities:
                vertices[v] = player_idx + 5        # cities: 5-8
                        
        return vertices
    
    
    def get_edges(self):
        players = self.players
        # Edges: 0=empty, 1-4=player roads
        edges = np.zeros(72, dtype=np.uint8)
        for player_idx, player in enumerate(players):
            for e in player.roads:
                edges[e] = player_idx + 1
                
        return edges
        
        
    def __str__(self) -> str:
        start = f"\n--------------------------------------\nGame State: turn={self.turn}\n--------------------------------------\n"
        end = f"--------------------------------------\n"
        
        robber_summary = f"  Robber is on hex {self.robber_hex}"
        
        board_summary = f"  Board State:\n" \
                        f"    Board hex numbers ('board.hex_numbers'): {self.board.hex_numbers.tolist()}\n" \
                        f"    Board hex terrain ('board.hex_terrain'): {self.board.hex_terrain.tolist()}\n"
        
        players_summary = ""
        for i, p in enumerate(self.players):
            players_summary += (
                f"  Player {i + 1}:\n"
                f"    Resources ('playerState.resources'): {p.resources.tolist()}\n"
                f"    Settlements ('playerState.settlements'): {sorted(p.settlements)}\n"
                f"    Cities ('playerState.cities'): {sorted(p.cities)}\n"
                f"    Roads ('playerState.roads'): {sorted(p.roads)}\n"
                f"    Victory points ('playerState.victory_points'): {p.victory_points}\n"
            )
        
        return "\n".join([start, robber_summary, board_summary, players_summary, end])
        
class PlayerState:
    def __init__(self):
        self.resources = np.zeros(5, dtype=np.uint8)  # Wheat, Wood, Sheep, Clay, Rock
        self.settlements = set()
        self.cities = set()
        self.roads = set()
        self.victory_points = 0
        
    def copy(self) -> PlayerState:
        ps_copy = PlayerState()
        ps_copy.resources = self.resources.copy()
        ps_copy.settlements = set(self.settlements.copy())
        ps_copy.cities = set(self.cities.copy())
        ps_copy.roads = set(self.roads.copy())
        ps_copy.victory_points = self.victory_points
        return ps_copy    