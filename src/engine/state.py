from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import random
import copy
from engine.action import BuildSettlement, BuildRoad, BuildCity, EndTurn, GameAction



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
    
    def get_current_player_idx(self) -> int:
        return self.turn % len(self.players) 
    
    def get_edges(self):
        players = self.players
        # Edges: 0=empty, 1-4=player roads
        edges = np.zeros(72, dtype=np.uint8)
        for player_idx, player in enumerate(players):
            for e in player.roads:
                edges[e] = player_idx + 1
                
        return edges
            
    def is_terminal(self) -> bool:
        return any(p.victory_points >= 10 for p in self.players)

    def get_winner(self) -> int | None:
        for i, p in enumerate(self.players):
            if p.victory_points >= 10:
                return i
        return None
    
    def get_current_player_idx(self) -> int:
        """
        Returns the index of the player whose turn it is.
        Assumes turns cycle through the list of players.
        """
        return self.turn % len(self.players)
    
    def terrain_tensor(self):
        """
        | 0 |      -  These first 6 numbers are for the terrain types 
        | 0 |      -  These first 6 numbers are for the terrain types
        | 0 |      -  These first 6 numbers are for the terrain types
        | 0 |      -  These first 6 numbers are for the terrain types
        | 0 |      -  These first 6 numbers are for the terrain types
        | 0 |      -  These first 6 numbers are for the terrain types
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  These second 10 are for the number
        | 0 |      -  Robber       
        """
        
        
        num_hexes = len(self.board.hex_terrain)
        num_terrain_types = 6                       # 0: dessert, 1: wood, 2: bricks, 3: sheep, 4: wheat, 5: ore
        num_numbers = 10                           # 2, 3, 4, 5, 6, 8, 9, 10, 11, 12  
        number_to_idx = {2:0, 3:1, 4:2, 5:3, 6:4, 8:5, 9:6, 10:7, 11:8, 12:9}
        
        tensor = np.zeros((num_hexes, num_terrain_types + num_numbers + 1), dtype=np.float32)

        for i in range(len(self.board.hex_terrain)):
            number = self.board.hex_numbers[i]
            t_type = self.board.hex_terrain[i]
            
            # Terrain Type
            tensor[i, t_type] = 1.0
            
            # Number 
            if number != 0:
                tensor[i, num_terrain_types + number_to_idx[number]] = 1.0
            
        # Robber
        tensor[self.robber_hex, -1] = 1.0
                
        return tensor
            
    def edge_tensor(self):
        num_edges = 72
        num_players = len(self.players)
        edge_tensor = np.zeros((num_edges, num_players + 1), dtype=np.float32)

        for p_idx, player in enumerate(self.players):
            for edge_idx in player.roads:
                edge_tensor[edge_idx, p_idx] = 1.0

        # Free edge channel
        occupied_edges = set().union(*[player.roads for player in self.players])
        for edge_idx in range(num_edges):
            if edge_idx not in occupied_edges:
                edge_tensor[edge_idx, -1] = 1.0
                
        return edge_tensor
    
    def vertex_tensor(self):
        num_vertices = 54
        num_players = len(self.players)
        
        vertex_tensor = np.zeros((num_vertices, num_players + 2), dtype=np.float32)
        
        for p_idx, player in enumerate(self.players):
            # Settlements
            for v in player.settlements:
                vertex_tensor[v, p_idx] = 1.0  # mark player ownership
                vertex_tensor[v, -2] = 1.0     # settlement flag
                
            # Cities
            for v in player.cities:
                vertex_tensor[v, p_idx] = 1.0  # mark player ownership
                vertex_tensor[v, -1] = 1.0     # city flag
        
        return vertex_tensor
        
    def resources_tensor(self):
        num_players = len(self.players)
        num_resources = 5  # wood, brick, wheat, sheep, ore
        
        resources_tensor = np.zeros((num_players, num_resources), dtype=np.float32)
            
        for player_idx, player in enumerate(self.players):
            resources_tensor[player_idx] = np.array([
                player.resources[0],  # wood
                player.resources[1],  # brick
                player.resources[2],  # wheat
                player.resources[3],  # sheep
                player.resources[4]   # ore
            ], dtype=np.float32)
    
        return resources_tensor
            
    def state_to_tensor(self):
        # print(self.resources_tensor())
        # print(self.terrain_tensor())
        # print(self.edge_tensor())
        # print(self.vertex_tensor())
        
        input_vector = np.concatenate([
            self.terrain_tensor().flatten(),
            self.vertex_tensor().flatten(),
            self.edge_tensor().flatten(),
            self.resources_tensor().flatten()
        ])
        
        print(input_vector)
        return input_vector
        
        
    def __str__(self) -> str:
        start = f"\n-------------------------------------------------------------------------------------------------------------\n                                              Game State: turn={self.turn} \n-------------------------------------------------------------------------------------------------------------\n"
        
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
        
        return "\n".join([start, robber_summary, board_summary, players_summary])
            
class PlayerState:
    def __init__(self):
        self.resources = np.zeros(5, dtype=np.uint8)  # Wood, Bricks, Sheep, Wheat, Ore
        self.settlements = set()
        self.cities = set()
        self.roads = set()
        self.victory_points = 0
        
    def charge(self, cost: np.ndarray) -> None:
        cost = cost.astype(self.resources.dtype)  # ensure same dtype
        if not np.all(self.resources >= cost):
            raise ValueError(f"Player cannot afford cost: resources={self.resources}, cost={cost}")
        
        self.resources -= cost
    
    def give(self, resources: np.ndarray) -> None:
        self.resources += resources
        
    def copy(self) -> PlayerState:
        ps_copy = PlayerState()
        ps_copy.resources = self.resources.copy()
        ps_copy.settlements = set(self.settlements.copy())
        ps_copy.cities = set(self.cities.copy())
        ps_copy.roads = set(self.roads.copy())
        ps_copy.victory_points = self.victory_points
        return ps_copy    