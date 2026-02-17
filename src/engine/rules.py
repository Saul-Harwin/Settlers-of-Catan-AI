from engine.state import PlayerState, GameState
from map.geometry import EDGE_VERTEX_INDICES, VERTEX_NEIGHBORS, TILE_VERTICES
import numpy as np
from engine.action import GameAction, BuildRoad, BuildSettlement, BuildCity, EndTurn


def can_place_road(player: PlayerState, edge: int, state: GameState) -> bool:    
    game_edges = state.get_edges()
    
    # 1. Is edge free?
    if game_edges[edge] != 0:
        return False
    
    v1, v2 = EDGE_VERTEX_INDICES[edge]
    
    # 2. Check robber isn't blocking it
    if is_robber_blocking(state, (v1, v2)):
        return False
    
    # 3. Is it blocked by someone else's settlement/city?
    other_player_vertices = set()
    for p in state.players:
        if p != player:
            other_player_vertices |= p.settlements | p.cities
            
    if v1 in other_player_vertices or v2 in other_player_vertices:
        return False
    
    # 4. Is it connected to player's roads or settlements?
    if not settlement_reachable(player, v1) and not settlement_reachable(player, v2):
        return False
    
    if state.turn > 0:
        # Only apply this rule after initial turn
        
        # 5. Can they afford it? 
        cost = [1, 1, 0, 0, 0]
        if not can_afford(resources=player.resources, cost=cost):
            return False  
    
    return True

def can_place_settlement(player: PlayerState, vertex: int, state: GameState) -> bool:
    # 1. Vertex must be empty
    game_vertices = state.get_vertices()
    if game_vertices[vertex] != 0:
        return False

    # 2. Check robber isn't blocking it
    if is_robber_blocking(state, vertex):
        return False
    
    # 3. Two-edge distance rule
    for neighbor in VERTEX_NEIGHBORS[vertex]:
        if game_vertices[neighbor] != 0:
            return False
        
        
    if state.turn > 0: 
        # Only apply these rules if not game start
        
        # Can they afford it? 
        cost = [1, 1, 1, 1, 0]
        if not can_afford(resources=player.resources, cost=cost):
            return False      
            
        # Must be connected to player's road 
        if not settlement_reachable(player, vertex):
            return False
    
    return True

def settlement_reachable(player: PlayerState, target: int) -> bool:
    visited = set()
    stack = list(player.settlements | player.cities)
    
    # visted is a set of vertice indicies that we have checked
    
    while stack:
        # Gets the top indice from the top of the stack 
        v = stack.pop()
        
        # If the vertex is the point we are trying to validate then it has succeeded and the vertex is connected to the players network
        if v == target:
            return True
        
        # Make sure we don't get infinite loops
        if v in visited:
            continue
        
        visited.add(v)
        
        for road_idx in player.roads:
            a, b = EDGE_VERTEX_INDICES[road_idx]
            
            # Grow the stack to include the vertices that are neighbouring the players roads if we haven't already checked them.
           
            # If v in our stack is start of a road then add the end of the road to stack    
            if v == a and b not in visited:
                stack.append(b)
            
            # If v in our stack is end of a road then add the start of the road to stack
            elif v == b and a not in visited:
                stack.append(a)
                
    return False
            
def is_robber_blocking(state: GameState, target: int) -> bool:
    covered_vertices = TILE_VERTICES[state.robber_hex]
    covered_roads = []
    
    for i in range(len(covered_vertices)):
        covered_roads.append((covered_vertices[i], covered_vertices[(i+1)%6]))

    # Make sure we check for road going the other direction. 
    covered_roads.extend((b, a) for a, b in covered_roads.copy())    
    
    # If the input has more than one index then it must be a road
    if isinstance(target, int):  # target is a vertex
        if target in covered_vertices:
            return True
        else:
            return False
    else:  # target is a road, expected as a tuple of two vertices
        if target in covered_roads:
            return True
        else:
            return False
    
def legal_actions(state: GameState, player_idx: int) -> list[GameAction]:
    # [wood, brick, sheep, wheat, rock]
    player = state.players[player_idx]
    actions = []
    
    # Roads
    if player.resources[0] > 0 and player.resources[1] > 0:
        for edge in get_legal_edges(state, player_idx):
            actions.append(BuildRoad(edge))
        
    # Settlements
    if player.resources[0] > 0 and player.resources[1] > 0 and player.resources[2] > 0 and player.resources[3] > 0:
        for vertex in get_legal_settlement_vertices(state, player_idx, require_connection=True):
            actions.append(BuildSettlement(vertex))

    # Cities
    if player.resources[3] > 1 and player.resources[4] > 2:
        for vertex in get_upgradeable_cities(state, player_idx):
            actions.append(BuildCity(vertex))
        
    actions.append(EndTurn())

    return actions       
        
def get_legal_edges(state: GameState, player_idx: int) -> set[int]:
    player = state.players[player_idx]

    # All occupied edges
    occupied = set().union(*(p.roads for p in state.players))

    legal = set()

    for edge_idx, (v1, v2) in enumerate(EDGE_VERTEX_INDICES):

        # 1. Must be unoccupied
        if edge_idx in occupied:
            continue

        # 2. Check robber isn't blocking it
        if is_robber_blocking(state, (v1, v2)):
            continue
        
        # 3. Is it connected to player's roads or settlements?
        connects = False
        
        if settlement_reachable(player, v1) or settlement_reachable(player, v2):
            connects = True

        if not connects:
            continue

        # 3. Blocked by opponent settlement?
        if _edge_blocked_by_opponent(state, player_idx, v1, v2):
            continue

        legal.add(edge_idx)

    return legal
    
def get_legal_settlement_vertices(state: GameState, player_idx: int, require_connection=True) -> set[int]:
    player = state.players[player_idx]

    # All occupied vertices
    occupied_vertices = set().union(*(p.settlements | p.cities for p in state.players))

    legal = set()

    for v in range(len(VERTEX_NEIGHBORS)):

        # 1. Must be empty
        if v in occupied_vertices:
            continue

        # 2. Distance rule
        if any(n in occupied_vertices for n in VERTEX_NEIGHBORS[v]):
            continue

        # 3. Must connect to own road
        if require_connection:
            connects = False
            
            for edge in EDGE_VERTEX_INDICES[v]:
                if edge in player.roads:
                    connects = True
                    break

            if not connects:
                continue

        legal.add(v)

    return legal
    
def get_upgradeable_cities(state: GameState, player_idx: int) -> set[int]:
    player = state.players[player_idx]
    return set(player.settlements)
    
def _edge_blocked_by_opponent(state, player_idx, v1, v2):
    for i, p in enumerate(state.players):
        if i == player_idx:
            continue
        if v1 in p.settlements or v1 in p.cities:
            return True
        if v2 in p.settlements or v2 in p.cities:
            return True
    return False

    
    
