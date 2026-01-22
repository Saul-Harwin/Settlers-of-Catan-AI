from engine.state import PlayerState, GameState
from map.geometry import EDGE_VERTEX_INDICES, VERTEX_NEIGHBORS 


def can_place_road(player: PlayerState, edge: int, state: GameState) -> bool:
    game_edges = state.get_edges()
    
    # 1. Is edge free?
    if game_edges[edge] != 0:
        return False
    
    v1, v2 = EDGE_VERTEX_INDICES[edge]
    
    # 2. Is it connected to player's roads or settlements?
    if settlement_reachable(player, v1) or settlement_reachable(player, v2):
        return True
    
    return False

def can_place_settlement(player: PlayerState, vertex: int, state: GameState) -> bool:
    game_vertices = state.get_vertices()
    
    # 1. Vertex must be empty
    if game_vertices[vertex] != 0:
        return False
    
    # 2. Two-edge distance rule
    for neighbor in VERTEX_NEIGHBORS[vertex]:
        if game_vertices[neighbor] != 0:
            return False
        
    # 3. Must be connected to player's road (except initial placement)
    if state.turn == 0:
        return True
    elif settlement_reachable(player, vertex):
        return True 
    
    return False

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
            