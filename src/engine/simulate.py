import matplotlib.pyplot as plt
import numpy as np
import random

from engine.state import GameState
from engine.action import GameAction, BuildRoad, BuildSettlement, BuildCity, EndTurn
from engine.rules import get_legal_edges, get_legal_settlement_vertices, get_upgradeable_cities, legal_actions

from logic.strategies import random_strategy, RandomStrategy, HeuristicStrategy

from tools.visualiser import draw, draw_many_states

from map.geometry import EDGE_VERTEX_INDICES, TILE_VERTICES, VERTEX_NEIGHBORS

def seed_starting_positions(state, player_idx, strategy, rng):
    """
    Place each player's starting settlement and one adjacent road using the given strategy.
    Works with any strategy that implements `score_vertex(state, vertex)`.
    """
    #  Player
    player = state.players[player_idx]
    
    # Get all legal starting vertices (no connection required)
    free_vertices = get_legal_settlement_vertices(state, player_idx, require_connection=False)
    if not free_vertices:
        return

    # Strategy selects the best vertex
    if hasattr(strategy, "score_vertex"):
        best_vertex = max(free_vertices, key=lambda v: strategy.score_vertex(state, v))
    else:
        # fallback to random selection
        best_vertex = rng.choice(list(free_vertices))

    player.settlements.add(best_vertex)
    player.victory_points += 1

    # Find incident edges for that vertex
    neigh_vertices = VERTEX_NEIGHBORS[best_vertex]
    incident_edges = set()
    for nv in neigh_vertices:
        edge_tuple = (min(best_vertex, nv), max(best_vertex, nv))
        for idx, (v1, v2) in enumerate(EDGE_VERTEX_INDICES):
            if (v1, v2) == edge_tuple:
                incident_edges.add(idx)

    # Only choose edges that are free
    occupied_edges = set().union(*(p.roads for p in state.players))
    free_incident_edges = incident_edges - occupied_edges
    if not free_incident_edges:
        return

    # Strategy selects the best road edge based on potential vertex scores
    if hasattr(strategy, "score_vertex"):
        best_edge = max(
            free_incident_edges,
            key=lambda e: max(
                strategy.score_vertex(state, EDGE_VERTEX_INDICES[e][0]),
                strategy.score_vertex(state, EDGE_VERTEX_INDICES[e][1])
            )
        )
    else:
        best_edge = rng.choice(list(free_incident_edges))

    player.roads.add(best_edge)

def simulate_game(game_state: GameState, n_turns: int = 10, visualise: bool = True, rng: random.Random = random) -> GameState:
    game_states = []
    strategies = [HeuristicStrategy(), HeuristicStrategy(), HeuristicStrategy(), HeuristicStrategy()]
    
    
    for i in range(len(game_state.players)):
        seed_starting_positions(game_state, i, strategies[i], rng)

    draw(game_state)

    for i in range(len(game_state.players)):
        seed_starting_positions(game_state, i, strategies[i], rng)
    
    draw(game_state)
    
    starting_resources(game_state)
    
    game_states.append(game_state.copy())
    
    for _ in range(n_turns):
        print(game_states[_])
        roll_val = roll(rng)
        
        if roll_val == 7:
            print("Robber activated! Moving robber to random hex. (No stealing implemented yet)")
            move_robber(game_state, new_hex=game_state.rng.randint(0, 18))      # Need to implement stealing logic first, otherwise just leave robber in place
            
        print(f"Turn {game_state.turn}: Player {game_state.turn % len(game_state.players) + 1} rolled a {roll_val}")
        
        resource_production(game_state, roll_val)

        step(game_state, [RandomStrategy(), RandomStrategy(), RandomStrategy(), HeuristicStrategy()])
        game_states.append(game_state.copy())
        
        if any(p.victory_points >= 10 for p in game_state.players):
            print("Game over!")
            break

    if visualise:
        draw(game_states)

    return game_states[-1]
    
def starting_resources(state: GameState):
    # [wood, brick, sheep, wheat, rock]
    
    for player in state.players:
        v = player.settlements.copy().pop()
        
        hexes  = [h for h in range(19) if v in TILE_VERTICES[h]]
        
        for h in hexes:
            terrain = state.board.hex_terrain[h]
            if terrain > 0:   # ignore desert
                player.resources[terrain - 1] += 1

def roll(rng) -> int:
    return rng.randint(1, 6) + rng.randint(1, 6)

def resource_production(state: GameState, roll: int) -> None:
    hexes = np.where(state.board.hex_numbers == roll)[0]
    
    for hex_idx in hexes:
        vertices = TILE_VERTICES[hex_idx]
        
        for i, player in enumerate(state.players):
            for v in player.settlements:
                if v in vertices:
                    # [wood, brick, sheep, wheat, rock]
                    print(f"roll: {roll} -> Player {i+1} gets resource from hex {hex_idx} ({state.board.hex_terrain[hex_idx]-1}) for settlement at vertex {v}")
                    
                    player.resources[state.board.hex_terrain[hex_idx]-1] += 1
            for v in player.cities:
                if v in vertices:
                    player.resources[state.board.hex_terrain[hex_idx]-1] += 2
                    
def step(state: GameState, strategies):
    player_idx = state.turn % len(state.players)

    # policy = policies[player_idx]
    strategy = strategies[player_idx]    # for testing, use same policy for all players
    chosen = strategy.select_action(state)

    actions = legal_actions(state, player_idx)
    print(f"Player {player_idx + 1} chooses action: {chosen} out of {actions}")

    apply_action(state, player_idx, chosen)
    
def apply_action(state: GameState, player_idx: int, action: GameAction) -> None:
    
    if isinstance(action, BuildRoad):
        build_road(state, player_idx, action.edge)

    elif isinstance(action, BuildSettlement):
        build_settlement(state, player_idx, action.vertex)

    elif isinstance(action, BuildCity):
        build_city(state, player_idx, action.vertex)

    elif isinstance(action, EndTurn):
        state.turn += 1

    else:
        raise ValueError("Unknown action type")

def build_road(state: GameState, player_idx: int, edge: int):
    # [wood, brick, sheep, wheat, rock]
    ROAD_COST = np.array([1, 1, 0, 0, 0], dtype=np.uint8)
    
    player = state.players[player_idx]
    
    # 1. Legality check
    legal_edges = get_legal_edges(state, player_idx)
    if edge not in legal_edges:
        raise ValueError(f"Illegal road placement: edge {edge}")

    # 2. Resource check
    if not np.all(player.resources >= ROAD_COST):
        raise ValueError("Insufficient resources to build road")
    
    # 3. Charge
    player.resources -= ROAD_COST

    # 4. Mutate
    player.roads.add(edge)
    
def build_settlement(state: GameState, player_idx: int, vertex: int) -> None:
    # [wood, brick, sheep, wheat, rock]
    SETTLEMENT_COST = np.array([1, 1, 1, 1, 0], dtype=np.uint8)
    
    player = state.players[player_idx]

    # Resource check (cheap invariant protection)
    if not np.all(player.resources >= SETTLEMENT_COST):
        raise ValueError("Insufficient resources to build settlement")

    # Charge
    player.resources -= SETTLEMENT_COST

    # Mutate board state
    player.settlements.add(vertex)

    # Victory points
    player.victory_points += 1

def build_city(state: GameState, player_idx: int, vertex: int) -> None:
    # [wood, brick, sheep, wheat, rock]
    CITY_COST = np.array([0, 0, 0, 2, 3], dtype=np.uint8)

    player = state.players[player_idx]

    # Invariant: must already own a settlement there
    if vertex not in player.settlements:
        raise ValueError("Cannot upgrade: no settlement at vertex")

    # Resource check
    if not np.all(player.resources >= CITY_COST):
        raise ValueError("Insufficient resources to build city")

    # Charge
    player.resources -= CITY_COST

    # Replace settlement with city
    player.settlements.remove(vertex)
    player.cities.add(vertex)

    # Victory points
    player.victory_points += 1   # net +1 (settlement already gave +1)

def move_robber(state: GameState, new_hex: int) -> None:
    state.robber_hex = new_hex
