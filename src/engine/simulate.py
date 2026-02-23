import matplotlib.pyplot as plt
import copy
import numpy as np
import random
from dataclasses import dataclass
from enum import IntEnum

from engine.state import GameState
from engine.action import GameAction, BuildRoad, BuildSettlement, BuildCity, EndTurn
from engine.rules import get_legal_edges, get_legal_settlement_vertices, get_upgradeable_cities, legal_actions

from logic.strategies import RandomStrategy, HeuristicStrategy

from tools.visualiser import draw, draw_many_states

from map.helpers import adjacent_hexes
from map.geometry import EDGE_VERTEX_INDICES, TILE_VERTICES, VERTEX_NEIGHBORS, PIP_WEIGHT

@dataclass
class EvalWeights:
    vp_weight: float = 10.0
    settlement_weight: float = 2.0
    city_weight: float = 3.0
    road_weight: float = 1.0
    resource_weight: float = 0.5
    
class Resource(IntEnum):
    WOOD  = 0
    BRICKS  = 1
    SHEEP = 2
    WHEAT = 3
    ROCK  = 4

def seed_starting_positions(state, player_idx, strategy, rng):
    """
    Place starting settlement + adjacent road.
    Uses full forward evaluation if strategy supports it.
    Falls back to random otherwise.
    """

    from engine.simulate import evaluate_state, EvalWeights

    player = state.players[player_idx]

    # All legal settlement vertices (no connection required during setup)
    free_vertices = get_legal_settlement_vertices(
        state, player_idx, require_connection=False
    )

    if not free_vertices:
        return

    # ------------------------------------------------------------------
    # RANDOM STRATEGY
    # ------------------------------------------------------------------
    if not hasattr(strategy, "select_action"):
        best_vertex = rng.choice(list(free_vertices))
        player.settlements.add(best_vertex)
        player.victory_points += 1
        _place_random_adjacent_road(state, player_idx, best_vertex, rng)
        return

    # ------------------------------------------------------------------
    # HEURISTIC / EVALUATION-BASED STRATEGY
    # ------------------------------------------------------------------

    best_combo = None
    best_score = -float("inf")
    weights = EvalWeights()

    for vertex in free_vertices:

        # Find all free adjacent roads for this vertex
        neigh_vertices = VERTEX_NEIGHBORS[vertex]
        candidate_edges = []

        occupied_edges = set().union(*(p.roads for p in state.players))

        for nv in neigh_vertices:
            edge_tuple = (min(vertex, nv), max(vertex, nv))
            for idx, (v1, v2) in enumerate(EDGE_VERTEX_INDICES):
                if (v1, v2) == edge_tuple and idx not in occupied_edges:
                    candidate_edges.append(idx)

        if not candidate_edges:
            continue

        # Evaluate each (settlement + road) combination
        for edge_idx in candidate_edges:

            hypothetical = state.copy()

            hp = hypothetical.players[player_idx]
            hp.settlements.add(vertex)
            hp.victory_points += 1
            hp.roads.add(edge_idx)

            score = evaluate_state(hypothetical, player_idx, weights)

            if score > best_score:
                best_score = score
                best_combo = (vertex, edge_idx)

    if best_combo is None:
        return

    # Apply best found combination
    best_vertex, best_edge = best_combo
    player.settlements.add(best_vertex)
    player.victory_points += 1
    player.roads.add(best_edge)

def simulate_game(game_state: GameState, n_turns: int = 10, visualise: bool = True, rng: random.Random = random) -> GameState:
    game_states = []
    strategies = [RandomStrategy(), RandomStrategy(), RandomStrategy(), HeuristicStrategy()]
    
    
    for i in range(len(game_state.players)):
        seed_starting_positions(game_state, i, strategies[i], rng)

    draw(game_state)

    for i in range(len(game_state.players)):
        seed_starting_positions(game_state, i, strategies[i], rng)
    
    draw(game_state)
    
    starting_resources(game_state)
    
    game_states.append(game_state.copy())
    
    for turn in range(n_turns):
        print(turn)
        print(game_state)
        roll_val = roll(rng)
        
        if roll_val == 7:
            print("Robber activated! Moving robber to random hex. (No stealing implemented yet)")
            move_robber(game_state, new_hex=game_state.rng.randint(0, 18))      # Need to implement stealing logic first, otherwise just leave robber in place
            
        print(f"Turn {game_state.turn}: Player {game_state.turn % len(game_state.players) + 1} rolled a {roll_val}")
        
        resource_production(game_state, roll_val)

        while game_state.turn == turn:
            step(game_state, strategies, rng)
            game_states.append(game_state.copy())
        
        if any(p.victory_points >= 10 for p in game_state.players):
            print("Game over!")
            break

    if visualise:
        draw(game_states)

    return game_states
    
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
                    
def step(state: GameState, strategies, rng):
    player_idx = state.get_current_player_idx()

    # policy = policies[player_idx]
    strategy = strategies[player_idx]    # for testing, use same policy for all players
    chosen = strategy.select_action(state, rng)

    actions = legal_actions(state, player_idx)
    print(f"Player {player_idx + 1} chooses action: {chosen} out of {actions}")

    production = compute_production_profile(state, state.players[player_idx])
    prod_values = np.array(production)
    
    state = execute_action(state, player_idx, chosen)

def build_road(state: GameState, player_idx: int, edge: int):
    # [wood, brick, sheep, wheat, ore]
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
    # [wood, brick, sheep, wheat, ore]
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
    # [wood, brick, sheep, wheat, ore]
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

def evaluate_state(state, player_idx: int, w: EvalWeights) -> float:
    player = state.players[player_idx]
    score = 0.0

    # Victory points
    score += w.vp_weight * player.victory_points

    # Structures
    score += w.settlement_weight * len(player.settlements)
    score += w.city_weight * len(player.cities)
    score += w.road_weight * len(player.roads)

    # Resources --------------------------------------
    # score += w.resource_weight * sum(player.resources)
    
    
    # 1. Resource Production Potential
    for vertex in player.settlements | player.cities:
        for hex_idx in adjacent_hexes(vertex):
            if hex_idx == state.robber_hex:
                continue
            number = state.board.hex_numbers[hex_idx]
            # Each hex contributes PIP_WEIGHT[number] to expected production
            score += w.resource_weight * PIP_WEIGHT.get(number, 0)


    # 2. Resource Diversity
    production = compute_production_profile(state, player)
    prod_values = np.array(production)

    # print(prod_values)

    # Count resources with meaningful production
    diversity = np.count_nonzero(prod_values > 0)

    score += w.resource_weight * diversity

    # Phase adjustment
    score *= 1.0 + 0.01 * state.turn

    return score

def apply_action(state: GameState, player_idx: int, action: GameAction) -> GameAction:
        # Local import avoids circular dependency
        from engine.state import GameState  
        
        new_state = copy.deepcopy(state)
        # player = new_state.players[player_idx]
        
        if isinstance(action, BuildRoad):
            build_road(new_state, player_idx, action.edge)

        elif isinstance(action, BuildSettlement):
            build_settlement(new_state, player_idx, action.vertex)

        elif isinstance(action, BuildCity):
            build_city(new_state, player_idx, action.vertex)

        elif isinstance(action, EndTurn):
            new_state.turn += 1

        else:
            raise ValueError("Unknown action type")
        
        return new_state

def execute_action(state, player_idx, action):
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
    
def _place_random_adjacent_road(state, player_idx, vertex, rng):
    neigh_vertices = VERTEX_NEIGHBORS[vertex]
    occupied_edges = set().union(*(p.roads for p in state.players))

    incident_edges = []
    for nv in neigh_vertices:
        edge_tuple = (min(vertex, nv), max(vertex, nv))
        for idx, (v1, v2) in enumerate(EDGE_VERTEX_INDICES):
            if (v1, v2) == edge_tuple and idx not in occupied_edges:
                incident_edges.append(idx)

    if incident_edges:
        state.players[player_idx].roads.add(rng.choice(incident_edges))
    
def compute_production_profile(state, player):
    board = state.board
    production = np.zeros(5, dtype=np.uint8)

    # Settlements
    for vertex in player.settlements:
        for hex_idx in adjacent_hexes(vertex):
            if hex_idx == state.robber_hex:
                continue
            
            resource = board.hex_terrain[hex_idx] - 1
            number   = board.hex_numbers[hex_idx]
            
            # If number is 0 then skip because it is a dessert tile
            if number == 0:
                continue
            
            production[resource] += PIP_WEIGHT.get(number, 0)

    # Cities (double production)
    for vertex in player.cities:
        for hex_idx in adjacent_hexes(vertex):
            if hex_idx == state.robber_hex:
                continue

            resource = board.hex_terrain[hex_idx] - 1
            number   = board.hex_numbers[hex_idx]
            
            # If number is 0 then skip because it is a dessert tile
            if number == 0:
                continue
            
            production[resource] += 2 * PIP_WEIGHT.get(number, 0)

    return production


