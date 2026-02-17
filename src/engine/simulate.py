<<<<<<< Updated upstream
from engine.state import GameState, PlayerState
from logic.strategies import random_strategy
from tools.visualiser import draw, draw_many_states
import matplotlib.pyplot as plt
import random
import numpy as np
from map.geometry import TILE_VERTICES


def simulate_game(game_state: GameState, n_turns: int = 10, visualise: bool = True) -> GameState:
    game_states = []
        
    for t in range(n_turns):
        print(game_state)
        
        # Initial Turn a bit different people pick first settlement 
        if t == 0:
            for i in range(2):
                for player in game_state.players:
                    random_strategy(player, game_state)
            starting_resources(game_state, player, list(player.settlements)[0])
        
        else:
            dice_roll(game_state, debug=True)
            
            print("\n------------------------  Build Phase  ------------------------")
            for i, player in enumerate(game_state.players):
                print(f"\nPlayer {i}:")
                random_strategy(player, game_state)
        
        game_states.append(game_state.copy())        
        game_state.turn += 1
        
        
    if visualise:
        # for state in game_states:
        # print(state)
        draw(game_states)
        # fig, ax = plt.subplots(figsize=(10, 10))  # adjusted aspect ratio
        # draw_many_states(game_states, ax)
            
    return game_states

def dice_roll(state: GameState, debug: bool):
    # Roll the dice
    dice = np.array([random.choice([1, 2, 3, 4, 5, 6]), random.choice([1, 2, 3, 4, 5, 6])])
    roll = np.sum(dice)    
    
    # Get the tiles that correspond to the roll
    desert_hex = np.where(state.board.hex_terrain == 0)[0]
    hexes = np.where(state.board.hex_numbers == roll)[0]
    
    # Add one to the index if that tile is where the dessert is
    hexes[hexes == desert_hex] += 1    
    
    if debug:
        print("\n-------------------------  Dice Role  -------------------------\n")
        print(f"{dice[0]}+{dice[1]}={roll}")
        print(f"Hexes: {hexes}\n")
    
    for hex_idx in hexes:
        if hex_idx == desert_hex:
            continue
        
        resource = state.board.hex_terrain[hex_idx]
        vertices = TILE_VERTICES[hex_idx]
        
        for player_idx, player in enumerate(state.players):
            count = sum(v in player.settlements for v in vertices) + (sum(v in player.cities for v in vertices)) * 2
            
            if count > 0:
                state.players[player_idx].resources[resource-1] += count

                if debug:
                    resource_types = ["Desert", "Clay", "Wood", "Sheep", "Wheat", "Rock"]
                    print(f"Player {player_idx} has received {count} {resource_types[resource]}")
            
def starting_resources(state: GameState, player: PlayerState, vertex: int) -> None:
    resources = np.zeros(5, dtype=np.uint8)    
    
    for i in range(len(TILE_VERTICES)):
        # If neighbouring
        if vertex in TILE_VERTICES[i]:
            terrain = state.board.hex_terrain[i]
            
            if terrain == 0:
                continue  # desert
            
            terrain -= 1            
            resources[terrain] +=  1
        else:
            continue

    player.give(resources)
    
=======
from engine.state import GameState
from simulation.strategies import random_strategy, random_policy
from tools.visualiser import draw, draw_many_states
import matplotlib.pyplot as plt
from engine.action import GameAction, BuildRoad, BuildSettlement, BuildCity, EndTurn
from engine.rules import get_legal_edges, get_legal_settlement_vertices, get_upgradeable_cities, legal_actions
import random
import numpy as np
from map.geometry import EDGE_VERTEX_INDICES, TILE_VERTICES, VERTEX_NEIGHBORS
    
def seed_starting_positions(state):
    for i, player in enumerate(state.players):
        free_edges = set(range(72)) - set.union(*(p.roads for p in state.players))
        free_vertices = get_legal_settlement_vertices(state, i, require_connection=False)
        
        v = random.choice(list(free_vertices))
        player.settlements.add(v)

        # choose any incident edge
        neighbough_vertices = VERTEX_NEIGHBORS[v]
        
        incident_edges = set()
        for nv in neighbough_vertices:
            edge = (v, nv) if v < nv else (nv, v)
            incident_edges.add(edge)
        
        edge_indices = set(idx for idx, (v1, v2) in enumerate(EDGE_VERTEX_INDICES) if (v1, v2) in incident_edges)
        edge = random.choice(list(edge_indices))
        
        player.roads.add(edge)

def simulate_game(game_state: GameState, n_turns: int = 10, visualise: bool = True) -> GameState:
    game_states = []
    
    seed_starting_positions(game_state)
    seed_starting_positions(game_state)
    
    starting_resources(game_state)
    
    game_states.append(game_state.copy())
    
    for _ in range(n_turns):
        print(game_states[_])
        roll_val = roll()
        
        if roll_val == 7:
            print("Robber activated! Moving robber to random hex. (No stealing implemented yet)")
            move_robber(game_state, new_hex=game_state.rng.randint(0, 18))      # Need to implement stealing logic first, otherwise just leave robber in place
            
        print(f"Turn {game_state.turn}: Player {game_state.turn % len(game_state.players) + 1} rolled a {roll_val}")
        
        resource_production(game_state, roll_val)

        step(game_state, [random_policy])
        game_states.append(game_state.copy())

    if visualise:
        draw(game_states)

    return game_state
    
def starting_resources(state: GameState):
    # [wood, brick, sheep, wheat, rock]
    
    for player in state.players:
        v = player.settlements.copy().pop()
        
        hexes  = [h for h in range(19) if v in TILE_VERTICES[h]]
        
        for h in hexes:
            terrain = state.board.hex_terrain[h]
            if terrain > 0:   # ignore desert
                player.resources[terrain - 1] += 1

def roll() -> int:
    return random.randint(1, 6) + random.randint(1, 6)

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
                    
def step(state: GameState, policies):
    player_idx = state.turn % len(state.players)

    # policy = policies[player_idx]
    policy = policies[0]    # for testing, use same policy for all players
    
    actions = legal_actions(state, player_idx)
    chosen = policy(state, player_idx, actions)

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
>>>>>>> Stashed changes
