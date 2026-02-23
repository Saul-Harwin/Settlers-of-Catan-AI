from engine.state import GameState

def get_free_vertices(state: GameState):
    return set(range(54)) - set.union(*(p.settlements | p.cities for p in state.players))
    