from map.geometry import *
import numpy as np

def adjacent_hexes(vertex):
    return [tile for tile, vertices in TILE_VERTICES.items() if vertex in vertices]