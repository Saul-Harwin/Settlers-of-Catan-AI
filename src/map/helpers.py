from map.geometry import *
import numpy as np

def adjacent_hexes(vertex):
    hexes = [tile for tile, vertices in TILE_VERTICES.items() if vertex in vertices]
    
    print(vertex)
    print("adjecent tile array:", hexes)
    return hexes