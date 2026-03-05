# geometry.py
# Canonical geometry for standard 19-hex Settlers of Catan board
# Orientation: POINTY-TOP hexes

import math

# ---------------------------------------------------------------------
# Hex constants
# ---------------------------------------------------------------------

HEX_RADIUS = 1.0
H_SPACING = math.sqrt(3) * HEX_RADIUS       # horizontal spacing between hex centers
V_SPACING = 1.5 * HEX_RADIUS                # vertical spacing between hex centers
R = HEX_RADIUS
H = HEX_RADIUS * math.sqrt(3) / 2           # horizontal offset to vertices from center
V = HEX_RADIUS / 2                           # vertical offset to vertices from center

# ---------------------------------------------------------------------
# Hex centers (19)
# Layout: 3–4–5–4–3
# ---------------------------------------------------------------------

HEX_CENTERS = [
    (-H_SPACING,  V_SPACING * 2),
    (0.0,         V_SPACING * 2),
    (H_SPACING,   V_SPACING * 2),

    (-1.5 * H_SPACING,  V_SPACING),
    (-0.5 * H_SPACING,  V_SPACING),
    ( 0.5 * H_SPACING,  V_SPACING),
    ( 1.5 * H_SPACING,  V_SPACING),

    (-2.0 * H_SPACING,  0.0),
    (-1.0 * H_SPACING,  0.0),
    ( 0.0,                       0.0),
    ( 1.0 * H_SPACING,  0.0),
    ( 2.0 * H_SPACING,  0.0),

    (-1.5 * H_SPACING, -V_SPACING),
    (-0.5 * H_SPACING, -V_SPACING),
    ( 0.5 * H_SPACING, -V_SPACING),
    ( 1.5 * H_SPACING, -V_SPACING),

    (-H_SPACING, -V_SPACING * 2),
    (0.0,                 -V_SPACING * 2),
    (H_SPACING,  -V_SPACING * 2),
]

# ---------------------------------------------------------------------
# Vertex coordinates (54)
# Derived from pointy-top geometry
# ---------------------------------------------------------------------

VERTEX_COORDS = [
    # Row 1 — 3 hexes → 7 vertices
    (-H_SPACING - H, 2*V_SPACING + V), (-H_SPACING, 2*V_SPACING + 2*V), ( -H, 2*V_SPACING + V), (0.0, 2*V_SPACING + 2*V),
    (H, 2*V_SPACING + V), (H_SPACING, 2*V_SPACING + 2*V), (H_SPACING + H, 2*V_SPACING + V),

    # Row 2 — 4 hexes → 9 vertices
    (-H_SPACING - 2*H, V_SPACING + V), (-H_SPACING -H, V_SPACING + 2*V), ( -2*H, V_SPACING + V), (-H, V_SPACING + 2*V), 
    (0.0, V_SPACING + V), (H, V_SPACING + 2*V), (2*H, V_SPACING + V), (H_SPACING + H, V_SPACING + 2*V), (H_SPACING + 2*H, V_SPACING + V),

    # Row 3 — 5 hexes → 11 vertices
    (-2*H_SPACING - H, V), (-2*H_SPACING, 2*V), (-H_SPACING - H, V), (-H_SPACING, 2*V), ( -H, V), (0.0, 2*V),
    (H, V), (H_SPACING, 2*V), (H_SPACING + H, V), (2*H_SPACING, 2*V), (2*H_SPACING + H, V),

    # Row 4 — 5 hexes → 11 vertices
    (-2*H_SPACING - H, -V), (-2*H_SPACING, -2*V), (-H_SPACING - H, -V), (-H_SPACING, -2*V), ( -H, -V), (0.0, -2*V),
    (H, -V), (H_SPACING, -2*V), (H_SPACING + H, -V), (2*H_SPACING, -2*V), (2*H_SPACING + H, -V),

    # Row 5 — 4 hexes → 9 vertices
    (-H_SPACING - 2*H, -V_SPACING - V), (-H_SPACING -H, -V_SPACING - 2*V), ( -2*H, -V_SPACING - V), (-H, -V_SPACING - 2*V), 
    (0.0, -V_SPACING - V), (H, -V_SPACING - 2*V), (2*H, -V_SPACING - V), (H_SPACING + H, -V_SPACING - 2*V), (H_SPACING + 2*H, -V_SPACING - V),

    # Row 6 — 3 hexes → 7 vertices
    (-H_SPACING - H, -2*V_SPACING - V), (-H_SPACING, -2*V_SPACING - 2*V), ( -H, -2*V_SPACING - V), (0.0, -2*V_SPACING - 2*V),
    (H, -2*V_SPACING - V), (2*H, -2*V_SPACING - 2*V), (3*H, -2*V_SPACING - V)
]


# ---------------------------------------------------------------------
# Edge definitions (72)
# ---------------------------------------------------------------------

EDGE_VERTEX_INDICES = [

    # Row 1 horizontal edges
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),

    # Vertical edges — Row1 → Row2
    (0, 8), (2, 10), (4, 12), (6, 14),

    # Row 2 horizontal edges
    (7, 8), (8, 9), (9, 10), (10, 11),
    (11, 12), (12, 13), (13, 14), (14, 15),

    # Vertical edges — Row2 → Row3
    (7, 17), (9, 19), (11, 21), (13, 23), (15, 25),

    # Row 3 horizontal edges
    (16, 17), (17, 18), (18, 19), (19, 20),
    (20, 21), (21, 22), (22, 23), (23, 24),
    (24, 25), (25, 26),

    # Vertical edges — Row3 → Row4
    (16, 27), (18, 29), (20, 31),
    (22, 33), (24, 35), (26, 37),

    # Row 4 horizontal edges
    (27, 28), (28, 29), (29, 30), (30, 31),
    (31, 32), (32, 33), (33, 34), (34, 35),
    (35, 36), (36, 37),

    # Vertical edges — Row4 → Row5
    (28, 38), (30, 40), (32, 42),
    (34, 44), (36, 46),

    # Row 5 horizontal edges
    (38, 39), (39, 40), (40, 41), (41, 42),
    (42, 43), (43, 44), (44, 45), (45, 46),

    # Vertical edges — Row5 → Row6
    (39, 47), (41, 49), (43, 51), (45, 53),

    # Row 6 horizontal edges
    (47, 48), (48, 49), (49, 50),
    (50, 51), (51, 52), (52, 53),
]


VERTEX_NEIGHBORS = {

    # Row 1
    0: {1, 8},
    1: {0, 2},
    2: {1, 3, 10},
    3: {2, 4},
    4: {3, 5, 12},
    5: {4, 6},
    6: {5, 14},

    # Row 2
    7: {8, 17},
    8: {7, 0, 9},
    9: {8, 10, 19},
    10: {9, 2, 11},
    11: {10, 12, 21},
    12: {11, 4, 13},
    13: {12, 14, 23},
    14: {13, 6, 15},
    15: {14, 25},

    # Row 3
    16: {17, 27},
    17: {16, 7, 18},
    18: {17, 19, 29},
    19: {18, 9, 20},
    20: {19, 21, 31},
    21: {20, 11, 22},
    22: {21, 23, 33},
    23: {22, 13, 24},
    24: {23, 25, 35},
    25: {24, 15, 26},
    26: {25, 37},

    # Row 4
    27: {16, 28},
    28: {27, 29, 38},
    29: {28, 18, 30},
    30: {29, 31, 40},
    31: {30, 20, 32},
    32: {31, 33, 42},
    33: {32, 22, 34},
    34: {33, 35, 44},
    35: {34, 24, 36},
    36: {25, 37, 46},
    37: {36, 26},

    # Row 5
    38: {28, 39},
    39: {38, 40, 47},
    40: {39, 30, 41},
    41: {40, 42, 49},
    42: {41, 32, 43},
    43: {42, 44, 51},
    44: {43, 34, 45},
    45: {44, 46, 53},
    46: {45, 36},

    # Row 6
    47: {39, 48},
    48: {47, 49},
    49: {48, 41, 50},
    50: {49, 51},
    51: {50, 43, 52},
    52: {51, 53},
    53: {52, 45},
}

TILE_VERTICES = {
    0: [0, 1, 2, 10, 9, 8],
    1: [2, 3, 4, 12, 11, 10],     
    2: [4, 5, 6, 14, 13, 12],     
    3: [7, 8, 9, 19, 18, 17],     
    4: [9, 10, 11, 21, 20, 19],   
    5: [11, 12, 13, 23, 22, 21],  
    6: [13, 14, 15, 25, 24, 23],  
    7: [16, 17, 18, 29, 28, 27],  
    8: [18, 19, 20, 31, 30, 29],  
    9: [20, 21, 22, 33, 32, 31],  
    10: [22, 23, 24, 35, 34, 33], 
    11: [24, 25, 26, 37, 36, 35], 
    12: [28, 29, 30, 40, 39, 38], 
    13: [30, 31, 32, 42, 41, 40], 
    14: [32, 33, 34, 44, 43, 42],
    15: [34, 35, 36, 46, 45, 44],
    16: [39, 40 , 41, 49, 48, 47],
    17: [41, 42, 43, 51, 50, 49],
    18: [43, 44, 45, 53, 52, 51]
}

PIP_WEIGHT = {
    2: 1, 12: 1,
    3: 2, 11: 2,
    4: 3, 10: 3,
    5: 4, 9: 4,
    6: 5, 8: 5,
}

# ---------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------

TERRAIN_COLORS = {
    0: "#EDC9AF",
    1: "#228B22",
    2: "#B22222",
    3: "#7FFF00",
    4: "#DAA520",
    5: "#708090",
}

PLAYER_COLORS = {
    0: "#4C00FF",
    1: "#FF00F2",
    2: "#00FFDD",
    3: "#FF6A00",
    4: "#2F0033",
    5: "#FFFFFF",
}

# Define action space indices:
ACTION_BUILD_ROAD_START = 0                # 0–71
ACTION_BUILD_SETTLEMENT_START = 72         # 72–125
ACTION_BUILD_CITY_START = 126              # 126–179
ACTION_END_TURN = 180                      # 180
ACTION_TRADE_START = 181                   # 181–200

TOTAL_ACTIONS = 201