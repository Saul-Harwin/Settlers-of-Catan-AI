import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import map.geometry as g
import numpy as np
def draw_hexes(ax, hex_terrain):
    """
    Draws the 19 Catan hexes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    hex_terrain : np.ndarray (uint8), shape (19,)
        Terrain ID per hex.
    """
    for i, (cx, cy) in enumerate(g.HEX_CENTERS):
        verts = []
        for k in range(6):
            angle = math.pi / 6 + k * math.pi / 3
            verts.append((
                cx + g.HEX_RADIUS * math.cos(angle),
                cy + g.HEX_RADIUS * math.sin(angle)
            ))

        ax.add_patch(Polygon(
            verts,
            closed=True,
            facecolor=g.TERRAIN_COLORS[int(hex_terrain[i])],
            edgecolor="black",
            linewidth=1.0,
            zorder=1
        ))
      
def draw_numbers(ax, hex_numbers):
    """
    Draw dice numbers on each hex.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    hex_numbers : np.ndarray, shape (19,)
        Number token for each hex (0 = desert / no number).
    """
    for i, (cx, cy) in enumerate(g.HEX_CENTERS):
        number = int(hex_numbers[i])
        if number == 0:
            continue  # skip desert

        ax.text(
            cx, cy, str(number),
            ha="center", va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            zorder=2
        )

def draw_roads(ax, edges):
    """
    Draw roads on edges.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    edges : np.ndarray, shape (72,)
        Road ownership (0=no road, 1-4=player).
    """
    
    for edge_idx, owner in enumerate(edges):
        if owner == 0:
            continue

        v1, v2 = g.EDGE_VERTEX_INDICES[edge_idx]
        x1, y1 = g.VERTEX_COORDS[v1]
        x2, y2 = g.VERTEX_COORDS[v2]

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=3,
            color=g.PLAYER_COLORS[owner - 1],
            zorder=3
        )
        
def draw_buildings(ax, vertices):
    """
    Draw settlements and cities.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    vertices : np.ndarray, shape (54,)
        0 = empty, 1-4 = settlement (player 0-3), 5-8 = city (player 0-3)
    """
    for i, val in enumerate(vertices):
        x, y = g.VERTEX_COORDS[i]
        
        if val == 0:
            continue

        if val <= 4:
            player = val - 1
            
            ax.add_patch(plt.Circle(
                (x, y),
                radius=0.18,
                color=g.PLAYER_COLORS[player],
                zorder=4
            ))
            
        else:
            player = val - 5
            
            ax.add_patch(
                plt.Rectangle(
                    (x - 0.18, y - 0.18),
                    width=0.36,
                    height=0.36,
                    color=g.PLAYER_COLORS[player],
                    zorder=4
                )
            )
        
def draw_robber(ax, robber_hex):
    """
    Draw the robber as a black circle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    robber_hex : int
        Hex index 0-18 where robber is located
    """
    cx, cy = g.HEX_CENTERS[robber_hex]
    ax.add_patch(plt.Circle(
        (cx, cy),
        radius=0.3,
        color="black",
        zorder=2
    ))


def draw_vertex_numbers(ax, vertex_numbers):
    """
    Draw vertex numbers for debugging.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    vertex_numbers : np.ndarray, shape (54,)
        Number to draw at each vertex.
    """
    for i, num in enumerate(vertex_numbers):
        x, y = g.VERTEX_COORDS[i]
        ax.text(
            x, y, str(num),
            ha="center", va="center",
            fontsize=12,
            color="red",
            zorder=5
        )


def draw_many_states(game_states, axes):
    """
    Draw multiple GameStates in a grid.
    """

    for ax, state in zip(axes, game_states):
        draw_single_state(state, ax)

    # Hide unused axes
    for ax in axes[len(game_states):]:
        ax.axis("off")

def draw_single_state(game_state, ax):
    """
    Draw a GameState object using your visualiser functions.
    """

    # --- convert GameState / PlayerState into flat arrays ---
    # Hex info comes from the Board
    hex_terrain = game_state.board.hex_terrain
    hex_numbers = game_state.board.hex_numbers

    vertices = game_state.get_vertices()
    edges = game_state.get_edges()
    
    # Robber location
    robber = game_state.robber_hex

    # --- draw ---
    draw_hexes(ax, hex_terrain)
    draw_numbers(ax, hex_numbers)
    draw_vertex_numbers(ax, vertex_numbers=np.arange(54))  # for debugging
    draw_roads(ax, edges)
    draw_buildings(ax, vertices)
    draw_robber(ax, robber)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Turn : {game_state.turn}")
    
    
    
    
def draw(game_states):
    figsize=(12, 12)
    
    # If we are trying to render multiple states
    if len(game_states) > 1:
        cols = 3
        n = len(game_states)
        rows = math.ceil(n / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]
        
        draw_many_states(game_states, axes)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        draw_single_state(game_states[0], ax)


    # Make the figure fullscreen
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()  # works on most backends (TkAgg, Qt5Agg)

    plt.autoscale()
    plt.suptitle("Settlers of Catan - Game State Visualisation")
    plt.tight_layout()
    plt.show()