"""
spawner.py

Cube spawning module for FRANK Phase 3 VLA.
Grid-based spawn ensures minimum separation between cubes.
"""

import numpy as np


# Three non-overlapping zones — one cube per zone
# Format: ([x_min, x_max], [y_min, y_max])
SPAWN_ZONES = [
    ([0.50, 0.52], [-0.02, 0.01]),   # left zone
    ([0.58, 0.61], [ 0.01, 0.03]),   # right zone
    ([0.52, 0.56], [ 0.07, 0.09]),   # back zone
]


def spawn_no_overlap():
    """
    Assign each cube to a random zone and sample a position within it.
    Zones are non-overlapping so cubes are guaranteed to be separated.
    Returns list of 3 (x, y) positions in shuffled zone order.
    """
    zones = SPAWN_ZONES.copy()
    np.random.shuffle(zones)

    positions = []
    for x_range, y_range in zones:
        x = np.random.uniform(x_range[0], x_range[1])
        y = np.random.uniform(y_range[0], y_range[1])
        positions.append(np.array([x, y]))

    return positions