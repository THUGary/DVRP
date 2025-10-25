"""
Compute travel time (traffic time) between two points.

Currently implemented as Manhattan distance (grid-based), but this
is the single place to replace with a more realistic traffic-time model later.
"""
from __future__ import annotations
from typing import Tuple

Point = Tuple[int, int]


def travel_time(a: Point, b: Point) -> int:
    """
    Return the travel time between two grid points.

    Args:
        a: (x, y)
        b: (x, y)

    Returns:
        int: travel time (currently Manhattan distance)
    """
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by)