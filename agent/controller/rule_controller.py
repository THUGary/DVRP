from __future__ import annotations
from typing import Deque, Tuple
from .base import BaseController, Target, Action

class RuleBasedController(BaseController):
    """Greedy controller: move one step toward the head of the target queue.
    If no targets, stay.
    """

    def act(self, current_pos: Tuple[int, int], target_queue: Deque[Target]) -> Action:
        if not target_queue:
            return (0, 0)
        tx, ty = target_queue[0]
        x, y = current_pos
        dx = 0 if tx == x else (1 if tx > x else -1)
        dy = 0 if ty == y else (1 if ty > y else -1)
        # If we're at the target, pop it and stay this turn
        if dx == 0 and dy == 0:
            target_queue.popleft()
            return (0, 0)
        # Enforce 4-directional moves only (no diagonals): if both dx and dy are non-zero,
        # pick one axis to move along. Policy: move along the axis with larger remaining distance;
        # tie-breaker: prefer x-axis. This yields actions in {(0,0),(1,0),(-1,0),(0,1),(0,-1)}.
        if dx != 0 and dy != 0:
            if abs(tx - x) >= abs(ty - y):
                return (dx, 0)
            else:
                return (0, dy)
        return (dx, dy)
