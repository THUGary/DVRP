from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Deque, Tuple

Target = Tuple[int, int]
Action = Tuple[int, int]  # (dx, dy) with each in {-1, 0, 1}

class BaseController(ABC):
    """Controller interface: convert a target queue into a local move."""

    def __init__(self, **params) -> None:
        self.params = params

    @abstractmethod
    def act(self, current_pos: Tuple[int, int], target_queue: Deque[Target]) -> Action:
        raise NotImplementedError
