from __future__ import annotations
from typing import List
from .base import BaseDemandGenerator, Demand


class NetDemandGenerator(BaseDemandGenerator):
	"""Placeholder for a learned demand generator.

	Implement training/inference in `models/generator_model/` later.
	"""

	def reset(self, seed=None) -> None:
		self._model = None  # TODO: load model

	def sample(self, t: int) -> List[Demand]:
		# TODO: use model to generate distributions/forecasts
		return []
