from __future__ import annotations
import random
from typing import List, Optional
from .base import BaseDemandGenerator, Demand


class RuleBasedGenerator(BaseDemandGenerator):
	"""Simple random generator.

	Params:
	- max_per_step: int (default 1) max number of new demands per timestep
	- max_c: int (default 1) max demand quantity
	- rng_seed: Optional[int]
	"""

	def reset(self, seed: Optional[int] = None) -> None:
		seed = seed if seed is not None else self.params.get("rng_seed")
		self._rng = random.Random(seed)

	def sample(self, t: int) -> List[Demand]:
		max_per_step = int(self.params.get("max_per_step", 1))
		max_c = int(self.params.get("max_c", 1))
		min_life = int(self.params.get("min_lifetime", 5))
		max_life = int(self.params.get("max_lifetime", 15))
		depot = self.params.get("depot")  # Optional[Tuple[int,int]]
		if max_life < min_life:
			max_life = min_life
		k = self._rng.randint(0, max_per_step)
		out: List[Demand] = []
		for _ in range(k):
			# 采样位置，若与 depot 重合则重采样（若未提供 depot 则不限制）
			attempts = 0
			while True:
				x = self._rng.randrange(self.width)
				y = self._rng.randrange(self.height)
				attempts += 1
				if depot is None or (x, y) != tuple(depot):
					break
				# 防止退化无限循环（例如 1x1 地图），超过一定尝试直接跳过
				if attempts > 100:
					# 无法生成与 depot 不重叠的位置，放弃本次生成
					x, y = None, None
					break
			c = self._rng.randint(1, max_c)
			life = self._rng.randint(min_life, max_life)
			end_t = t + life
			if x is not None and y is not None:
				out.append(Demand(x=x, y=y, t=t, c=c, end_t=end_t))
		return out
