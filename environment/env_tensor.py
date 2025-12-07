from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

from agent.generator.base import BaseDemandGenerator, Demand

Tensor = torch.Tensor
GeneratorFactory = Callable[[], BaseDemandGenerator]
ActionInput = Union[Tensor, Sequence[Sequence[Sequence[int]]]]
SeedSequence = Optional[Sequence[Optional[int]]]


@dataclass
class TensorEnvObservation:
    """Batched observation returned by :class:`TensorGridEnvironment`."""

    time: Tensor  # shape: (batch,)
    depot: Tensor  # shape: (batch, 2)
    agent_pos: Tensor  # shape: (batch, num_agents, 2)
    agent_load: Tensor  # shape: (batch, num_agents)
    agent_service_time: Tensor  # shape: (batch, num_agents)
    demands_pos: Tensor  # shape: (batch, max_demands, 2)
    demands_capacity: Tensor  # shape: (batch, max_demands)
    demands_start: Tensor  # shape: (batch, max_demands)
    demands_end: Tensor  # shape: (batch, max_demands)
    demands_service_time: Tensor  # shape: (batch, max_demands)
    demand_mask: Tensor  # bool mask, shape: (batch, max_demands)

    def active_mask(self) -> Tensor:
        """Mask of demands that are currently visible (t <= time)."""

        time = self.time.view(-1, 1)
        return torch.logical_and(self.demand_mask, self.demands_start <= time)


class TensorGridEnvironment:
    """Tensor-based DVRP environment for batched rollouts on GPU.

    The implementation mirrors the high-level behavior of ``GridEnvironment`` but
    stores the mutable state inside torch tensors. Each batch element behaves as
    an independent environment instance that can be stepped in parallel.

    Parameters
    ----------
    width, height: Grid dimensions.
    num_agents: Number of couriers/vehicles per environment instance.
    capacity: Vehicle capacity (refilled when the agent returns to depot).
    depot: Tuple of (x, y) coordinates used as the starting position.
    batch_size: Number of parallel environment instances to maintain.
    max_demands: Maximum number of outstanding demands stored per batch item.
    generator or generator_factory: Source for demand generation. When a single
        ``BaseDemandGenerator`` instance is provided, it will be deep-copied
        ``batch_size`` times to ensure independent sampling streams.
    device: Torch device (``"cpu"`` or ``"cuda"``) for the internal buffers.
    include_service_time: Enable service duration modeling similar to the scalar
        environment.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        num_agents: int,
        capacity: int,
        depot: Tuple[int, int] = (0, 0),
        batch_size: int = 1,
        max_demands: int = 256,
        generator: Optional[BaseDemandGenerator] = None,
        generator_factory: Optional[GeneratorFactory] = None,
        device: Union[str, torch.device] = "cpu",
        include_service_time: bool = False,
        max_time: int = 5000,
        expiry_penalty_scale: float = 5.0,
        switch_penalty_scale: float = 0.01,
        capacity_reward_scale: float = 10.0,
        wait_penalty_scale: float = 0.001,
        move_penalty_scale: float = 0.0,
        depot_return_bonus_scale: float = 0.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_demands <= 0:
            raise ValueError("max_demands must be positive")

        self.device = torch.device(device)
        self.width = int(width)
        self.height = int(height)
        self.num_agents = int(num_agents)
        self.capacity = int(capacity)
        self.depot_xy = (int(depot[0]), int(depot[1]))
        self.batch_size = int(batch_size)
        self.max_demands = int(max_demands)
        self.include_service_time = bool(include_service_time)
        self.max_time = int(max_time)
        self.expiry_penalty_scale = float(expiry_penalty_scale)
        self.switch_penalty_scale = float(switch_penalty_scale)
        self.capacity_reward_scale = float(capacity_reward_scale)
        self.wait_penalty_scale = float(wait_penalty_scale)
        self.move_penalty_scale = float(move_penalty_scale)
        self.depot_return_bonus_scale = float(depot_return_bonus_scale)

        self._generators = self._create_generators(generator, generator_factory)
        self._alloc_buffers()
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self, seeds: SeedSequence = None) -> TensorEnvObservation:
        """Reset all environment instances."""

        if seeds is not None and len(seeds) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} seeds, got {len(seeds)}")

        # reset generators
        for idx, gen in enumerate(self._generators):
            seed = None if seeds is None else seeds[idx]
            if gen is not None:
                gen.reset(seed)

        self.time = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        self.agent_pos = torch.full(
            (self.batch_size, self.num_agents, 2),
            fill_value=0,
            dtype=torch.long,
            device=self.device,
        )
        self.agent_pos[..., 0] = self.depot_xy[0]
        self.agent_pos[..., 1] = self.depot_xy[1]
        self.agent_load = torch.full(
            (self.batch_size, self.num_agents),
            fill_value=self.capacity,
            dtype=torch.long,
            device=self.device,
        )
        self.agent_service_time = torch.zeros_like(self.agent_load)
        self.agent_servicing_capacity = torch.zeros(
            (self.batch_size, self.num_agents), dtype=torch.float32, device=self.device
        )
        self.prev_actions = torch.zeros(
            (self.batch_size, self.num_agents, 2), dtype=torch.long, device=self.device
        )

        self._clear_demands()
        self._init_stats()
        return self._obs()

    def step(
        self,
        actions: ActionInput,
        *,
        verbose: bool = False,
    ) -> Tuple[TensorEnvObservation, Tensor, Tensor, Dict[str, Tensor]]:
        """Advance all batched environments by one tick."""

        action_tensor = self._normalize_actions(actions)
        served_from_completion = self._progress_services()

        self._spawn_new_demands()
        expired_capacity = self._expire_demands()

        switches = self._compute_switches(action_tensor)
        prev_pos = self.agent_pos.clone()
        prev_load = self.agent_load.clone()

        self._apply_actions(action_tensor)
        movement = self._movement_distance(prev_pos)
        self._refill_capacity(prev_load)
        depot_x, depot_y = self.depot_xy
        prev_at_depot = torch.logical_and(prev_pos[..., 0] == depot_x, prev_pos[..., 1] == depot_y)
        now_at_depot = torch.logical_and(self.agent_pos[..., 0] == depot_x, self.agent_pos[..., 1] == depot_y)
        returned = torch.logical_and(~prev_at_depot, now_at_depot)
        depot_returns = returned.sum(dim=1).to(torch.float32)

        served_from_visits = self._serve_demands()
        raw_capacity = served_from_completion + served_from_visits
        wait_penalty = self._compute_wait_penalty()
        travel_cost = -self.move_penalty_scale * movement
        switch_penalty = -self.switch_penalty_scale * switches
        capacity_term = self.capacity_reward_scale * raw_capacity
        depot_bonus = self.depot_return_bonus_scale * depot_returns

        reward_terms = {
            "service_bonus": capacity_term,
            "travel_cost": travel_cost,
            "waiting_penalty": wait_penalty,
            "depot_return_bonus": depot_bonus,
        }
        reward = capacity_term + wait_penalty + travel_cost + depot_bonus + switch_penalty

        self._update_stats(
            movement=movement,
            switches=switches,
            wait_penalty=wait_penalty,
            travel_cost=travel_cost,
            depot_bonus=depot_bonus,
            switch_penalty=switch_penalty,
            capacity_term=capacity_term,
            served_capacity=raw_capacity,
            expired_capacity=expired_capacity,
        )

        self.prev_actions = action_tensor
        self.time += 1

        done = self._compute_done()
        info = self._build_info(done, verbose, reward_terms)
        return self._obs(), reward, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_generators(
        self,
        generator: Optional[BaseDemandGenerator],
        generator_factory: Optional[GeneratorFactory],
    ) -> List[Optional[BaseDemandGenerator]]:
        gens: List[Optional[BaseDemandGenerator]] = []
        if generator_factory is not None:
            for _ in range(self.batch_size):
                gens.append(generator_factory())
        elif generator is not None:
            for _ in range(self.batch_size):
                gens.append(copy.deepcopy(generator))
        else:
            gens = [None for _ in range(self.batch_size)]
        return gens

    def _alloc_buffers(self) -> None:
        shape = (self.batch_size, self.max_demands)
        self.demands_pos = torch.full(
            (self.batch_size, self.max_demands, 2),
            fill_value=-1,
            dtype=torch.long,
            device=self.device,
        )
        self.demands_capacity = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.demands_start = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.demands_end = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.demands_service_time = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.demand_mask = torch.zeros(shape, dtype=torch.bool, device=self.device)

    def _clear_demands(self) -> None:
        self.demands_pos.fill_(-1)
        self.demands_capacity.zero_()
        self.demands_start.zero_()
        self.demands_end.zero_()
        self.demands_service_time.zero_()
        self.demand_mask.zero_()

    def _init_stats(self) -> None:
        device = self.device
        batch = self.batch_size
        self.stats: Dict[str, Tensor] = {
            "demand_count": torch.zeros(batch, dtype=torch.long, device=device),
            "demand_capacity": torch.zeros(batch, dtype=torch.float32, device=device),
            "served_count": torch.zeros(batch, dtype=torch.long, device=device),
            "served_capacity": torch.zeros(batch, dtype=torch.float32, device=device),
            "expired_capacity": torch.zeros(batch, dtype=torch.float32, device=device),
            "total_distance": torch.zeros(batch, dtype=torch.float32, device=device),
            "switch_count": torch.zeros(batch, dtype=torch.long, device=device),
            "episode_reward": torch.zeros(batch, dtype=torch.float32, device=device),
            "wait_penalty": torch.zeros(batch, dtype=torch.float32, device=device),
            "move_penalty": torch.zeros(batch, dtype=torch.float32, device=device),
            "switch_penalty": torch.zeros(batch, dtype=torch.float32, device=device),
            "capacity_reward_term": torch.zeros(batch, dtype=torch.float32, device=device),
            "service_bonus_term": torch.zeros(batch, dtype=torch.float32, device=device),
            "travel_cost_term": torch.zeros(batch, dtype=torch.float32, device=device),
            "waiting_penalty_term": torch.zeros(batch, dtype=torch.float32, device=device),
            "depot_return_bonus_term": torch.zeros(batch, dtype=torch.float32, device=device),
        }

    def _obs(self) -> TensorEnvObservation:
        depot = torch.full(
            (self.batch_size, 2),
            fill_value=0,
            dtype=torch.long,
            device=self.device,
        )
        depot[:, 0] = self.depot_xy[0]
        depot[:, 1] = self.depot_xy[1]
        return TensorEnvObservation(
            time=self.time.clone(),
            depot=depot,
            agent_pos=self.agent_pos.clone(),
            agent_load=self.agent_load.clone(),
            agent_service_time=self.agent_service_time.clone(),
            demands_pos=self.demands_pos.clone(),
            demands_capacity=self.demands_capacity.clone(),
            demands_start=self.demands_start.clone(),
            demands_end=self.demands_end.clone(),
            demands_service_time=self.demands_service_time.clone(),
            demand_mask=self.demand_mask.clone(),
        )

    def _normalize_actions(self, actions: ActionInput) -> Tensor:
        act = actions
        if not isinstance(actions, torch.Tensor):
            act = torch.as_tensor(actions, device=self.device)
        act = act.to(torch.long)
        if act.ndim == 2:
            act = act.unsqueeze(0)
        if act.shape[0] == 1 and self.batch_size > 1:
            act = act.expand(self.batch_size, -1, -1)
        if act.shape[0] != self.batch_size or act.shape[1] != self.num_agents or act.shape[2] != 2:
            raise ValueError(
                f"Expected actions with shape ({self.batch_size}, {self.num_agents}, 2), got {tuple(act.shape)}"
            )
        act = act.clamp(min=-1, max=1)
        if self.include_service_time:
            mask = self.agent_service_time > 0
            act = act.masked_fill(mask.unsqueeze(-1), 0)
        return act

    def _progress_services(self) -> Tensor:
        if not self.include_service_time:
            return torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        served = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        active_before = self.agent_service_time > 0
        self.agent_service_time = torch.clamp(self.agent_service_time - 1, min=0)
        finished = torch.logical_and(active_before, self.agent_service_time == 0)
        for batch_idx in range(self.batch_size):
            finished_agents = torch.nonzero(finished[batch_idx], as_tuple=False).flatten()
            if finished_agents.numel() == 0:
                continue
            completed = 0
            for agent_idx in finished_agents.tolist():
                cap = float(self.agent_servicing_capacity[batch_idx, agent_idx].item())
                if cap <= 0:
                    continue
                served[batch_idx] += cap
                completed += 1
                self.agent_load[batch_idx, agent_idx] = max(
                    0,
                    int(self.agent_load[batch_idx, agent_idx].item()) - int(round(cap)),
                )
                self.agent_servicing_capacity[batch_idx, agent_idx] = 0.0
            if completed > 0:
                self.stats["served_count"][batch_idx] += completed
        return served

    def _spawn_new_demands(self) -> None:
        for batch_idx, gen in enumerate(self._generators):
            if gen is None:
                continue
            t = int(self.time[batch_idx].item())
            new_demands = gen.sample(t)
            if not new_demands:
                continue
            free_slots = torch.nonzero(~self.demand_mask[batch_idx], as_tuple=False).flatten()
            cursor = 0
            inserted_capacity = 0.0
            for demand in new_demands:
                if cursor >= len(free_slots):
                    break
                slot = int(free_slots[cursor].item())
                cursor += 1
                self._write_demand(batch_idx, slot, demand)
                inserted_capacity += float(demand.c)
            self.stats["demand_count"][batch_idx] += cursor
            self.stats["demand_capacity"][batch_idx] += float(inserted_capacity)

    def _write_demand(self, batch_idx: int, slot: int, demand: Demand) -> None:
        self.demands_pos[batch_idx, slot, 0] = int(demand.x)
        self.demands_pos[batch_idx, slot, 1] = int(demand.y)
        self.demands_capacity[batch_idx, slot] = float(demand.c)
        self.demands_start[batch_idx, slot] = int(demand.t)
        self.demands_end[batch_idx, slot] = int(demand.end_t)
        self.demands_service_time[batch_idx, slot] = int(getattr(demand, "service_time", 0))
        self.demand_mask[batch_idx, slot] = True

    def _expire_demands(self) -> Tensor:
        time = self.time.view(-1, 1)
        active_mask = torch.logical_and(self.demand_mask, self.demands_end >= time)
        expired = torch.logical_and(self.demand_mask, ~active_mask)
        expired_capacity = (self.demands_capacity * expired.float()).sum(dim=1)
        # clear expired entries
        self.demand_mask &= active_mask
        return expired_capacity

    def _compute_switches(self, actions: Tensor) -> Tensor:
        prev = self.prev_actions
        prev_non_zero = torch.any(prev != 0, dim=-1)
        changed = torch.any(actions != prev, dim=-1)
        switches = torch.logical_and(prev_non_zero, changed).sum(dim=1)
        return switches.to(torch.float32)

    def _apply_actions(self, actions: Tensor) -> None:
        new_pos = self.agent_pos + actions
        new_pos[..., 0] = new_pos[..., 0].clamp(0, self.width - 1)
        new_pos[..., 1] = new_pos[..., 1].clamp(0, self.height - 1)
        for batch_idx in range(self.batch_size):
            pos_map: Dict[Tuple[int, int], List[int]] = {}
            for agent_idx in range(self.num_agents):
                pos = (
                    int(new_pos[batch_idx, agent_idx, 0].item()),
                    int(new_pos[batch_idx, agent_idx, 1].item()),
                )
                pos_map.setdefault(pos, []).append(agent_idx)
            for pos, indices in pos_map.items():
                if len(indices) <= 1:
                    continue
                if pos == self.depot_xy:
                    continue
                winner = indices[0]
                for loser in indices[1:]:
                    new_pos[batch_idx, loser] = self.agent_pos[batch_idx, loser]
        self.agent_pos = new_pos

    def _movement_distance(self, prev_pos: Tensor) -> Tensor:
        delta = self.agent_pos - prev_pos
        dist = torch.sqrt(delta.to(torch.float32).pow(2).sum(dim=-1))
        total = dist.sum(dim=1)
        return total

    def _refill_capacity(self, prev_load: Tensor) -> None:
        depot_x, depot_y = self.depot_xy
        at_depot = torch.logical_and(
            self.agent_pos[..., 0] == depot_x,
            self.agent_pos[..., 1] == depot_y,
        )
        self.agent_load = prev_load.clone()
        self.agent_load = torch.where(
            at_depot,
            torch.full_like(self.agent_load, self.capacity),
            self.agent_load,
        )

    def _serve_demands(self) -> Tensor:
        served = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        served_count = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        for batch_idx in range(self.batch_size):
            time = int(self.time[batch_idx].item())
            mask = torch.nonzero(self.demand_mask[batch_idx], as_tuple=False).flatten()
            if mask.numel() == 0:
                continue
            agent_positions: Dict[Tuple[int, int], List[int]] = {}
            for agent_idx in range(self.num_agents):
                pos = (
                    int(self.agent_pos[batch_idx, agent_idx, 0].item()),
                    int(self.agent_pos[batch_idx, agent_idx, 1].item()),
                )
                agent_positions.setdefault(pos, []).append(agent_idx)
            for slot in mask.tolist():
                start_t = int(self.demands_start[batch_idx, slot].item())
                if start_t > time:
                    continue
                pos = (
                    int(self.demands_pos[batch_idx, slot, 0].item()),
                    int(self.demands_pos[batch_idx, slot, 1].item()),
                )
                candidates = agent_positions.get(pos)
                if not candidates:
                    continue
                required = float(self.demands_capacity[batch_idx, slot].item())
                service_time = int(self.demands_service_time[batch_idx, slot].item())
                chosen = None
                for agent_idx in candidates:
                    if self.agent_load[batch_idx, agent_idx] >= required and (
                        not self.include_service_time or self.agent_service_time[batch_idx, agent_idx] == 0
                    ):
                        chosen = agent_idx
                        break
                if chosen is None:
                    continue
                if self.include_service_time and service_time > 0:
                    self.agent_service_time[batch_idx, chosen] = service_time
                    self.agent_servicing_capacity[batch_idx, chosen] = required
                else:
                    self.agent_load[batch_idx, chosen] = max(
                        0,
                        int(self.agent_load[batch_idx, chosen].item()) - int(round(required)),
                    )
                    served[batch_idx] += required
                    served_count[batch_idx] += 1
                self.demand_mask[batch_idx, slot] = False
        self.stats["served_count"] += served_count
        return served

    def _compute_wait_penalty(self) -> Tensor:
        time = self.time.view(-1, 1)
        mask = torch.logical_and(self.demand_mask, self.demands_start <= time)
        total_capacity = (self.demands_capacity * mask.float()).sum(dim=1)
        penalty = -self.wait_penalty_scale * total_capacity
        return penalty

    def _compute_done(self) -> Tensor:
        at_limit = self.time >= self.max_time
        depot_x, depot_y = self.depot_xy
        agents_at_depot = torch.logical_and(
            self.agent_pos[..., 0] == depot_x,
            self.agent_pos[..., 1] == depot_y,
        ).all(dim=1)
        active_demands = torch.logical_and(
            self.demand_mask,
            self.demands_start <= self.time.view(-1, 1),
        )
        empty_demands = ~active_demands.any(dim=1)
        # Done if: time limit reached OR (no active demands AND all agents at depot)
        done = torch.logical_or(at_limit, torch.logical_and(empty_demands, agents_at_depot))
        return done

    def _build_info(self, done: Tensor, verbose: bool, reward_terms: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if verbose:
            finished = torch.nonzero(done, as_tuple=False).flatten()
            for idx in finished.tolist():
                print(f"[TensorEnv] Episode {idx} finished at t={int(self.time[idx].item())}")
        info = {
            "episode_stats": {key: tensor.clone().detach() for key, tensor in self.stats.items()},
            "done_mask": done.clone(),
            "reward_terms": {key: tensor.clone().detach() for key, tensor in reward_terms.items()},
        }
        return info

    def _update_stats(
        self,
        *,
        movement: Tensor,
        switches: Tensor,
        wait_penalty: Tensor,
        travel_cost: Tensor,
        depot_bonus: Tensor,
        switch_penalty: Tensor,
        capacity_term: Tensor,
        served_capacity: Tensor,
        expired_capacity: Tensor,
    ) -> None:
        self.stats["total_distance"] += movement
        self.stats["switch_count"] += switches.to(torch.long)
        self.stats["wait_penalty"] += wait_penalty
        self.stats["move_penalty"] += travel_cost
        self.stats["depot_return_bonus_term"] += depot_bonus
        self.stats["switch_penalty"] += switch_penalty
        self.stats["capacity_reward_term"] += capacity_term
        self.stats["service_bonus_term"] += capacity_term
        self.stats["travel_cost_term"] += travel_cost
        self.stats["waiting_penalty_term"] += wait_penalty
        self.stats["served_capacity"] += served_capacity
        self.stats["expired_capacity"] += expired_capacity
        self.stats["episode_reward"] += (
            capacity_term + wait_penalty + travel_cost + depot_bonus + switch_penalty
        )
