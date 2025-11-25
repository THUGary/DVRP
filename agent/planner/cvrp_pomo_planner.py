from __future__ import annotations

import importlib.util
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import torch

from .base import AgentState, BasePlanner, Target

SnapshotNode = Tuple[int, int, int, int, int]


def _load_module_from(root: str, filename: str, module_name: str):
    """Dynamically load a module (e.g., CVRPEnv, CVRPModel) from the provided POMO repo."""
    file_path = os.path.join(root, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Could not locate {filename} under {root}. Set Config.cvrp_planner_params['pomo_root'] to your POMO/CVRP/POMO folder."
        )
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build import spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_pomo_modules(pomo_root: str) -> Dict[str, object]:
    root = os.path.abspath(os.path.expanduser(pomo_root))
    env_mod = _load_module_from(root, "CVRPEnv.py", "pomo_cvrp_env")
    model_mod = _load_module_from(root, "CVRPModel.py", "pomo_cvrp_model")
    return {
        "env_cls": getattr(env_mod, "CVRPEnv"),
        "model_cls": getattr(model_mod, "CVRPModel"),
        "reset_state_cls": getattr(env_mod, "Reset_State"),
        "step_state_cls": getattr(env_mod, "Step_State"),
    }


@dataclass
class _Snapshot:
    depot_xy: torch.Tensor  # [1,1,2]
    node_xy: torch.Tensor   # [1,N,2]
    node_demand: torch.Tensor  # [1,N]
    nodes_meta: List[SnapshotNode]
    valid_count: int


class CVRPPOMOPlanner(BasePlanner):
    """Planner wrapper that reuses the POMO CVRP model for DVRP snapshots."""

    def __init__(
        self,
        *,
        pomo_root: str,
        env_params: Dict[str, object],
        model_params: Dict[str, object],
        checkpoint: Optional[str],
        device: str = "cpu",
        max_nodes: int = 100,
        coord_normalizer: Optional[float] = None,
        grid_width: int = 20,
        grid_height: int = 20,
        capacity: int = 200,
        selection_policy: str = "earliest_due",
        **params,
    ) -> None:
        super().__init__(**params)
        self.device = torch.device(device)
        self.max_nodes = int(max_nodes)
        self.coord_normalizer = float(coord_normalizer) if coord_normalizer is not None else float(max(grid_width, grid_height))
        self.selection_policy = selection_policy
        self.capacity = capacity
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive for CVRPPOMOPlanner")
        if self.device.type != "cpu":
            raise ValueError("CVRPPOMOPlanner currently supports CPU inference only; set device='cpu'.")

        modules = _ensure_pomo_modules(pomo_root)
        env_cls = modules["env_cls"]
        model_cls = modules["model_cls"]

        class DVRPEnv(env_cls):  # type: ignore[misc]
            def load_from_tensors(self, depot_xy: torch.Tensor, node_xy: torch.Tensor, node_demand: torch.Tensor, valid_count: int) -> None:
                self.batch_size = depot_xy.size(0)
                if node_xy.size(1) != self.problem_size:
                    raise ValueError(f"Configured problem_size={self.problem_size} but received {node_xy.size(1)} nodes")
                device = depot_xy.device
                self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
                depot_demand = torch.zeros(size=(self.batch_size, 1), device=device, dtype=node_demand.dtype)
                self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
                self.BATCH_IDX = torch.arange(self.batch_size, device=device)[:, None].expand(self.batch_size, self.pomo_size)
                self.POMO_IDX = torch.arange(self.pomo_size, device=device)[None, :].expand(self.batch_size, self.pomo_size)
                self.reset_state.depot_xy = depot_xy
                self.reset_state.node_xy = node_xy
                self.reset_state.node_demand = node_demand
                self.step_state.BATCH_IDX = self.BATCH_IDX
                self.step_state.POMO_IDX = self.POMO_IDX
                self._valid_count = valid_count

            def reset(self):  # type: ignore[override]
                state, reward, done = super().reset()
                valid = getattr(self, "_valid_count", None)
                if valid is not None and valid < self.problem_size:
                    device = self.visited_ninf_flag.device
                    problem = self.problem_size
                    pomo = self.pomo_size
                    mask = torch.arange(problem, device=device)
                    invalid = mask >= valid
                    if invalid.any():
                        expanded = invalid.view(1, 1, problem).expand(self.batch_size, pomo, problem)
                        self.visited_ninf_flag[:, :, 1:][expanded] = float('-inf')
                        self.ninf_mask = self.visited_ninf_flag.clone()
                return state, reward, done

        self._env = DVRPEnv(**env_params)
        self._model = model_cls(**model_params).to(self.device)
        self._model.eval()
        if checkpoint:
            ckpt_path = os.path.abspath(os.path.expanduser(checkpoint))
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"CVRP checkpoint not found at {ckpt_path}")
            payload = torch.load(ckpt_path, map_location=self.device)
            state = payload.get("model_state_dict", payload)
            self._model.load_state_dict(state, strict=False)
        self._last_snapshot: Optional[_Snapshot] = None

    # ------------------------------------------------------------------
    # BasePlanner API
    # ------------------------------------------------------------------
    def plan(
        self,
        observations: List[SnapshotNode],
        agent_states: List[AgentState],
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,
        global_nodes: Optional[List[SnapshotNode]] = None,
        serve_mark: Optional[List[int]] = None,
        unserved_count: Optional[int] = None,
    ) -> List[Deque[Target]]:
        if not observations:
            return [deque([depot]) for _ in agent_states]

        snapshot = self._build_snapshot(observations, depot)
        self._last_snapshot = snapshot

        routes = self._solve_snapshot(snapshot)
        if not routes:
            return [deque([depot]) for _ in agent_states]
        plans: List[Deque[Target]] = [deque() for _ in agent_states]
        for idx, route in enumerate(routes):
            if idx >= len(plans):
                break
            for node in route:
                plans[idx].append((node[0], node[1]))
            plans[idx].append(depot)
        for idx in range(len(plans)):
            if not plans[idx]:
                plans[idx].append(depot)
        return plans

    # ------------------------------------------------------------------
    # Snapshot helpers
    # ------------------------------------------------------------------
    def _build_snapshot(self, observations: Sequence[SnapshotNode], depot: Tuple[int, int]) -> _Snapshot:
        nodes = list(observations)
        if self.selection_policy == "earliest_due":
            nodes.sort(key=lambda n: n[4])
        elif self.selection_policy == "highest_demand":
            nodes.sort(key=lambda n: n[3], reverse=True)
        else:  # FIFO by arrival
            nodes.sort(key=lambda n: n[2])
        nodes = nodes[: self.max_nodes]
        valid = len(nodes)
        depot_xy = torch.tensor(
            [[[depot[0] / self.coord_normalizer, depot[1] / self.coord_normalizer]]],
            dtype=torch.float32,
            device=self.device,
        )
        node_xy = torch.zeros((1, self._env.problem_size, 2), dtype=torch.float32, device=self.device)
        node_demand = torch.zeros((1, self._env.problem_size), dtype=torch.float32, device=self.device)
        for idx, node in enumerate(nodes):
            if idx >= self._env.problem_size:
                break
            node_xy[0, idx, 0] = node[0] / self.coord_normalizer
            node_xy[0, idx, 1] = node[1] / self.coord_normalizer
            node_demand[0, idx] = float(node[3]) / float(max(self.capacity, 1))
        return _Snapshot(depot_xy=depot_xy, node_xy=node_xy, node_demand=node_demand, nodes_meta=list(nodes), valid_count=valid)

    def _solve_snapshot(self, snapshot: _Snapshot) -> List[List[SnapshotNode]]:
        self._env.load_from_tensors(snapshot.depot_xy, snapshot.node_xy, snapshot.node_demand, snapshot.valid_count)
        reset_state, _, _ = self._env.reset()
        self._model.pre_forward(reset_state)
        state, reward, done = self._env.pre_step()
        while not done:
            selected, _ = self._model(state)
            state, reward, done = self._env.step(selected)
        reward = reward.detach().cpu()
        best_idx = int(torch.argmax(reward[0]).item())
        seq = self._env.selected_node_list[0, best_idx].detach().cpu().tolist()
        return self._decode_routes(seq, snapshot)

    def _decode_routes(self, sequence: List[int], snapshot: _Snapshot) -> List[List[SnapshotNode]]:
        routes: List[List[SnapshotNode]] = []
        current: List[SnapshotNode] = []
        for idx in sequence:
            if idx == 0:
                if current:
                    routes.append(current)
                    current = []
                continue
            real_idx = idx - 1
            if 0 <= real_idx < snapshot.valid_count:
                current.append(snapshot.nodes_meta[real_idx])
        if current:
            routes.append(current)
        return routes