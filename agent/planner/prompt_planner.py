"""
PromptPlanner - 使用prompt_model中的CVRPSolver实现静态VRP求解。

该Planner使用prompt_model.py中的CVRPSolver类来求解静态VRP问题，
替代v2_planner中使用的static_model.py。

关键设计：
1. 使用CVRPSolver作为后端求解器
2. 适配v2_planner的接口，保持相同的输入输出格式
3. 处理坐标和需求的归一化，匹配CVRPSolver的期望格式

注意：
- 该Planner仅支持静态模式（类似于v2_planner的static模式）
- 动态适配器功能未实现
"""

from __future__ import annotations
from typing import Deque, List, Tuple, Optional, Dict, Any
from collections import deque
import os
import math

from .base import BasePlanner, AgentState, Target

import torch

# 从prompt_model中导入必要的组件
try:
    from models_v2.prompt_model import CVRPSolver, get_CVRPSolver, augment_xy_data_by_8_fold
except ImportError:
    raise ImportError("无法导入CVRPSolver，请确保prompt_model.py在正确的位置")

# 导入标准化常量（使用与v2_planner相同的常量）
try:
    from configs import COORD_NORM, DEMAND_NORM
except ImportError:
    # 后备值
    COORD_NORM = 20.0
    DEMAND_NORM = 30.0


class PromptPlanner(BasePlanner):
    """
    使用prompt_model.CVRPSolver的Planner。
    
    架构：
    - 使用CVRPSolver作为后端求解器
    - 处理输入数据的预处理和后处理
    - 提供与v2_planner兼容的接口
    
    支持模式：
    - static: 使用CVRPSolver求解静态VRP问题
    
    标准化方案：
    - 坐标: 归一化到[0,1]通过除以COORD_NORM（网格大小）
    - 需求: 归一化通过DEMAND_NORM（=车辆容量）
    """
    
    def __init__(
        self,
        model_path: str = None,
        keys_path: str = None,
        problem_size: int = None,
        pomo_size: int = None,
        device: str = "cuda",
        grid_width: int = 20,
        grid_height: int = 20,
        full_capacity: int = 30,  # 固定车辆容量
        max_time: int = 100,
        use_augmentation: bool = True,
        **params,
    ) -> None:
        super().__init__(**params)
        self.model_path = model_path
        self.keys_path = keys_path
        self.problem_size = problem_size
        self.pomo_size = pomo_size
        self.device = device
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.full_capacity = full_capacity
        self.max_time = max_time
        self.use_augmentation = use_augmentation
        
        # 标准化常量（使用与v2_planner相同的常量）
        self.coord_norm = COORD_NORM
        self.demand_norm = DEMAND_NORM
        self.capacity_norm = float(full_capacity)
        self.time_norm = float(max_time)
        
        # 模型车辆容量（用于归一化）
        self._model_vehicle_capacity = float(full_capacity) / self.demand_norm
        
        # 延迟加载求解器
        self._solver = None
        self._loaded = False
        
        # 设置默认路径
        if self.model_path is None:
            self.model_path = "checkpoints/prompt_vrp/checkpoint-10000.pt"
        
        if self.keys_path is None:
            self.keys_path = "models_v2/keys_new_16"
    
    def _ensure_solver_loaded(self):
        """延迟加载CVRPSolver"""
        if self._loaded:
            return
        
        print(f"[PromptPlanner] 加载模型从 {self.model_path}")
        print(f"[PromptPlanner] 加载keys从 {self.keys_path}")
        
        # 创建CVRPSolver
        self._solver = get_CVRPSolver(
            model_path=self.model_path,
            keys_path=self.keys_path,
            problem_size=self.problem_size,
            pomo_size=self.pomo_size,
            use_cuda=(self.device == "cuda"),
            cuda_device_num=0 if self.device == "cuda" else None,
        )
        
        self._loaded = True
    
    def load_from_ckpt(self, ckpt_path: str) -> None:
        """加载检查点（用于兼容旧API）"""
        self.model_path = ckpt_path
        self._loaded = False  # 强制重新加载
    
    def plan(
        self,
        observations: List[Tuple[int, int, int, int, int]],  # [(x,y,t_arrival,demand,t_due), ...]
        agent_states: List[AgentState],  # x,y,s（负载）
        depot: Tuple[int, int],
        t: int,
        horizon: int = 1,
        current_plans: Optional[List[Deque[Target]]] = None,
        global_nodes: Optional[List[Tuple[int, int, int, int, int]]] = None,
        serve_mark: Optional[List[int]] = None,
        unserved_count: Optional[int] = None,
    ) -> List[Deque[Target]]:
        """
        返回每个agent的目标队列（deque[(x,y), ...]）
        
        Args:
            observations: 当前可见的需求列表 [(x, y, t_arrival, demand, t_due), ...]
            agent_states: agent状态列表 [AgentState(x, y, s), ...]
            depot: depot坐标 (x, y)
            t: 当前时间
            horizon: 规划时间窗口
            current_plans: 当前已有的规划路径（用于延续执行）
            
        Returns:
            每个agent的目标队列
        """
        self._ensure_solver_loaded()
        
        num_agents = len(agent_states)
        nodes = list(observations)
        N = len(nodes)
        
        # 静态模式：如果已有有效计划，继续执行
        if current_plans is not None:
            has_valid_plans = any(len(plan) > 0 for plan in current_plans)
            if has_valid_plans:
                # 返回相同的deque引用，以便控制器的popleft()持续生效
                return list(current_plans)
        
        # 如果没有节点，所有agent返回depot
        if N == 0:
            return [deque([depot] * max(1, horizon)) for _ in range(num_agents)]
        
        # 准备输入数据
        with torch.no_grad():
            # 转换坐标和需求为tensor
            # depot_xy: (1, 1, 2)
            depot_tensor = torch.tensor([[list(depot)]], dtype=torch.float32, device=self.device)
            
            # node_xy: (1, N, 2)
            node_coords = [[n[0], n[1]] for n in nodes]
            node_xy_tensor = torch.tensor([node_coords], dtype=torch.float32, device=self.device)
            
            # node_demand: (1, N)
            node_demands = [n[3] for n in nodes]  # 需求
            node_demand_tensor = torch.tensor([node_demands], dtype=torch.float32, device=self.device)
            
            # print(f"Depot形状{depot_tensor.shape=}, Node形状{node_xy_tensor.shape=}, Demand形状{node_demand_tensor.shape=}")
            # 解决VRP问题
            distances, routes = self._solver.solve_cvrp(
                depot_xy=depot_tensor,
                node_xy=node_xy_tensor,
                node_demand=node_demand_tensor,
                Up_Bound=self.coord_norm,  # 坐标上界
                Demand_scaler=self.demand_norm,  # 需求缩放器（车辆容量）
            )
            
            # routes的格式：List[List[List[int]]]，外层是batch，中层是循环，内层是节点索引
            # 对于单个batch，routes[0]是循环列表
            if len(routes) > 0:
                batch_routes = routes[0]  # 取第一个batch的结果
            else:
                batch_routes = []
            
            # 将循环分配给车辆
            vehicle_routes = self._distribute_cycles_to_vehicles(batch_routes, num_agents)
            
            # 转换为目标队列
            result = []
            for a in range(num_agents):
                targets = deque()
                if a < len(vehicle_routes):
                    for node_idx in vehicle_routes[a]:
                        if node_idx == 0:
                            targets.append(depot)
                        else:
                            # 节点索引从1开始（1对应第一个节点）
                            idx = node_idx - 1
                            if idx < len(nodes):
                                targets.append((nodes[idx][0], nodes[idx][1]))
                if not targets:
                    targets.append(depot)
                result.append(targets)
            
            return result
    
    def _distribute_cycles_to_vehicles(
        self,
        cycles: List[List[int]],
        num_vehicles: int,
    ) -> List[List[int]]:
        """
        将循环分配给多个车辆。
        
        CVRPSolver返回的循环是不包含depot的节点索引列表。
        我们需要将这些循环分配给多个车辆，并在每个循环前后添加depot。
        
        Args:
            cycles: 循环列表，每个循环是节点索引列表（从1开始）
            num_vehicles: 车辆数量
            
        Returns:
            每个车辆的路线列表，包含depot标记（0）
        """
        if not cycles:
            # 没有节点可访问，每辆车只停留在depot
            return [[0] for _ in range(num_vehicles)]
        
        # 初始化车辆路线
        vehicle_routes = [[] for _ in range(num_vehicles)]
        
        if len(cycles) <= num_vehicles:
            # 情况1：循环数少于或等于车辆数
            # 每个循环分配给一辆车
            for i, cycle in enumerate(cycles):
                # 添加循环节点，并在前后添加depot
                # 注意：车辆从depot出发，所以第一个循环前不需要depot
                vehicle_routes[i].extend(cycle)  # 添加循环节点
                vehicle_routes[i].append(0)  # 返回depot
            # 空闲车辆只停留在depot
            for i in range(len(cycles), num_vehicles):
                vehicle_routes[i].append(0)
        else:
            # 情况2：循环数多于车辆数
            # 将循环分配给车辆，保持循环完整
            # 使用平衡分配：分配给总节点数最少的车辆
            loads = [0] * num_vehicles
            
            for cycle in cycles:
                # 找到负载最少的车辆
                min_vehicle = min(range(num_vehicles), key=lambda v: loads[v])
                # 添加完整循环（节点+depot返回）
                vehicle_routes[min_vehicle].extend(cycle)
                vehicle_routes[min_vehicle].append(0)  # depot标记此循环结束
                loads[min_vehicle] += len(cycle)
        
        return vehicle_routes
    
    def _split_tour_to_routes(
        self,
        tour: List[int],
        num_vehicles: int,
    ) -> List[List[int]]:
        """
        将单个旅程拆分为多车辆路线（在depot返回处拆分）。
        
        由于所有agent从depot出发并返回depot，最优旅程由于容量限制也形成循环
        （depot -> 节点 -> depot），我们不应该破坏循环。
        
        1. 如果循环数 <= 车辆数：每个车辆分配一个循环
        2. 如果循环数 > 车辆数：一些车辆顺序执行多个循环
        
        Args:
            tour: 节点索引列表（0 = depot）
            num_vehicles: 车辆数量
            
        Returns:
            路线列表，每个车辆一个。每个路线是节点序列，以depot（0）结束。
            如果一辆车执行多个循环，它们被连接起来（node1, node2, 0, node3, node4, 0表示两个循环）。
        """
        # 提取循环（完整路线：depot -> 节点 -> depot）
        cycles = []
        current_cycle = []
        
        for node in tour:
            if node == 0:  # depot
                if current_cycle:
                    cycles.append(current_cycle)
                    current_cycle = []
            else:
                current_cycle.append(node)
        
        # 如果旅程不以depot结束，不要忘记最后一个循环
        if current_cycle:
            cycles.append(current_cycle)
        
        if not cycles:
            # 没有节点可访问，每辆车只停留在depot
            return [[0] for _ in range(num_vehicles)]
        
        # 初始化每辆车的路线
        routes = [[] for _ in range(num_vehicles)]
        
        if len(cycles) <= num_vehicles:
            # 情况1：循环数少于或等于车辆数
            # 每个循环分配给一辆车
            for i, cycle in enumerate(cycles):
                routes[i].extend(cycle)
                routes[i].append(0)  # 返回depot
            # 空闲车辆只停留在depot
            for i in range(len(cycles), num_vehicles):
                routes[i].append(0)
        else:
            # 情况2：循环数多于车辆数
            # 将循环分配给车辆，保持循环完整
            # 使用平衡分配：分配给总节点数最少的车辆
            loads = [0] * num_vehicles
            
            for cycle in cycles:
                # 找到负载最少的车辆
                min_vehicle = min(range(num_vehicles), key=lambda v: loads[v])
                # 添加完整循环（节点+depot返回）
                routes[min_vehicle].extend(cycle)
                routes[min_vehicle].append(0)  # depot标记此循环结束
                loads[min_vehicle] += len(cycle)
        
        return routes


def create_prompt_planner(
    model_path: Optional[str] = None,
    keys_path: Optional[str] = None,
    problem_size: Optional[int] = None,
    pomo_size: Optional[int] = None,
    device: str = "cuda",
    **kwargs,
) -> PromptPlanner:
    """工厂函数创建PromptPlanner，使用默认路径。"""
    # 设置默认路径
    if model_path is None:
        default_model = "checkpoints/prompt_vrp/checkpoint-10000.pt"
        if os.path.exists(default_model):
            model_path = default_model
        else:
            # 尝试其他可能的位置
            alt_paths = [
                "prompt_vrp/checkpoint-10000.pt",
                "checkpoint-10000.pt",
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    model_path = path
                    break
    
    if keys_path is None:
        default_keys = "keys_new_16"
        if os.path.exists(default_keys):
            keys_path = default_keys
        else:
            # 尝试其他可能的位置
            alt_paths = [
                "prompt_model/keys_new_16",
                "data/keys_new_16",
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    keys_path = path
                    break
    
    if model_path is None or keys_path is None:
        print(f"[警告] 未找到模型或keys文件，将使用默认路径")
        print(f"模型路径: {model_path}")
        print(f"keys路径: {keys_path}")
    
    return PromptPlanner(
        model_path=model_path,
        keys_path=keys_path,
        problem_size=problem_size,
        pomo_size=pomo_size,
        device=device,
        **kwargs,
    )