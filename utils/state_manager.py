from __future__ import annotations
from typing import List, Tuple, Deque, Dict, Any
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GlobalNodeList:
    """全局节点列表，包含所有需求节点信息
    
    节点格式: (x, y, t_arrival, t_due, demand)
    """
    nodes: List[Tuple[int, int, int, int, int]] = field(default_factory=list)  # [(x, y, t, end_t, c), ...]
    serve_mark: List[int] = field(default_factory=list)  # 服务标记，0未服务，1已服务
    
    def add_node(self, x: int, y: int, t: int, end_t: int, c: int) -> None:
        """添加新节点
        
        Args:
            x: x坐标
            y: y坐标
            t: 出现时间（t_arrival）
            end_t: 截止时间（t_due）
            c: 需求量（demand）
        """
        self.nodes.append((x, y, t, end_t, c))
        self.serve_mark.append(0)
    
    def mark_served(self, x: int, y: int) -> None:
        """标记节点为已服务"""
        for i, (nx, ny, _, _, _) in enumerate(self.nodes):
            if nx == x and ny == y and self.serve_mark[i] == 0:
                self.serve_mark[i] = 1
                break
    
    def get_unserved_count(self) -> int:
        """获取未服务节点数量"""
        return self.serve_mark.count(0)
    
    def reset(self) -> None:
        """重置全局节点列表"""
        self.nodes.clear()
        self.serve_mark.clear()


@dataclass
class PlanningState:
    """规划状态管理器"""
    current_plans: List[Deque[Tuple[int, int]]] = field(default_factory=list)  # 每个agent的当前规划路径
    global_nodes: GlobalNodeList = field(default_factory=GlobalNodeList)
    
    def reset(self, num_agents: int) -> None:
        """重置规划状态"""
        self.current_plans = [deque() for _ in range(num_agents)]
        self.global_nodes.reset()
    
    def update_plans(self, new_plans: List[Deque[Tuple[int, int]]]) -> None:
        """更新规划结果
        
        Note: We store the deque references directly (not copies) so that
        controller's popleft() operations persist across steps.
        """
        self.current_plans = list(new_plans)
    
    def get_unserved_count(self) -> int:
        """获取未服务节点数量"""
        return self.global_nodes.get_unserved_count()


def update_planning_state(
    planning_state: PlanningState,
    agent_states: List[Tuple[int, int, int]],  # [(x, y, s), ...]
    new_demands: List[Tuple[int, int, int, int, int]],  # 本时间步新增的需求 [(x, y, t, c, end_t), ...]
    obs_demands: List[Tuple[int, int, int, int, int]],  # 当前观测到的所有需求 [(x, y, t, c, end_t), ...]
    depot: Tuple[int, int] = None,  # depot 坐标，用于在清理时保留 depot 目标
) -> None:
    """
    更新规划状态：
    1. 从未来路径中删除已服务的节点
    2. 添加新出现的需求到全局节点列表
    3. 更新serve_mark
    
    Args:
        planning_state: 规划状态对象
        agent_states: 当前所有agent的状态
        new_demands: 本时间步新生成的需求，格式 [(x, y, t_arrival, c, t_due), ...]
        obs_demands: 当前环境中观测到的所有需求，格式 [(x, y, t_arrival, c, t_due), ...]
        depot: depot 坐标 (x, y)，用于在清理计划时保留 depot 目标
    """
    # 1. 添加新需求到全局节点列表
    for (x, y, t, c, end_t) in new_demands:
        planning_state.global_nodes.add_node(x, y, t, end_t, c)
    
    # 2. 检查并删除已服务的节点
    # 构建当前环境中存在的需求位置集合
    current_demand_positions = {(x, y) for (x, y, _, _, _) in obs_demands}
    
    # 检查全局节点列表中未服务的节点
    for i, ((nx, ny, _, _, _), mark) in enumerate(zip(planning_state.global_nodes.nodes, planning_state.global_nodes.serve_mark)):
        if mark == 0:  # 未服务的节点
            # 如果该位置不在当前需求中，说明已被服务
            if (nx, ny) not in current_demand_positions:
                planning_state.global_nodes.serve_mark[i] = 1
    
    # NOTE: We do NOT modify current_plans here anymore.
    # The controller is responsible for popping targets from plans when agent arrives.
    # Modifying plans here would interfere with the controller's stay-to-serve logic.
    # 
    # The cleaned_plan logic was removing targets that were already served,
    # but this also caused the plan deque to be replaced, breaking the reference
    # that the controller uses.