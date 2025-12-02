from __future__ import annotations

from typing import Dict, Tuple


RewardTerms = Dict[str, float]


def compute_static_reward(
    service_bonus: float,
    travel_cost: float,
    return_bonus: float,
) -> Tuple[float, RewardTerms]:
    """Reward recipe for static VRP episodes.

    Static demands should only optimize the negative travel distance so that
    the solver focuses purely on minimizing route length. Service and depot
    return bonuses remain for logging but no longer affect the reward.
    """

    reward = travel_cost
    terms: RewardTerms = {
        "travel_cost": travel_cost,
    }
    return reward, terms


def compute_dynamic_reward(
    service_bonus: float,
    travel_cost: float,
    waiting_penalty: float,
    return_bonus: float,
    switch_penalty: float,
    approach_bonus: float,
    exploration_penalty: float,
    crowding_penalty: float,
) -> Tuple[float, RewardTerms]:
    """Reward recipe for dynamic VRP episodes.

    Dynamic runs benefit from additional shaping: penalize waiting demands,
    discourage frequent target switches, reward proactive moves toward
    outstanding requests, and nudge agents to spread out (via crowding and
    exploration penalties).
    """

    reward = (
        service_bonus
        + travel_cost
        + waiting_penalty
        + return_bonus
        + switch_penalty
        + approach_bonus
        + exploration_penalty
        + crowding_penalty
    )
    terms: RewardTerms = {
        "service_bonus": service_bonus,
        "travel_cost": travel_cost,
        "waiting_penalty": waiting_penalty,
        "depot_return_bonus": return_bonus,
        "switch_penalty": switch_penalty,
        "approach_bonus": approach_bonus,
        "exploration_penalty": exploration_penalty,
        "crowding_penalty": crowding_penalty,
    }
    return reward, terms
