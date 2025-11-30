from __future__ import annotations

"""Canonical list of supported demand distributions for DVRP generators."""

SUPPORTED_DEMAND_DISTRIBUTIONS: tuple[str, ...] = (
    "uniform",
    "gaussian",
    "cluster",
    "explosion",
    "implosion",
)

__all__ = ["SUPPORTED_DEMAND_DISTRIBUTIONS"]
