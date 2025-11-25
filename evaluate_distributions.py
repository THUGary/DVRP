import copy
import numpy as np

from run_evaluate import run_episode_return_metrics
from configs import get_default_config, Config

# 支持的分布名称
DISTRIBUTIONS = ["uniform", "gaussian", "cluster", "explosion", "implosion"]


# ================================
# ★ 修复参数防止 NaN 概率
# ================================
def sanitize_cfg(cfg: Config):
    """
    清洗配置，防止 rule_generator 出现 NaN 概率。
    不修改原 cfg，只修复明显导致 /0 或 NaN 的参数。
    """

    g = cfg.generator_params

    # -----------------------------
    # 1. scale_factor 防止为 0
    # -----------------------------
    if "scale_factor" in g:
        if g["scale_factor"] is None or g["scale_factor"] <= 0:
            g["scale_factor"] = 1.0  # 给一个最安全下限
    else:
        g["scale_factor"] = 1.0

    # -----------------------------
    # 2. neighborhood_radius 不能 0
    # -----------------------------
    if "neighborhood_radius" in g:
        if g["neighborhood_radius"] is None or g["neighborhood_radius"] <= 0:
            g["neighborhood_radius"] = 3
    else:
        g["neighborhood_radius"] = 3

    # -----------------------------
    # 3. num_centers 必须 ≥1
    # -----------------------------
    if "num_centers" in g:
        if g["num_centers"] is None or g["num_centers"] < 1:
            g["num_centers"] = 3
    else:
        g["num_centers"] = 3

    # -----------------------------
    # 4. 分布名称合法化
    # -----------------------------
    if g.get("distribution") not in DISTRIBUTIONS:
        g["distribution"] = "uniform"

    return cfg


# ===================================
# ★ 主评估函数
# ===================================
def evaluate_distributions(cfg: Config, num_runs=10):

    results = {}

    for dist in DISTRIBUTIONS:
        print(f"\n=== Evaluating distribution: {dist} ===")

        metrics_list = []

        for seed in range(num_runs):

            # ★ 深拷贝 Config
            local_cfg = copy.deepcopy(cfg)

            # ★ 设置分布
            local_cfg.generator_params["distribution"] = dist

            # ★ 重要：清洗配置
            local_cfg = sanitize_cfg(local_cfg)

            local_cfg.seed = seed

            # ★ 运行一次 episode
            episode_metrics = run_episode_return_metrics(
                local_cfg,
                seed=seed,
                render=False,
                fps=0,
                planner="greedy",
                static_demands=True
            )

            metrics_list.append(episode_metrics)

        # ------------------------------
        # 统计标量指标
        # ------------------------------
        numeric_keys = [
            k for k in metrics_list[0].keys()
            if isinstance(metrics_list[0][k], (int, float, np.integer, np.floating))
        ]

        dist_mean = {k: float(np.mean([m[k] for m in metrics_list])) for k in numeric_keys}
        dist_std = {k: float(np.std([m[k] for m in metrics_list])) for k in numeric_keys}

        results[dist] = {"mean": dist_mean, "std": dist_std}

    return results


# ===================================
# ★ 入口
# ===================================
if __name__ == "__main__":
    cfg = get_default_config()

    result = evaluate_distributions(cfg, num_runs=10)

    print("\n==== Final Distribution Evaluation Results ====")
    print(result)
