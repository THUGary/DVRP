from __future__ import annotations
import argparse
import json
import os
import time
from typing import Tuple, List, Dict, Any

# Ensure project root on sys.path when running from nested training directory
import sys
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.planner_model.model import DVRPNet  # 需与现有模型实现一致
from agent.controller.distance import travel_time

MODEL_CONFIG_DEFAULTS = {
    "d_model": 128,
    "nhead": 8,
    "nlayers": 2,
    "adapter_dim": 64,
    "lateness_lambda": 0.0,
}
MODEL_CONFIG_DEFAULT_PATH = os.path.join(str(_ROOT), "training", "planner", "model_config.json")


def _load_model_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(resolved):
        print(f"[MODEL-CONFIG] file not found at {resolved}, using defaults.")
        return {}
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("model config must be a JSON object")
        return data
    except Exception as exc:
        print(f"[MODEL-CONFIG] failed to load {resolved}: {exc}, using defaults.")
        return {}


def _resolve_model_param(arg_value: Any, config: Dict[str, Any], key: str, fallback: Any) -> Any:
    if arg_value is not None:
        return arg_value
    if key in config:
        return config[key]
    return fallback


class PlanRowsDataset(Dataset):
    """读取 data_gen.py 生成的 rows 格式数据"""
    def __init__(self, path: str):
        super().__init__()
        blob = torch.load(path, map_location="cpu")
        self.rows: List[Dict[str, Any]] = blob["rows"]
        self.meta: Dict[str, Any] = blob.get("meta", {})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def collate_rows(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """将变长 nodes 的 row 批量化，支持多 agent 和 (A,K) 标签。
    期望 item 包含：
      - nodes: List[(x,y,t_arrival,c,t_due)]，长度 N
      - node_mask: List[bool]，长度 N（可选，默认全 False）
      - agents: List[(x,y,s,t_agent)]，长度 A
      - depot: (dx,dy,t)
      - labels_ak: List[List[int]]，形状 [A,K]，值域 [0..N]（0 表示 depot，1..N 表示 nodes[0..N-1]）
            - history_positions/history_targets: List[List[(x,y)]]，按 agent 存储的历史轨迹
    """
    B = len(batch)
    maxN = max((len(item["nodes"]) for item in batch), default=0)
    maxA = max((len(item.get("agents", [item.get("agent")]) or []) for item in batch), default=1)

    def _max_seq(it: Dict[str, Any]) -> int:
        seqs = it.get("labels_ak") or []
        length = 0
        for seq in seqs:
            length = max(length, len(seq))
        legacy = it.get("labels_k")
        if legacy is not None:
            length = max(length, len(legacy))
        return length
    maxK = max((_max_seq(item) for item in batch), default=0)
    maxK = max(1, maxK)
    def _max_hist_len(key: str) -> int:
        length = 0
        for item in batch:
            seqs = item.get(key) or []
            for seq in seqs:
                if seq:
                    length = max(length, len(seq))
        return length

    max_hist = max(1, _max_hist_len("history_positions"), _max_hist_len("history_targets"))

    nodes = torch.zeros(B, maxN, 5, dtype=torch.float32)
    node_mask = torch.ones(B, maxN, dtype=torch.bool)
    agents = torch.zeros(B, maxA, 4, dtype=torch.float32)
    depot = torch.zeros(B, 1, 2, dtype=torch.float32)
    labels_ak = torch.zeros((B, maxA, maxK), dtype=torch.long)
    label_mask = torch.zeros((B, maxA, maxK), dtype=torch.bool)
    valid_N = torch.zeros(B, dtype=torch.long)
    cap_full = torch.zeros(B, maxA, dtype=torch.float32)
    history_pos = torch.full((B, maxA, max_hist, 2), fill_value=-1.0, dtype=torch.float32)
    history_tgt = torch.full((B, maxA, max_hist, 2), fill_value=-1.0, dtype=torch.float32)
    time_now = torch.zeros(B, dtype=torch.float32)

    for b, item in enumerate(batch):
        Ni = len(item["nodes"])
        valid_N[b] = Ni
        if Ni > 0:
            nodes[b, :Ni] = torch.tensor(item["nodes"], dtype=torch.float32)
            mask_i = torch.tensor(item.get("node_mask", [False] * Ni), dtype=torch.bool)
            node_mask[b, :Ni] = mask_i

        # agents
        agents_entry: List[Tuple[float, float, float, float]] = []
        if "agents" in item and item["agents"] is not None:
            if isinstance(item["agents"], (list, tuple)):
                agents_entry = list(item["agents"])
            else:
                agents_entry = [item["agents"]]
        elif "agent" in item and item["agent"] is not None:
            agents_entry = [item["agent"]]

        for a in range(min(maxA, len(agents_entry))):
            ax, ay, s, ta = agents_entry[a]
            agents[b, a] = torch.tensor([ax, ay, s, ta], dtype=torch.float32)

        depot_raw = item["depot"]
        if isinstance(depot_raw, (list, tuple)):
            if len(depot_raw) < 2:
                raise RuntimeError("Depot entry must provide at least (x, y)")
            dx, dy = depot_raw[0], depot_raw[1]
        else:
            dx = dy = float(depot_raw)
        depot[b, 0, 0] = float(dx)
        depot[b, 0, 1] = float(dy)
        depot_xy = (int(dx), int(dy))
        time_now[b] = float(item.get("time_now", depot_raw[2] if isinstance(depot_raw, (list, tuple)) and len(depot_raw) > 2 else 0.0))

        # labels
        seqs = item.get("labels_ak") or []
        for a in range(min(maxA, len(seqs))):
            seq = seqs[a][:maxK]
            if not seq:
                continue
            la = torch.tensor(seq, dtype=torch.long)
            labels_ak[b, a, : la.numel()] = la
            label_mask[b, a, : la.numel()] = True
        if not seqs and "labels_k" in item:
            seq = item["labels_k"][:maxK]
            if seq:
                la = torch.tensor(seq, dtype=torch.long)
                labels_ak[b, 0, : la.numel()] = la
                label_mask[b, 0, : la.numel()] = True

        # cap_full per agent must come from row's full_capacity (来自 Config.capacity)
        if "full_capacity" not in item:
            raise RuntimeError("Row is missing 'full_capacity' (Config.capacity). Regenerate dataset with full_capacity set to Config.capacity.")
        full_c = float(item["full_capacity"])
        if full_c <= 0:
            raise RuntimeError("Row 'full_capacity' must be > 0 (Config.capacity). Found: {}".format(full_c))
        cap_full[b, :].fill_(full_c)

        hist_pos_entry = item.get("history_positions") or []
        hist_tgt_entry = item.get("history_targets") or []
        for a in range(maxA):
            if a < len(hist_pos_entry) and hist_pos_entry[a]:
                seq_pos = hist_pos_entry[a][-max_hist:]
            elif a < len(agents_entry):
                seq_pos = [(int(agents_entry[a][0]), int(agents_entry[a][1]))]
            else:
                seq_pos = [depot_xy]
            for t_idx, (hx, hy) in enumerate(seq_pos):
                if t_idx >= max_hist:
                    break
                history_pos[b, a, t_idx, 0] = float(hx)
                history_pos[b, a, t_idx, 1] = float(hy)

            if a < len(hist_tgt_entry) and hist_tgt_entry[a]:
                seq_tgt = hist_tgt_entry[a][-max_hist:]
            else:
                seq_tgt = [depot_xy]
            for t_idx, (tx, ty) in enumerate(seq_tgt):
                if t_idx >= max_hist:
                    break
                history_tgt[b, a, t_idx, 0] = float(tx)
                history_tgt[b, a, t_idx, 1] = float(ty)

    return {
        "nodes": nodes,
        "node_mask": node_mask,
        "agents": agents,        # [B,A,4]
        "depot": depot,          # [B,1,3]
        "labels_ak": labels_ak,  # [B,A,K]
        "label_mask": label_mask,
        "valid_N": valid_N,
        "cap_full": cap_full,    # [B,A]
        "history_positions": history_pos,
        "history_targets": history_tgt,
        "time_now": time_now,
    }


# def _tensor_stats(name: str, t: torch.Tensor) -> str:
#     try:
#         if t.numel() == 0:
#             return f"{name}: empty"
#         return f"{name}: shape={tuple(t.shape)} min={float(t.min()):.4e} max={float(t.max()):.4e} mean={float(t.mean()):.4e}"
#     except Exception:
#         return f"{name}: cannot compute stats"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DVRPNet on rows generated by data_gen.py")
    # data
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--map_wid", type=int, default=20)
    p.add_argument("--agent_num", type=int, default=2)
    p.add_argument("--prefix", type=str, default="plans")
    # model
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--nlayers", type=int, default=None)
    p.add_argument("--lateness_lambda", type=float, default=0.0)
    p.add_argument("--coord_norm", type=float, default=None, help="坐标归一化尺度 (默认=map_wid)")
    p.add_argument("--capacity", type=float, default=200.0, help="数据集中车辆容量")
    p.add_argument("--capacity_norm", type=float, default=None, help="容量归一化尺度 (默认=capacity)")
    p.add_argument("--max_time", type=float, default=100.0, help="环境最大时间，用于时间归一化")
    p.add_argument("--time_norm", type=float, default=None, help="时间归一化尺度 (默认=max_time)")
    p.add_argument("--stage", type=str, choices=["static", "dynamic"], default="static", help="静态预训练或动态适配阶段")
    p.add_argument("--adapter_dim", type=int, default=None, help="动态阶段的时间窗适配器维度 (stage=dynamic 有效)")
    p.add_argument("--model_config", type=str, default=MODEL_CONFIG_DEFAULT_PATH, help="JSON file that overrides model structure defaults")
    p.add_argument("--static_ckpt", type=str, default=None, help="动态阶段初始化所用的静态模型 checkpoint")
    p.add_argument("--freeze_base", action="store_true", help="动态阶段仅训练 adapter 参数")
    # train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints/planner")
    p.add_argument("--debug", action="store_true", help="Enable debug logging for NaN/Inf diagnostics")
    # early stopping / checkpointing
    p.add_argument("--early_stop", action="store_true", help="Enable early stopping on validation loss")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs")
    p.add_argument("--min_delta", type=float, default=1e-5, help="Minimum relative improvement on val loss to reset patience")
    p.add_argument("--save_best", action="store_true", help="Save best model (by val loss) to ckpt dir as planner_best_...pt")
    return p


def save_ckpt(model: DVRPNet, ckpt_dir: str, map_wid: int, agent_num: int, epoch: int | str, stage: str) -> str:
    stage_tag = stage.lower()
    stage_folder = "static_planner" if stage_tag == "static" else "dynamic_planner"
    target_dir = os.path.join(ckpt_dir, stage_folder)
    os.makedirs(target_dir, exist_ok=True)
    name = f"planner_{stage_tag}_{map_wid}_{agent_num}_{epoch}.pt"
    path = os.path.join(target_dir, name)
    torch.save({"model": model.state_dict()}, path)
    print(f"[CKPT] saved => {path}")
    return path


@torch.no_grad()
def evaluate(model: DVRPNet, loader: DataLoader, device: torch.device, lateness_lambda: float, amp: bool) -> Tuple[float, float]:
    model.eval()
    total_loss, total_cnt, total_corr = 0.0, 0, 0
    for batch in loader:
        nodes = batch["nodes"].to(device)
        node_mask = batch["node_mask"].to(device)
        agents = batch["agents"].to(device)
        depot = batch["depot"].to(device)
        depot_xy = depot[..., :2]
        labels_ak = batch["labels_ak"].to(device)  # [B,A,K]
        label_mask = batch["label_mask"].to(device)
        history_pos = batch["history_positions"].to(device)
        history_tgt = batch["history_targets"].to(device)
        time_now = batch["time_now"].to(device)

        B = nodes.size(0)
        A = agents.size(1)
        K = labels_ak.size(2)

        # 编码一次
        feats = {"nodes": nodes, "node_mask": node_mask, "depot": depot_xy, "time_now": time_now}
        enc = model.encode(feats)
        cur_mask = enc["node_mask"].clone()  # [B,N]
        ag = agents.clone()
        # cap_full must be provided by dataset (from Config.capacity). No fallback allowed.
        if "cap_full" not in batch:
            raise RuntimeError("Batch missing 'cap_full' — dataset must include full_capacity (Config.capacity). Regenerate data if needed.")
        cap_full = batch["cap_full"].to(device)

        for step in range(K):
            mask_step = label_mask[:, :, step]
            if not torch.any(mask_step):
                continue
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model.decode(
                    enc_nodes=enc["H_nodes"],
                    enc_depot=enc["H_depot"],
                    node_mask=cur_mask,
                    agents_tensor=ag,
                    nodes=nodes,
                    lateness_lambda=lateness_lambda,
                    history_positions=history_pos,
                    history_target_coords=history_tgt,
                    time_now=time_now,
                )  # [B,A,N+1]

                # debug
                # print(("eval batch nodes", nodes.detach().cpu().tolist()))
                # print(("eval batch depot", depot.detach().cpu().tolist()))
                # print(("eval batch agents", agents.detach().cpu().tolist()))
                # print(("eval batch labels_ak", labels_ak.detach().cpu().tolist()))

                # logits: [B,A,N+1], labels step: [B,A]
                labels_step = labels_ak[:, :, step]
                logits_flat = logits.reshape(-1, logits.size(-1))      # [B*A, N+1]
                targets_flat = labels_step.reshape(-1)                 # [B*A]
                mask_flat = mask_step.reshape(-1)
                if not torch.any(mask_flat):
                    continue
                loss = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
                # debug
                # pred = torch.argmax(logits, dim=-1)  # [B,A]
                # print(f"[EVAL-DEBUG] Step {step} targets={targets_flat.detach().cpu().tolist()} preds={pred.detach().cpu().tolist()} loss={loss.item():.4f}")

            # debug: concise output when loss is non-finite
            if not torch.isfinite(loss):
                # print persample losses
                for b in range(B):
                    for a in range(A):
                        idx = b * A + a
                        logit_ba = logits_flat[idx: idx + 1, :]  # [1,N+1]
                        target_ba = targets_flat[idx: idx + 1]   # [1]
                        loss_ba = F.cross_entropy(logit_ba, target_ba, reduction="none")  # [1]
                        print(f"[EVAL-DEBUG] Sample (b={b} a={a}) logits={logit_ba.detach().cpu().tolist()} target={target_ba.detach().cpu().tolist()} loss={loss_ba.item():.4f}")

                pred = torch.argmax(logits, dim=-1)
                print(f"[EVAL-DEBUG] Non-finite loss at val step {step} | preds={pred.detach().cpu().tolist()} | labels={labels_step.detach().cpu().tolist()}")
                print(f"logits: {logits.detach().cpu().tolist()}")
                # skip this batch
                continue

            # 额外调试：检查标签是否违反容量（demand > space）
            try:
                Bsz = nodes.size(0)
                A = agents.size(1)
                Nn = nodes.size(1)
                violations = []
                for b in range(Bsz):
                    for a in range(A):
                        idx = int(labels_step[b, a].item())
                        if 1 <= idx <= Nn:
                            demand = float(nodes[b, idx - 1, 3].item())
                            space = float(ag[b, a, 2].item())
                            if demand > space + 1e-6:
                                violations.append((b, a, step, idx, demand, space))
                if violations:
                    print(f"[EVAL-DEBUG][cap-violation] found {len(violations)} label(s) requiring demand > space at step={step}: {violations[:5]}{' ...' if len(violations)>5 else ''}")
            except Exception as _:
                pass

            # 统计
            pred = torch.argmax(logits, dim=-1)  # [B,A]
            total_loss += loss.item() * int(mask_flat.sum().item())
            total_cnt += int(mask_flat.sum().item())
            total_corr += ((pred == labels_step) & mask_step).sum().item()

            # Teacher-forcing 使用真值标签更新状态
            sel = labels_step  # [B,A] (0=depot, 1..N=node)
            Nn = nodes.size(1)
            # 更新 mask（本步被任一 agent 选中的节点标 True）；注意 nodes 映射 idx-1
            oh = torch.zeros(B, Nn, dtype=torch.bool, device=device)
            for b in range(B):
                for a in range(A):
                    if not mask_step[b, a]:
                        continue
                    idx = int(sel[b, a].item())
                    if 1 <= idx <= Nn:
                        oh[b, idx - 1] = True
            cur_mask = cur_mask | oh
            # 更新各 agent 位置/时间/容量（depot 则恢复满容量）
            for b in range(B):
                for a in range(A):
                    if not mask_step[b, a]:
                        continue
                    idx = int(sel[b, a].item())
                    if 1 <= idx <= Nn:
                        dest_xy = nodes[b, idx - 1, :2].long()
                    else:
                        dest_xy = depot[b, 0, :2].long()
                    cur_xy = ag[b, a, :2].long()
                    dt = (cur_xy[0] - dest_xy[0]).abs() + (cur_xy[1] - dest_xy[1]).abs()
                    ag[b, a, :2] = dest_xy.to(ag.dtype)
                    ag[b, a, 3] = ag[b, a, 3] + dt.to(ag.dtype)
                    if 1 <= idx <= Nn:
                        d = nodes[b, idx - 1, 3].item()
                        ag[b, a, 2] = torch.clamp(ag[b, a, 2] - d, min=0.0)
                    else:
                        # depot: restore to full capacity captured at planning start
                        ag[b, a, 2] = cap_full[b, a]

    return total_loss / max(1, total_cnt), total_corr / max(1, total_cnt)

def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")
    coord_norm = float(args.coord_norm) if args.coord_norm is not None else float(args.map_wid)
    capacity_norm = float(args.capacity_norm) if args.capacity_norm is not None else float(args.capacity)
    time_norm = float(args.time_norm) if args.time_norm is not None else float(args.max_time)

    model_config = _load_model_config(args.model_config)
    d_model = int(_resolve_model_param(args.d_model, model_config, "d_model", MODEL_CONFIG_DEFAULTS["d_model"]))
    nhead = int(_resolve_model_param(args.nhead, model_config, "nhead", MODEL_CONFIG_DEFAULTS["nhead"]))
    nlayers = int(_resolve_model_param(args.nlayers, model_config, "nlayers", MODEL_CONFIG_DEFAULTS["nlayers"]))
    adapter_dim_cfg = int(_resolve_model_param(args.adapter_dim, model_config, "adapter_dim", MODEL_CONFIG_DEFAULTS["adapter_dim"]))
    lateness_lambda = float(_resolve_model_param(args.lateness_lambda, model_config, "lateness_lambda", MODEL_CONFIG_DEFAULTS["lateness_lambda"]))

    stage_dir = "static_rows" if args.stage == "static" else "dynamicrows"
    data_root = args.data_dir
    normalized_root = os.path.abspath(os.path.expanduser(data_root))
    if os.path.basename(os.path.normpath(normalized_root)) == stage_dir:
        stage_data_dir = normalized_root
    else:
        stage_data_dir = os.path.join(normalized_root, stage_dir)
    train_path = os.path.join(stage_data_dir, f"{args.prefix}_train_{args.map_wid}_{args.agent_num}.pt")
    val_path = os.path.join(stage_data_dir, f"{args.prefix}_val_{args.map_wid}_{args.agent_num}.pt")
    # debug
    # print(f"Train data path: {train_path}")
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        raise FileNotFoundError(f"Missing data files: {train_path} or {val_path}. Run data_gen.py first.")

    # Dataset / Loader
    trn_ds = PlanRowsDataset(train_path)
    val_ds = PlanRowsDataset(val_path)
    train_loader = DataLoader(trn_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_rows)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_rows)

    # 模型/优化
    adapter_dim = adapter_dim_cfg if args.stage == "dynamic" else 0
    model = DVRPNet(
        d_model=d_model,
        nhead=nhead,
        nlayers=nlayers,
        coord_norm=coord_norm,
        capacity_norm=capacity_norm,
        time_norm=time_norm,
        adapter_dim=adapter_dim,
    ).to(device)
    if args.stage == "dynamic":
        if not args.static_ckpt:
            raise ValueError("Dynamic stage requires --static_ckpt pointing to a trained static model checkpoint.")
        ckpt_path = os.path.abspath(os.path.expanduser(args.static_ckpt))
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Static checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        state = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INIT] loaded static checkpoint from {ckpt_path}; missing={missing}, unexpected={unexpected}")
        if args.freeze_base:
            for name, param in model.named_parameters():
                if "time_adapter" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    last_ckpt = None
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = None
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            epoch_loss = 0.0
            epoch_cnt = 0
            for batch in train_loader:
                nodes = batch["nodes"].to(device)
                node_mask = batch["node_mask"].to(device)
                agents = batch["agents"].to(device)
                depot = batch["depot"].to(device)
                depot_xy = depot[..., :2]
                labels_ak = batch["labels_ak"].to(device)  # [B,A,K]
                label_mask = batch["label_mask"].to(device)
                valid_N = batch["valid_N"].to(device)
                history_pos = batch["history_positions"].to(device)
                history_tgt = batch["history_targets"].to(device)
                time_now = batch["time_now"].to(device)
                # 可选：打印批次统计
                # print(_tensor_stats("Train batch nodes", nodes))
                # print(_tensor_stats("Train batch depot", depot))
                # print(_tensor_stats("Train batch agents", agents))
                # print(_tensor_stats("Train batch labels_ak", labels_ak))

                # 编码一次
                feats = {"nodes": nodes, "node_mask": node_mask, "depot": depot_xy, "time_now": time_now}
                enc = model.encode(feats)
                cur_mask = enc["node_mask"].clone()
                ag = agents.clone()
                # cap_full must be present in batch; no fallback to agents' s
                if "cap_full" not in batch:
                    raise RuntimeError("Batch missing 'cap_full' — dataset must include full_capacity (Config.capacity). Regenerate data if needed.")
                cap_full = batch["cap_full"].to(device)  # [B,A]

                Bsz = nodes.size(0)
                A = agents.size(1)
                K = labels_ak.size(2)
                loss_sum = 0.0
                if args.debug:
                    print(f"[TRAIN-DEBUG] Epoch {epoch:03d} | New batch B={Bsz} A={A} K={K} | valid_N={valid_N.detach().cpu().tolist()}")
                    # print(f"[TRAIN-DEBUG] mask (B x N): {cur_mask.detach().cpu().tolist()}")
                for step in range(K):
                    mask_step = label_mask[:, :, step]
                    if not torch.any(mask_step):
                        continue
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        logits = model.decode(
                            enc_nodes=enc["H_nodes"],
                            enc_depot=enc["H_depot"],
                            node_mask=cur_mask,
                            agents_tensor=ag,
                            nodes=nodes,
                            lateness_lambda=lateness_lambda,
                            history_positions=history_pos,
                            history_target_coords=history_tgt,
                            time_now=time_now,
                        )  # [B,A,N+1]
                        labels_step = labels_ak[:, :, step]                  # [B,A]
                        logits_flat = logits.reshape(-1, logits.size(-1))    # [B*A,N+1]
                        targets_flat = labels_step.reshape(-1)               # [B*A]
                        mask_flat = mask_step.reshape(-1)
                        if not torch.any(mask_flat):
                            continue
                        loss_step = F.cross_entropy(logits_flat[mask_flat], targets_flat[mask_flat])
                        # if args.debug:
                            # print(f"[TRAIN-DEBUG] mask (B x N): {cur_mask.detach().cpu().tolist()}")
                            # print(f"[TRAIN-DEBUG] logits: {logits_flat}, targets: {targets_flat}")

                        # 调试：检查标签容量可行性（不影响训练，只打印）
                        if args.debug:
                            try:
                                Bsz = nodes.size(0)
                                A = agents.size(1)
                                Nn = nodes.size(1)
                                violations = []
                                for b in range(Bsz):
                                    for a in range(A):
                                        idx = int(labels_step[b, a].item())
                                        if 1 <= idx <= Nn:
                                            demand = float(nodes[b, idx - 1, 3].item())
                                            space = float(ag[b, a, 2].item())
                                            if demand > space + 1e-6:
                                                violations.append((b, a, step, idx, demand, space))
                                if violations:
                                    print(f"[TRAIN-DEBUG][cap-violation] found {len(violations)} label(s) requiring demand > space at step={step}: {violations[:10]}{' ...' if len(violations)>10 else ''}")
                            except Exception as _:
                                pass

                        # debug: if non-finite loss, print concise diagnostics
                        if not torch.isfinite(loss_step) and args.debug:
                            print(f"[TRAIN-DEBUG] logits: {logits_flat}, targets: {targets_flat}")
                            losses_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [B*A]
                            losses_ba = losses_flat.view(Bsz, A)  # [B,A]
                            print(f"[TRAIN-DEBUG] per-sample losses (B x A): {losses_ba.detach().cpu().tolist()}")
                            # print(f"[TRAIN-DEBUG] per-batch mean losses: {per_batch_mean.detach().cpu().tolist()}")
                            pred = torch.argmax(logits.detach(), dim=-1)  # [B,A]
                            print(f"[TRAIN-DEBUG] Non-finite loss at epoch {epoch} step {step} | preds={pred.detach().cpu().tolist()} | labels={labels_step.detach().cpu().tolist()}")
                            # raise to stop training for inspection
                            raise RuntimeError("Non-finite training loss encountered; check debug output.")
                        # if args.debug:
                        #     losses_flat = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [B*A]
                        #     losses_ba = losses_flat.view(Bsz, A)  # [B,A]
                        #     print(f"[TRAIN-DEBUG] per-sample losses (B x A): {losses_ba.detach().cpu().tolist()}")
                                
                            pred = torch.argmax(logits, dim=-1)  # [B,A]
                            print(f"[Epoch {epoch:03d} Step {step:02d}] targets={targets_flat.detach().cpu().tolist()} preds={pred.detach().cpu().tolist()} loss={loss_step.item():.4f}")
                        # 反传该步损失
                        opt.zero_grad(set_to_none=True)
                        scaler.scale(loss_step).backward()
                        scaler.step(opt)
                        scaler.update()

                    # 累计统计
                    valid_cnt = int(mask_flat.sum().item())
                    epoch_loss += loss_step.item() * valid_cnt
                    epoch_cnt += valid_cnt

                    # Teacher-forcing: 使用真值标签更新 mask/agents
                    sel = labels_step  # [B,A] (0=depot, 1..N=node)
                    Nn = nodes.size(1)
                    # 更新 mask
                    oh = torch.zeros(Bsz, Nn, dtype=torch.bool, device=device)
                    for b in range(Bsz):
                        for a in range(A):
                            if not mask_step[b, a]:
                                continue
                            idx = int(sel[b, a].item())
                            if 1 <= idx <= Nn:
                                oh[b, idx - 1] = True
                    cur_mask = cur_mask | oh
                    # 更新 agents 状态（depot 则恢复满容量）
                    for b in range(Bsz):
                        for a in range(A):
                            if not mask_step[b, a]:
                                continue
                            idx = int(sel[b, a].item())
                            if 1 <= idx <= Nn:
                                dest_xy = nodes[b, idx - 1, :2].long()
                            else:
                                dest_xy = depot[b, 0, :2].long()
                            cur_xy = ag[b, a, :2].long()
                            dt = (cur_xy[0] - dest_xy[0]).abs() + (cur_xy[1] - dest_xy[1]).abs()
                            ag[b, a, :2] = dest_xy.to(ag.dtype)
                            ag[b, a, 3] = ag[b, a, 3] + dt.to(ag.dtype)
                            if 1 <= idx <= Nn:
                                d = nodes[b, idx - 1, 3].item()
                                ag[b, a, 2] = torch.clamp(ag[b, a, 2] - d, min=0.0)
                            else:
                                ag[b, a, 2] = cap_full[b, a]

            # 评价与保存
            avg_train_loss = epoch_loss / max(1, epoch_cnt)
            val_loss, val_acc = evaluate(model, val_loader, device, lateness_lambda, args.amp)
            dt = time.time() - t0
            print(f"[Epoch {epoch:03d}] train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc@step={val_acc:.4f} | {dt:.1f}s")
            # Best-model tracking & early stopping
            improved = (val_loss + float(args.min_delta)) < float(best_val_loss)
            if improved:
                best_val_loss = float(val_loss)
                epochs_no_improve = 0
                if args.save_best:
                    # save with epoch label 'best'
                    best_ckpt_path = save_ckpt(model, args.ckpt_dir, args.map_wid, args.agent_num, f"best_{epoch}", args.stage)
            else:
                epochs_no_improve += 1

            # periodic save
            if epoch % 50 == 0:
                last_ckpt = save_ckpt(model, args.ckpt_dir, args.map_wid, args.agent_num, epoch, args.stage)

            # check early stopping
            if args.early_stop and epochs_no_improve >= int(args.patience):
                print(f"[EARLY-STOP] No improvement in val_loss for {epochs_no_improve} epochs (patience={args.patience}). Stopping at epoch {epoch}.")
                break

    except KeyboardInterrupt:
        print("[INTERRUPTED] saving checkpoint...")
        last_epoch = epoch if 'epoch' in locals() else 0
        last_ckpt = save_ckpt(model, args.ckpt_dir, args.map_wid, args.agent_num, last_epoch, args.stage)
        return
    finally:
        last_epoch = epoch if 'epoch' in locals() else args.epochs
        last_ckpt = save_ckpt(model, args.ckpt_dir, args.map_wid, args.agent_num, last_epoch, args.stage)


if __name__ == "__main__":
    main()