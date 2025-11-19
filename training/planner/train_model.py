from __future__ import annotations
import argparse
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
    """
    B = len(batch)
    maxN = max((len(item["nodes"]) for item in batch), default=0)
    maxA = max((len(item.get("agents", [item.get("agent")]) or []) for item in batch), default=1)

    def _get_k(it: Dict[str, Any]) -> int:
        if "labels_ak" in it:
            return len(it["labels_ak"][0]) if len(it["labels_ak"]) > 0 else 0
        if "labels_k" in it:
            return len(it["labels_k"])  # 兼容单 agent 老格式
        return 0
    maxK = max((_get_k(item) for item in batch), default=0)

    nodes = torch.zeros(B, maxN, 5, dtype=torch.float32)
    node_mask = torch.ones(B, maxN, dtype=torch.bool)
    agents = torch.zeros(B, maxA, 4, dtype=torch.float32)
    depot = torch.zeros(B, 1, 3, dtype=torch.float32)
    labels_ak = torch.full((B, maxA, maxK), fill_value=-1, dtype=torch.long)
    valid_N = torch.zeros(B, dtype=torch.long)
    cap_full = torch.zeros(B, maxA, dtype=torch.float32)

    for b, item in enumerate(batch):
        Ni = len(item["nodes"])
        valid_N[b] = Ni
        if Ni > 0:
            nodes[b, :Ni] = torch.tensor(item["nodes"], dtype=torch.float32)
            mask_i = torch.tensor(item.get("node_mask", [False] * Ni), dtype=torch.bool)
            node_mask[b, :Ni] = mask_i

        # agents
        if "agents" in item and item["agents"] is not None:
            A_i = len(item["agents"]) if isinstance(item["agents"], (list, tuple)) else 1
            for a in range(min(maxA, A_i)):
                ax, ay, s, ta = item["agents"][a]
                agents[b, a] = torch.tensor([ax, ay, s, ta], dtype=torch.float32)
        elif "agent" in item and item["agent"] is not None:
            ax, ay, s, ta = item["agent"]
            agents[b, 0] = torch.tensor([ax, ay, s, ta], dtype=torch.float32)

        dx, dy, td = item["depot"]
        depot[b, 0] = torch.tensor([dx, dy, td], dtype=torch.float32)

        # labels
        if "labels_ak" in item:
            lab = item["labels_ak"]  # List[List[int]] [A,K]
            A_i = len(lab)
            for a in range(min(maxA, A_i)):
                la = torch.tensor(lab[a], dtype=torch.long)
                if la.numel() != maxK:
                    tmp = torch.full((maxK,), 0, dtype=torch.long)
                    tmp[: la.numel()] = la
                    la = tmp
                labels_ak[b, a] = la
        elif "labels_k" in item:
            la = torch.tensor(item["labels_k"], dtype=torch.long)
            if la.numel() != maxK:
                tmp = torch.full((maxK,), 0, dtype=torch.long)
                tmp[: la.numel()] = la
                la = tmp
            labels_ak[b, 0] = la

        # cap_full per agent must come from row's full_capacity (来自 Config.capacity)
        if "full_capacity" not in item:
            raise RuntimeError("Row is missing 'full_capacity' (Config.capacity). Regenerate dataset with full_capacity set to Config.capacity.")
        full_c = float(item["full_capacity"])
        if full_c <= 0:
            raise RuntimeError("Row 'full_capacity' must be > 0 (Config.capacity). Found: {}".format(full_c))
        cap_full[b, :].fill_(full_c)

    return {
        "nodes": nodes,
        "node_mask": node_mask,
        "agents": agents,        # [B,A,4]
        "depot": depot,          # [B,1,3]
        "labels_ak": labels_ak,  # [B,A,K]
        "valid_N": valid_N,
        "cap_full": cap_full,    # [B,A]
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
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--lateness_lambda", type=float, default=0.0)
    # train
    p.add_argument("--k", type=int, default=3, help="每个样本的监督步数，需与数据集一致")
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
    return p


def save_ckpt(model: DVRPNet, ckpt_dir: str, map_wid: int, agent_num: int, epoch: int) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    name = f"planner_{map_wid}_{agent_num}_{epoch}.pt"
    path = os.path.join(ckpt_dir, name)
    torch.save({"model": model.state_dict()}, path)
    print(f"[CKPT] saved => {path}")
    return path


@torch.no_grad()
def evaluate(model: DVRPNet, loader: DataLoader, device: torch.device, lateness_lambda: float, amp: bool, k: int) -> Tuple[float, float]:
    model.eval()
    total_loss, total_cnt, total_corr = 0.0, 0, 0
    for batch in loader:
        nodes = batch["nodes"].to(device)
        node_mask = batch["node_mask"].to(device)
        agents = batch["agents"].to(device)
        depot = batch["depot"].to(device)
        labels_ak = batch["labels_ak"].to(device)  # [B,A,K]

        B = nodes.size(0)
        A = agents.size(1)
        K = labels_ak.size(2)

        # 编码一次
        feats = {"nodes": nodes, "node_mask": node_mask, "depot": depot}
        enc = model.encode(feats)
        cur_mask = enc["node_mask"].clone()  # [B,N]
        ag = agents.clone()
        # cap_full must be provided by dataset (from Config.capacity). No fallback allowed.
        if "cap_full" not in batch:
            raise RuntimeError("Batch missing 'cap_full' — dataset must include full_capacity (Config.capacity). Regenerate data if needed.")
        cap_full = batch["cap_full"].to(device)

        for step in range(K):
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model.decode(
                    enc_nodes=enc["H_nodes"],
                    enc_depot=enc["H_depot"],
                    node_mask=cur_mask,
                    agents_tensor=ag,
                    nodes=nodes,
                    lateness_lambda=lateness_lambda,
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
                loss = F.cross_entropy(logits_flat, targets_flat)
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
            pred = torch.argmax(logits, dim=-1)  # [B,A] (0=depot, 1..N=node)
            total_loss += loss.item() * B
            total_cnt += B * A
            total_corr += (pred == labels_step).sum().item()

            # Teacher-forcing 使用真值标签更新状态
            sel = labels_step  # [B,A] (0=depot, 1..N=node)
            Nn = nodes.size(1)
            # 更新 mask（本步被任一 agent 选中的节点标 True）；注意 nodes 映射 idx-1
            oh = torch.zeros(B, Nn, dtype=torch.bool, device=device)
            for b in range(B):
                for a in range(A):
                    idx = int(sel[b, a].item())
                    if 1 <= idx <= Nn:
                        oh[b, idx - 1] = True
            cur_mask = cur_mask | oh
            # 更新各 agent 位置/时间/容量（depot 则恢复满容量）
            for b in range(B):
                for a in range(A):
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

    # 数据路径
    train_path = os.path.join(args.data_dir, f"{args.prefix}_train_{args.map_wid}_{args.agent_num}.pt")
    val_path = os.path.join(args.data_dir, f"{args.prefix}_val_{args.map_wid}_{args.agent_num}.pt")
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
    model = DVRPNet(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    last_ckpt = None
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
                labels_ak = batch["labels_ak"].to(device)  # [B,A,K]
                valid_N = batch["valid_N"].to(device)
                # 可选：打印批次统计
                # print(_tensor_stats("Train batch nodes", nodes))
                # print(_tensor_stats("Train batch depot", depot))
                # print(_tensor_stats("Train batch agents", agents))
                # print(_tensor_stats("Train batch labels_ak", labels_ak))

                # 编码一次
                feats = {"nodes": nodes, "node_mask": node_mask, "depot": depot}
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
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        logits = model.decode(
                            enc_nodes=enc["H_nodes"],
                            enc_depot=enc["H_depot"],
                            node_mask=cur_mask,
                            agents_tensor=ag,
                            nodes=nodes,
                            lateness_lambda=args.lateness_lambda,
                        )  # [B,A,N+1]
                        labels_step = labels_ak[:, :, step]                  # [B,A]
                        logits_flat = logits.reshape(-1, logits.size(-1))    # [B*A,N+1]
                        targets_flat = labels_step.reshape(-1)               # [B*A]
                        loss_step = F.cross_entropy(logits_flat, targets_flat)
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
                    epoch_loss += loss_step.item() * Bsz
                    epoch_cnt += Bsz * A

                    # Teacher-forcing: 使用真值标签更新 mask/agents
                    sel = labels_step  # [B,A] (0=depot, 1..N=node)
                    Nn = nodes.size(1)
                    # 更新 mask
                    oh = torch.zeros(Bsz, Nn, dtype=torch.bool, device=device)
                    for b in range(Bsz):
                        for a in range(A):
                            idx = int(sel[b, a].item())
                            if 1 <= idx <= Nn:
                                oh[b, idx - 1] = True
                    cur_mask = cur_mask | oh
                    # 更新 agents 状态（depot 则恢复满容量）
                    for b in range(Bsz):
                        for a in range(A):
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
            val_loss, val_acc = evaluate(model, val_loader, device, args.lateness_lambda, args.amp, args.k)
            dt = time.time() - t0
            print(f"[Epoch {epoch:03d}] train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_acc@step={val_acc:.4f} | {dt:.1f}s")

            if epoch % 50 == 0:
                last_ckpt = save_ckpt(model, args.ckpt_dir, args.map_wid, args.agent_num, epoch)

    except KeyboardInterrupt:
        print("[INTERRUPTED] saving checkpoint...")
        last_epoch = epoch if 'epoch' in locals() else 0
        last_ckpt = save_ckpt(model, args.ckpt_dir, args.map_wid, args.agent_num, last_epoch)
        return
    finally:
        last_epoch = epoch if 'epoch' in locals() else args.epochs
        last_ckpt = save_ckpt(model, args.ckpt_dir, args.map_wid, args.agent_num, last_epoch)


if __name__ == "__main__":
    main()