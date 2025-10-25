from __future__ import annotations
import argparse
import os
import math
import time
from typing import Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.planner_model.model import DVRPNet
from models.planner_model.dataset import DVRPSyntheticDataset, dvrp_collate


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DVRPNet (single-step decoder) with synthetic dataset")
    # data
    p.add_argument("--dataset_size", type=int, default=100_000)
    p.add_argument("--val_size", type=int, default=5_000)
    p.add_argument("--grid_w", type=int, default=20)
    p.add_argument("--grid_h", type=int, default=20)
    p.add_argument("--min_nodes", type=int, default=5)
    p.add_argument("--max_nodes", type=int, default=30)
    p.add_argument("--t_horizon", type=int, default=50)
    p.add_argument("--due_min_slack", type=int, default=3)
    p.add_argument("--due_max_slack", type=int, default=15)
    p.add_argument("--max_demand", type=int, default=3)
    p.add_argument("--max_capacity", type=int, default=10)
    p.add_argument("--lambda_late", type=float, default=1.0)

    # model
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--lateness_lambda", type=float, default=0.0, help="soft time-window bias during training")

    # train
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--val_interval", type=int, default=500)
    p.add_argument("--ckpt_dir", type=str, default="checkpoints")
    p.add_argument("--ckpt_name", type=str, default="dvrpnet.pt")
    return p


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # logits: [B, Nmax+1], labels: [B]
    pred = torch.argmax(logits, dim=-1)
    correct = (pred == labels).sum().item()
    return correct / labels.size(0)


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # datasets
    train_ds = DVRPSyntheticDataset(
        size=args.dataset_size,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        max_nodes=args.max_nodes,
        min_nodes=args.min_nodes,
        max_demand=args.max_demand,
        max_capacity=args.max_capacity,
        t_horizon=args.t_horizon,
        due_min_slack=args.due_min_slack,
        due_max_slack=args.due_max_slack,
        lambda_late=args.lambda_late,
        seed=args.seed,
    )
    val_ds = DVRPSyntheticDataset(
        size=args.val_size,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        max_nodes=args.max_nodes,
        min_nodes=args.min_nodes,
        max_demand=args.max_demand,
        max_capacity=args.max_capacity,
        t_horizon=args.t_horizon,
        due_min_slack=args.due_min_slack,
        due_max_slack=args.due_max_slack,
        lambda_late=args.lambda_late,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, collate_fn=dvrp_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=dvrp_collate
    )

    # model
    model = DVRPNet(d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    global_step = 0
    best_val_acc = 0.0

    for epoch in tqdm(range(1, args.epochs + 1), desc=f"train: "):
        # 
        model.train()
        t0 = time.time()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", leave=False):
            nodes = batch["nodes"].to(device)           # [B, Nmax, 5]
            node_mask = batch["node_mask"].to(device)   # [B, Nmax] True=masked
            agents = batch["agents"].to(device)         # [B, 1, 4]
            depot = batch["depot"].to(device)           # [B, 1, 3]
            labels = batch["labels"].to(device)         # [B]
            valid_N = batch["valid_N"].to(device)       # [B]

            with torch.cuda.amp.autocast(enabled=args.amp):
                # encode + decode_step（单步），返回 logits over [nodes, depot]
                feats = {"nodes": nodes, "node_mask": node_mask, "agents": agents, "depot": depot}
                logits = model.decode_step(
                    feats,
                    lateness_lambda=args.lateness_lambda,
                    current_time=agents[..., -1].squeeze(1).squeeze(1).mean().item() if False else 0,  # 训练时可设为0或不启用
                )  # [B, Nmax+1]

                # 为每个样本构建 mask，使 pad 的部分不参与选择，同时 depot（最后一列）不屏蔽
                # 逻辑：对超过 valid_N[b] 的列屏蔽；对 <= valid_N[b] 的列用 node_mask；第 Nmax 列为 depot，不屏蔽
                B, Np1 = logits.shape
                Nmax = Np1 - 1
                # 遮掉 pad 区间：索引 [valid_N[b], Nmax)
                range_ids = torch.arange(Nmax, device=device).unsqueeze(0).expand(B, -1)  # [B,Nmax]
                pad_mask = range_ids >= valid_N.unsqueeze(1)  # [B,Nmax]
                eff_mask = torch.logical_or(node_mask, pad_mask)  # [B,Nmax]
                full_mask = torch.cat([eff_mask, torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)
                logits = logits.masked_fill(full_mask, float("-inf"))

                loss = F.cross_entropy(logits, labels)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            global_step += 1
            if global_step % args.log_interval == 0:
                acc = accuracy_from_logits(logits.detach(), labels)
                dt = time.time() - t0
                tqdm.write(f"[Ep {epoch}] step {global_step} | loss {loss.item():.4f} | acc {acc:.4f} | {dt:.1f}s")
                t0 = time.time()

            if global_step % args.val_interval == 0:
                val_loss, val_acc = evaluate(model, val_loader, device, args)
                tqdm.write(f"  >> VAL @ step {global_step}: loss {val_loss:.4f} | acc {val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_ckpt(model, os.path.join(args.ckpt_dir, args.ckpt_name))
                    tqdm.write(f"  >> Saved best checkpoint to {os.path.join(args.ckpt_dir, args.ckpt_name)}")

    # Final save
    save_ckpt(model, os.path.join(args.ckpt_dir, args.ckpt_name))
    print(f"Training finished. Best val acc: {best_val_acc:.4f}. Checkpoint saved.")


@torch.no_grad()
def evaluate(model: DVRPNet, loader: DataLoader, device: torch.device, args: argparse.Namespace):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for batch in loader:
        nodes = batch["nodes"].to(device)
        node_mask = batch["node_mask"].to(device)
        agents = batch["agents"].to(device)
        depot = batch["depot"].to(device)
        labels = batch["labels"].to(device)
        valid_N = batch["valid_N"].to(device)

        with torch.cuda.amp.autocast(enabled=args.amp):
            feats = {"nodes": nodes, "node_mask": node_mask, "agents": agents, "depot": depot}
            logits = model.decode_step(feats, lateness_lambda=args.lateness_lambda, current_time=0)
            B, Np1 = logits.shape
            Nmax = Np1 - 1
            range_ids = torch.arange(Nmax, device=device).unsqueeze(0).expand(B, -1)
            pad_mask = range_ids >= valid_N.unsqueeze(1)
            eff_mask = torch.logical_or(node_mask, pad_mask)
            full_mask = torch.cat([eff_mask, torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)
            logits = logits.masked_fill(full_mask, float("-inf"))

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (torch.argmax(logits, dim=-1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(1, total), total_correct / max(1, total)


def save_ckpt(model: DVRPNet, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


if __name__ == "__main__":
    main()