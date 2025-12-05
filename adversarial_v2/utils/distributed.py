"""
Distributed Training Utilities

Provides helper functions for multi-GPU training using PyTorch DDP.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    """Check if distributed training is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(
    num_gpus: int,
    local_rank: int,
    backend: str = "nccl",
) -> Tuple[torch.device, bool]:
    """
    Setup distributed training environment.
    
    Args:
        num_gpus: Number of GPUs to use
        local_rank: Local rank of this process
        backend: Distributed backend ("nccl" for GPU, "gloo" for CPU)
        
    Returns:
        device: torch device for this process
        is_distributed: True if using distributed training
    """
    if num_gpus <= 1:
        # Single GPU mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, False
    
    # Multi-GPU mode with DDP
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA")
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group if not already done
    if not dist.is_initialized():
        # Use environment variables set by torchrun/torch.distributed.launch
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(local_rank)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(num_gpus)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(local_rank)
        
        # Use localhost for single-node multi-GPU
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"
        
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=num_gpus,
            rank=local_rank,
        )
    
    return device, True


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if is_distributed():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """
    Wrap model with DistributedDataParallel if distributed training is active.
    
    Args:
        model: PyTorch model to wrap
        device: Device the model is on
        find_unused_parameters: Set True if some parameters might not be used
        
    Returns:
        Model wrapped with DDP (or original model if not distributed)
    """
    if not is_distributed():
        return model
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    # Wrap with DDP
    return DDP(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Unwrap DDP model to get the underlying module.
    
    Args:
        model: Model that might be wrapped in DDP
        
    Returns:
        Underlying module
    """
    if isinstance(model, DDP):
        return model.module
    return model


def reduce_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """
    Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        op: Reduction operation ("mean" or "sum")
        
    Returns:
        Reduced tensor
    """
    if not is_distributed():
        return tensor
    
    # Clone to avoid modifying the original
    rt = tensor.clone()
    
    if op == "sum":
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    elif op == "mean":
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= get_world_size()
    else:
        raise ValueError(f"Unknown reduction op: {op}")
    
    return rt


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all processes.
    
    Args:
        tensor: Tensor to broadcast
        src: Source rank
        
    Returns:
        Broadcasted tensor
    """
    if not is_distributed():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def sync_params(model: nn.Module, src: int = 0):
    """
    Synchronize model parameters from source rank to all processes.
    
    Args:
        model: Model to synchronize
        src: Source rank
    """
    if not is_distributed():
        return
    
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def barrier():
    """Synchronize all processes."""
    if is_distributed():
        dist.barrier()


def print_rank0(*args, **kwargs):
    """Print only on rank 0."""
    if is_main_process():
        print(*args, **kwargs)
