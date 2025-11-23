import os
import sys
import argparse
import torch
# --- ADD: Import DataParallel ---
from torch.nn import DataParallel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
# Lazy import of SummaryWriter to avoid importing TensorBoard (and its
# protobuf C extensions) at module import time which can fail on some systems.
SummaryWriter = None
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime

# Robustly add project root to sys.path regardless of script location (searches for configs.py)
import pathlib
_ROOT = pathlib.Path(__file__).resolve().parent
while _ROOT != _ROOT.parent and not (_ROOT / "configs.py").exists():
    _ROOT = _ROOT.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import CONDITION_DIM

# --- Defaults (can be overridden by CLI) ---
NORMALIZED_DATA_CACHE = "data/ruled_generator/normalized_dataset_extended_diversity.pt"
MODEL_SAVE_PATH = "checkpoints/diffusion_model.pth"
LOG_DIR = "runs/generator_training"

# Hyperparameters (defaults)
EPOCHS = 1000
BATCH_SIZE = 6000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
# Early stopping patience ---
EARLY_STOP_PATIENCE = 10


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train diffusion-based demand generator")
    p.add_argument("--data", type=str, default=NORMALIZED_DATA_CACHE, help="Path to normalized dataset .pt")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE, help="Early stopping patience (epochs)")
    p.add_argument("--log_dir", type=str, default=LOG_DIR)
    p.add_argument("--out", type=str, default=MODEL_SAVE_PATH, help="Path to save best model")
    p.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"], help="Training device")
    p.add_argument("--dry_run", action="store_true", help="Load data and model then exit without training")
    return p

# --- PyTorch Dataset and DataLoader ---

class EpisodeDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    """Pads sequences to the max length in a batch."""
    conditions = torch.stack([item['condition'] for item in batch])
    demands = [item['demands'] for item in batch]
    demands_padded = pad_sequence(demands, batch_first=True, padding_value=0.0)
    return {'conditions': conditions, 'demands': demands_padded}

# --- Training Loop ---

def main():
    args = build_argparser().parse_args()
    data_path = args.data
    out_path = args.out
    log_dir = args.log_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    wd = args.weight_decay
    patience = args.patience

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # --- 1. Load Pre-processed Data ---
    if not os.path.exists(data_path):
        print(f"Error: Normalized data not found at {data_path}")
        print("Please run 'python training/generator/normalize_data.py' first.")
        sys.exit(1)
        
    print(f"Loading normalized data from cache: {data_path}")
    # --- FIX: Load the test_data as well ---
    train_data, val_data, test_data = torch.load(data_path, weights_only=False)
    
    train_dataset = EpisodeDataset(train_data)
    val_dataset = EpisodeDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    print(f"Data loaded: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

    if args.dry_run:
        print("Dry run: dataset and model config loaded successfully.")
        return

    # --- 2. Initialize Model and Logging ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"lr_{lr}_wd_{wd}_batch{batch_size}_{timestamp}"
    # Attempt to import SummaryWriter lazily. If unavailable or import fails
    # (e.g., protobuf/libstdc++ C-extension issues), fall back to a dummy
    # writer that no-ops to keep training runnable for smoke tests.
    try:
        if SummaryWriter is None:
            from torch.utils.tensorboard import SummaryWriter as _SummaryWriter
            SummaryWriter = _SummaryWriter
        writer = SummaryWriter(os.path.join(log_dir, run_name))
        print(f"TensorBoard logs will be saved to: {os.path.join(log_dir, run_name)}")
    except Exception as _err:
        # Provide a minimal no-op writer with the same methods used in this script.
        class _DummyWriter:
            def add_scalar(self, *args, **kwargs):
                return
            def add_hparams(self, *args, **kwargs):
                return
            def close(self, *args, **kwargs):
                return
        writer = _DummyWriter()
        print(f"[WARN] TensorBoard SummaryWriter unavailable, continuing without logging ({_err})")

    model = DemandDiffusionModel(condition_dim=CONDITION_DIM)

    # Use DataParallel to wrap the model for multi-GPU training ---
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    # Counter for early stopping ---
    epochs_no_improve = 0
    global_step = 0

    # --- 3. Training and Validation ---
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [T]")
        for batch in pbar:
            conditions = batch['conditions'].to(device, non_blocking=True)
            demands = batch['demands'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            noise, predicted_noise = model(demands, conditions)
            loss = F.mse_loss(predicted_noise, noise)
            
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            global_step += 1
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # Validation loop (robust to empty val set)
        model.eval()
        if len(val_loader) > 0:
            val_loss = 0.0
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [V]")
                for batch in pbar_val:
                    conditions = batch['conditions'].to(device, non_blocking=True)
                    demands = batch['demands'].to(device, non_blocking=True)
                    noise, predicted_noise = model(demands, conditions)
                    loss = F.mse_loss(predicted_noise, noise)
                    val_loss += loss.item()
                    pbar_val.set_postfix(loss=loss.item())
            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = avg_train_loss
            print("[WARN] Validation set empty; using train loss to drive scheduler/early-stop.")
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # When using DataParallel, the model is wrapped. We need to save the .module attribute.
            state_dict_to_save = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            torch.save(state_dict_to_save, out_path)
            print(f"New best model saved to {out_path} (Val Loss: {best_val_loss:.6f})")
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs with no improvement.")
            break # Exit the training loop

    print("\nTraining complete.")

    # --- 4. Final Testing on the Unseen Test Set ---
    print("\n--- Starting Final Evaluation on Test Set ---")
    print(f"Loading best model from: {out_path}")

    # --- FIX: Load state_dict into the base model before wrapping ---
    # Create a new instance of the base model
    eval_model = DemandDiffusionModel(condition_dim=CONDITION_DIM)
    # Load the saved weights
    eval_model.load_state_dict(torch.load(out_path))
    
    # Wrap it for evaluation if using multiple GPUs
    if torch.cuda.device_count() > 1:
        eval_model = DataParallel(eval_model)
        
    eval_model.to(device)
    eval_model.eval()

    # Create a DataLoader for the test set
    test_dataset = EpisodeDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    test_loss = 0.0
    with torch.no_grad():
        pbar_test = tqdm(test_loader, desc="[Final Test]")
        for batch in pbar_test:
            conditions = batch['conditions'].to(device, non_blocking=True)
            demands = batch['demands'].to(device, non_blocking=True)
            # Use the new eval_model instance
            noise, predicted_noise = eval_model(demands, conditions)
            loss = F.mse_loss(predicted_noise, noise)
            test_loss += loss.item()
            pbar_test.set_postfix(loss=loss.item())

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nFinal Test Loss on unseen data: {avg_test_loss:.6f}")
    writer.add_hparams(
        {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE},
        {"hparam/test_loss": avg_test_loss},
    )

    writer.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    main()