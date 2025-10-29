import os
import torch
# --- ADD: Import DataParallel ---
from torch.nn import DataParallel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import sys

# Add project root to path to allow importing from other folders
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.generator_model.diffusion_model import DemandDiffusionModel
from agent.generator.data_utils import CONDITION_DIM

# --- Configuration ---
NORMALIZED_DATA_CACHE = "data/ruled_generator/normalized_dataset_extended_diversity.pt"
MODEL_SAVE_PATH = "checkpoints/diffusion_model.pth"
LOG_DIR = "runs/generator_training"

# Hyperparameters
EPOCHS = 1000
BATCH_SIZE = 6000
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
# Early stopping patience ---
EARLY_STOP_PATIENCE = 10 

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Pre-processed Data ---
    if not os.path.exists(NORMALIZED_DATA_CACHE):
        print(f"Error: Normalized data not found at {NORMALIZED_DATA_CACHE}")
        print("Please run 'python utils/normalize_dataset.py' first.")
        sys.exit(1)
        
    print(f"Loading normalized data from cache: {NORMALIZED_DATA_CACHE}")
    # --- FIX: Load the test_data as well ---
    train_data, val_data, test_data = torch.load(NORMALIZED_DATA_CACHE, weights_only=False)
    
    train_dataset = EpisodeDataset(train_data)
    val_dataset = EpisodeDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    print(f"Data loaded: Train batches={len(train_loader)}, Val batches={len(val_loader)}")

    # --- 2. Initialize Model and Logging ---
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name = f"lr_{LEARNING_RATE}_wd_{WEIGHT_DECAY}_batch{BATCH_SIZE}_{timestamp}"
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))
    print(f"TensorBoard logs will be saved to: {os.path.join(LOG_DIR, run_name)}")

    model = DemandDiffusionModel(condition_dim=CONDITION_DIM)

    # Use DataParallel to wrap the model for multi-GPU training ---
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = DataParallel(model)

    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    # Counter for early stopping ---
    epochs_no_improve = 0
    global_step = 0

    # --- 3. Training and Validation ---
    for epoch in range(EPOCHS):
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

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [V]")
            for batch in pbar_val:
                conditions = batch['conditions'].to(device, non_blocking=True)
                demands = batch['demands'].to(device, non_blocking=True)
                noise, predicted_noise = model(demands, conditions)
                loss = F.mse_loss(predicted_noise, noise)
                val_loss += loss.item()
                pbar_val.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # When using DataParallel, the model is wrapped. We need to save the .module attribute.
            state_dict_to_save = model.module.state_dict() if isinstance(model, DataParallel) else model.state_dict()
            torch.save(state_dict_to_save, MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.6f})")
            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered after {EARLY_STOP_PATIENCE} epochs with no improvement.")
            break # Exit the training loop

    print("\nTraining complete.")

    # --- 4. Final Testing on the Unseen Test Set ---
    print("\n--- Starting Final Evaluation on Test Set ---")
    print(f"Loading best model from: {MODEL_SAVE_PATH}")

    # --- FIX: Load state_dict into the base model before wrapping ---
    # Create a new instance of the base model
    eval_model = DemandDiffusionModel(condition_dim=CONDITION_DIM)
    # Load the saved weights
    eval_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
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