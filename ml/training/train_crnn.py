"""
Training pipeline for line-level Tamil Handwritten Recognition CRNN model.
Uses PyTorch CTCLoss, tracking training loss, validation loss, and Character Error Rate (CER).
Saves model checkpoints with metadata and vocabulary versioning.
"""

import os
import sys
import time
import shutil
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

from app.ocr.crnn_model import TamilHandwrittenCRNN
from app.ocr.unicode_utils import VOCAB_SIZE, ctc_decode_greedy, normalize_tamil_text
from ml.data.synthetic_generator import generate_dataset_splits
from ml.data.dataset import TamilLineDataset, ctc_collate_fn

def compute_levenshtein(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein edit distance."""
    if len(s1) < len(s2):
        return compute_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def evaluate_cer(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    """Compute Character Error Rate (CER) across validation dataset."""
    model.eval()
    total_edits = 0
    total_chars = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["images"].to(device)
            texts = batch["texts"]

            log_probs = model(images)  # [T, B, num_classes]
            probs = torch.exp(log_probs)
            pred_indices = torch.argmax(probs, dim=2)  # [T, B]
            pred_indices = pred_indices.permute(1, 0)  # [B, T]

            for i in range(images.size(0)):
                seq = pred_indices[i].cpu().numpy().tolist()
                pred_text = ctc_decode_greedy(seq)
                gt_text = normalize_tamil_text(texts[i])

                edits = compute_levenshtein(gt_text, pred_text)
                total_edits += edits
                total_chars += max(1, len(gt_text))

    return total_edits / float(total_chars) if total_chars > 0 else 1.0

def train_tamil_crnn(
    epochs: int = 15,
    batch_size: int = 8,
    lr: float = 1e-3,
    device_name: str = "cpu",
    checkpoint_dir: str = "ml/checkpoints",
    backend_model_dir: str = "backend/models"
):
    print("=" * 60)
    print("TamilScript OCR - CRNN Training Pipeline")
    print(f"Device: {device_name} | Epochs: {epochs} | Batch Size: {batch_size} | Vocab Size: {VOCAB_SIZE}")
    print("=" * 60)

    # Set seeds for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device(device_name)

    # 1. Generate non-overlapping dataset splits
    print("\nGenerating training, validation, and held-out test splits...")
    train_samples, val_samples, test_samples = generate_dataset_splits(
        num_train=120,
        num_val=25,
        num_test=20
    )
    print(f"Train samples: {len(train_samples)} | Validation samples: {len(val_samples)} | Test samples: {len(test_samples)}")

    train_ds = TamilLineDataset(train_samples)
    val_ds = TamilLineDataset(val_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=ctc_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=ctc_collate_fn)

    # 2. Instantiate Model, Loss, and Optimizer
    model = TamilHandwrittenCRNN(num_classes=VOCAB_SIZE).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(backend_model_dir, exist_ok=True)
    best_cer = float("inf")
    best_checkpoint_path = os.path.join(checkpoint_dir, "tamil_crnn.pt")

    # 3. Training Loop
    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            log_probs = model(images)  # [T, B, num_classes]

            # Ensure input_lengths <= T
            T = log_probs.size(0)
            input_lengths = torch.clamp(input_lengths, max=T)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        epoch_time = time.time() - start_time

        # Validation CER
        val_cer = evaluate_cer(model, val_loader, device)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val CER: {val_cer * 100:.2f}% | Time: {epoch_time:.2f}s")

        # Save checkpoint if best or at final epoch
        if val_cer <= best_cer or epoch == epochs:
            best_cer = val_cer
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": VOCAB_SIZE,
                "val_cer": val_cer,
                "avg_train_loss": avg_train_loss
            }
            torch.save(checkpoint_data, best_checkpoint_path)

    # 4. Copy best checkpoint to backend models directory
    backend_dest = os.path.join(backend_model_dir, "tamil_crnn.pt")
    shutil.copyfile(best_checkpoint_path, backend_dest)
    print("\n" + "=" * 60)
    print(f"Training Complete! Best Val CER: {best_cer * 100:.2f}%")
    print(f"Saved Checkpoint to: {best_checkpoint_path}")
    print(f"Copied Checkpoint to: {backend_dest}")
    print("=" * 60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Tamil CRNN OCR Model")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    train_tamil_crnn(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device_name=args.device)
