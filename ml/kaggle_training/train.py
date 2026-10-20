"""
Modular training pipeline for Tamil Handwritten OCR models on GPU.
Features AMP (Automatic Mixed Precision), Cosine LR Schedule, Early Stopping,
checkpoint resumption, and full CSV metrics logging.
"""

import os
import sys
import time
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .config import TrainingConfig, VOCAB_SIZE
from .models import get_model
from .dataset_loader import TamilOCRDataset, ctc_collate_fn, load_iiit_indic_dataset
from .evaluate import evaluate_model, save_evaluation_artifacts

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    grad_clip: float = 5.0
) -> float:
    """Train single epoch with optional AMP."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                log_probs = model(images)  # [T, B, C]
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            log_probs = model(images)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)

def run_training(
    config: TrainingConfig = TrainingConfig(),
    candidate_name: str = "crnn",
    resume_checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
    """Execute complete training loop and return evaluation results."""
    print("=" * 70)
    print(f"TamilScript OCR GPU Training — Model: {candidate_name.upper()}")
    print("=" * 70)

    # 1. Device check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device} | CUDA Available: {torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")

    torch.manual_seed(config.seed)

    # 2. Load dataset
    train_samples, val_samples, test_samples = load_iiit_indic_dataset()
    if not train_samples:
        raise RuntimeError("No training samples available. Ensure internet is active or dataset is attached.")

    train_ds = TamilOCRDataset(train_samples, is_train=True)
    val_ds = TamilOCRDataset(val_samples, is_train=False)
    test_ds = TamilOCRDataset(test_samples, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=config.num_workers
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=config.num_workers
    )

    # 3. Model, Loss, Optimizer, Scheduler, Scaler
    model = get_model(candidate_name, num_classes=VOCAB_SIZE).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.min_learning_rate)
    scaler = torch.cuda.amp.GradScaler() if (config.use_amp and device.type == "cuda") else None

    start_epoch = 1
    best_val_cer = 1.0
    patience_counter = 0

    checkpoint_dir = config.get_checkpoint_dir()
    os.makedirs(checkpoint_dir, exist_ok=True)
    history_path = os.path.join(checkpoint_dir, "training_history.csv")

    # Resumption support
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        print(f"Resuming from checkpoint: {resume_checkpoint_path}")
        ckpt = torch.load(resume_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_cer = ckpt.get("val_cer", 1.0)
        print(f"Resumed at epoch {start_epoch} with baseline best CER: {best_val_cer:.4f}")

    # CSV Logger
    if not os.path.exists(history_path):
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_cer", "val_wer", "lr", "elapsed_sec"])

    # 4. Training loop
    for epoch in range(start_epoch, config.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, config.gradient_clip_val)
        val_eval = evaluate_model(model, val_loader, device, max_eval_samples=500)
        scheduler.step()
        
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        val_cer = val_eval["cer"]
        val_wer = val_eval["wer"]

        print(f"Epoch {epoch:02d}/{config.epochs:02d} | Loss: {train_loss:.4f} | Val CER: {val_cer * 100:.2f}% | Val WER: {val_wer * 100:.2f}% | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")

        # Save history
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 4), val_cer, val_wer, f"{current_lr:.2e}", round(elapsed, 1)])

        # Save regular resumable checkpoint
        resumable_path = os.path.join(checkpoint_dir, f"{candidate_name}_checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "candidate_name": candidate_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_cer": val_cer,
            "vocab_size": VOCAB_SIZE
        }, resumable_path)

        # Check for best checkpoint
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            patience_counter = 0
            best_model_path = os.path.join(checkpoint_dir, f"{candidate_name}_best.pt")
            torch.save({
                "epoch": epoch,
                "candidate_name": candidate_name,
                "model_state_dict": model.state_dict(),
                "val_cer": val_cer,
                "vocab_size": VOCAB_SIZE
            }, best_model_path)
            print(f" >> [BEST MODEL SAVED] Epoch {epoch} -> Val CER: {val_cer * 100:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered after {config.patience} epochs without improvement.")
                break

    # 5. Final Evaluation on Held-Out Test Set
    print("\nRunning final evaluation on untouched test set with best model...")
    best_ckpt = torch.load(os.path.join(checkpoint_dir, f"{candidate_name}_best.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    test_eval = evaluate_model(model, test_loader, device, max_eval_samples=2000)
    print("=" * 60)
    print("FINAL TEST METRICS:")
    print(f" - Test CER: {test_eval['cer'] * 100:.2f}%")
    print(f" - Test WER: {test_eval['wer'] * 100:.2f}%")
    print(f" - Exact Match Accuracy: {test_eval['exact_match_pct']:.2f}%")
    print(f" - Average Inference Latency: {test_eval['average_latency_ms']:.2f} ms")
    print("=" * 60)

    save_evaluation_artifacts(test_eval, output_dir=checkpoint_dir)
    return test_eval
