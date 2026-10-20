"""
Production export script for TamilScript OCR.
Packages trained PyTorch weights, strips optimizer states for compact delivery,
and generates JSON metadata, vocabulary files, preprocessing configs, and SHA-256 hashes.
"""

import os
import sys
import json
import hashlib
import torch
from typing import Dict, Any, Optional

from .config import VOCABULARY, CHAR_TO_IDX, VOCAB_SIZE

def compute_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def export_production_checkpoint(
    source_checkpoint_path: str,
    output_dir: str = "/kaggle/working",
    eval_metrics: Optional[Dict[str, Any]] = None,
    dataset_name: str = "c3rl/IIIT-INDIC-HW-WORDS-Tamil",
    model_architecture: str = "TamilHandwrittenCRNN"
) -> Dict[str, str]:
    """
    Export stripped, production-ready inference checkpoint and companion metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_pt_path = os.path.join(output_dir, "tamil_crnn_best.pt")

    # 1. Load trained checkpoint
    print(f"Loading trained weights from: {source_checkpoint_path}")
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        val_cer = checkpoint.get("val_cer", 0.0)
        epoch = checkpoint.get("epoch", 1)
    else:
        state_dict = checkpoint
        val_cer = 0.0
        epoch = 1

    # 2. Save pure inference checkpoint (model_state_dict + vocab_size only, ~31.4 MB)
    clean_checkpoint = {
        "model_state_dict": state_dict,
        "vocab_size": VOCAB_SIZE,
        "architecture": model_architecture,
        "best_epoch": epoch,
        "val_cer": val_cer
    }
    torch.save(clean_checkpoint, target_pt_path)
    file_size_mb = os.path.getsize(target_pt_path) / (1024 * 1024)
    sha256_hash = compute_sha256(target_pt_path)
    print(f"Exported clean inference checkpoint: {target_pt_path} ({file_size_mb:.2f} MB, SHA-256: {sha256_hash[:12]}...)")

    # 3. Export vocabulary.json
    vocab_path = os.path.join(output_dir, "vocabulary.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": VOCAB_SIZE,
            "blank_token": "<blank>",
            "blank_index": 0,
            "vocabulary": VOCABULARY,
            "char_to_idx": CHAR_TO_IDX
        }, f, ensure_ascii=False, indent=2)

    # 4. Export preprocessing_config.json
    prep_path = os.path.join(output_dir, "preprocessing_config.json")
    with open(prep_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_height": 32,
            "max_width": 768,
            "color_mode": "grayscale",
            "channels": 1,
            "rescaling": "1.0 / 255.0",
            "background_normalization": "invert if mean < 0.3 (ensure dark text on white bg)",
            "interpolation": "cv2.INTER_AREA",
            "unicode_normalization": "NFC"
        }, f, ensure_ascii=False, indent=2)

    # 5. Export requirements-model.txt
    req_path = os.path.join(output_dir, "requirements-model.txt")
    with open(req_path, "w", encoding="utf-8") as f:
        f.write("torch>=2.0.0\nopencv-python-headless>=4.8.0\npillow>=10.0.0\nnumpy>=1.24.0\n")

    # 6. Export model_metadata.json
    meta_path = os.path.join(output_dir, "model_metadata.json")
    metadata = {
        "model_name": "TamilScript OCR",
        "model_architecture": model_architecture,
        "framework": "PyTorch",
        "file_name": "tamil_crnn_best.pt",
        "file_size_mb": round(file_size_mb, 2),
        "sha256": sha256_hash,
        "vocabulary_size": VOCAB_SIZE,
        "blank_token_index": 0,
        "input_tensor_shape": [1, 1, 32, "variable_width"],
        "dataset_source": dataset_name,
        "dataset_license": "Academic and Research Use (IIIT-H / CVIT)",
        "training_epochs": epoch,
        "validation_cer": val_cer,
        "evaluation_metrics": eval_metrics or {},
        "deployable_on_cpu": True
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 7. Export checksums.sha256
    checksum_path = os.path.join(output_dir, "checksums.sha256")
    with open(checksum_path, "w", encoding="utf-8") as f:
        f.write(f"{sha256_hash}  tamil_crnn_best.pt\n")

    print("All production model artifacts successfully exported.")
    return {
        "checkpoint": target_pt_path,
        "metadata": meta_path,
        "vocabulary": vocab_path,
        "preprocessing": prep_path,
        "checksum": checksum_path
    }
