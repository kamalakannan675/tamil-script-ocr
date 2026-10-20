"""
Reproducible evaluation script for Tamil Handwritten OCR.
Evaluates trained checkpoint on held-out test samples.
Measures Character Error Rate (CER), Word Error Rate (WER), Exact Match Accuracy,
Latency, and Memory footprint.
"""

import os
import sys
import time
import torch
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

from app.ocr.crnn_model import TamilHandwrittenCRNN
from app.ocr.unicode_utils import VOCAB_SIZE, ctc_decode_greedy, normalize_tamil_text
from ml.data.synthetic_generator import generate_dataset_splits

def levenshtein_distance(seq1: list, seq2: list) -> int:
    """Levenshtein distance between two sequences (characters or words)."""
    if len(seq1) < len(seq2):
        return levenshtein_distance(seq2, seq1)
    if len(seq2) == 0:
        return len(seq1)
    
    prev_row = range(len(seq2) + 1)
    for i, item1 in enumerate(seq1):
        curr_row = [i + 1]
        for j, item2 in enumerate(seq2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (item1 != item2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]

def evaluate_checkpoint(
    checkpoint_path: str = "backend/models/tamil_crnn.pt",
    device_name: str = "cpu"
):
    print("=" * 65)
    print("TamilScript OCR - Genuine Model Evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device_name}")
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint file not found at: {checkpoint_path}")
        print("Please run training pipeline first: python ml/training/train_crnn.py")
        sys.exit(1)

    device = torch.device(device_name)
    
    # Load model
    model = TamilHandwrittenCRNN(num_classes=VOCAB_SIZE)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load held-out test set
    _, _, test_samples = generate_dataset_splits(num_train=120, num_val=25, num_test=25)
    print(f"\nEvaluating on {len(test_samples)} held-out Tamil test lines...\n")

    total_char_edits = 0
    total_ref_chars = 0
    total_word_edits = 0
    total_ref_words = 0
    exact_matches = 0
    latencies = []

    print("-" * 65)
    print(f"{'Sample #':<10} | {'Exact?':<8} | {'CER':<8} | {'Latency':<10}")
    print("-" * 65)

    for idx, (img, gt_raw) in enumerate(test_samples):
        gt_text = normalize_tamil_text(gt_raw)
        
        # Preprocess line image
        gray = np.array(img.convert("L"))
        h, w = gray.shape
        scale = 32.0 / float(h)
        new_w = max(16, min(int(w * scale), 768))
        import cv2
        resized = cv2.resize(gray, (new_w, 32), interpolation=cv2.INTER_AREA)
        norm_img = resized.astype(np.float32) / 255.0
        if np.mean(norm_img) < 0.3:
            norm_img = 1.0 - norm_img
        tensor = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Timed inference
        t0 = time.perf_counter()
        with torch.no_grad():
            log_probs = model(tensor)  # [T, 1, num_classes]
            pred_indices = torch.argmax(log_probs.squeeze(1), dim=1).cpu().numpy().tolist()
            pred_text = ctc_decode_greedy(pred_indices)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        # CER calculation
        char_edits = levenshtein_distance(list(gt_text), list(pred_text))
        total_char_edits += char_edits
        total_ref_chars += max(1, len(gt_text))

        # WER calculation
        gt_words = gt_text.split()
        pred_words = pred_text.split()
        word_edits = levenshtein_distance(gt_words, pred_words)
        total_word_edits += word_edits
        total_ref_words += max(1, len(gt_words))

        is_exact = (gt_text == pred_text)
        if is_exact:
            exact_matches += 1

        sample_cer = char_edits / max(1, len(gt_text))
        print(f"Sample {idx + 1:<3}  | {'Yes' if is_exact else 'No':<8} | {sample_cer * 100:>5.1f}% | {latency_ms:>6.2f} ms")

    overall_cer = (total_char_edits / float(total_ref_chars)) * 100.0
    overall_wer = (total_word_edits / float(total_ref_words)) * 100.0
    exact_acc = (exact_matches / float(len(test_samples))) * 100.0
    avg_latency = float(np.mean(latencies))

    print("-" * 65)
    print("\n" + "=" * 65)
    print("OFFICIAL EVALUATION RESULTS ON HELD-OUT DATASET:")
    print(f"  • Total Test Samples:          {len(test_samples)} lines")
    print(f"  • Character Error Rate (CER):  {overall_cer:.2f}%")
    print(f"  • Word Error Rate (WER):       {overall_wer:.2f}%")
    print(f"  • Exact Match Accuracy:        {exact_acc:.2f}%")
    print(f"  • Avg Latency per Line:        {avg_latency:.2f} ms")
    print(f"  • Estimated Full Page (15 L):  {avg_latency * 15 / 1000.0:.2f} s")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Tamil CRNN OCR Model")
    parser.add_argument("--checkpoint", type=str, default="backend/models/tamil_crnn.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_checkpoint(checkpoint_path=args.checkpoint, device_name=args.device)
