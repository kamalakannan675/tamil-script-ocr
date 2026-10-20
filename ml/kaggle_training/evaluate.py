"""
Comprehensive evaluation script for Tamil Handwritten OCR models.
Calculates CER (Character Error Rate), WER (Word Error Rate), Exact Match %,
latency, and exports detailed analysis reports and failure cases.
"""

import os
import json
import time
import csv
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from .config import VOCAB_SIZE, IDX_TO_CHAR
from .dataset_loader import normalize_text

def ctc_decode_greedy(predictions: List[int]) -> str:
    """Standard greedy CTC decoder collapsing duplicate frames and removing blank (0)."""
    collapsed = []
    prev_idx = -1
    for idx in predictions:
        if idx != prev_idx:
            if idx != 0 and idx in IDX_TO_CHAR:
                collapsed.append(IDX_TO_CHAR[idx])
            prev_idx = idx
    return normalize_text("".join(collapsed))

def levenshtein(seq1: list, seq2: list) -> int:
    """Edit distance between two sequences (characters or words)."""
    if len(seq1) < len(seq2):
        return levenshtein(seq2, seq1)
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

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_eval_samples: int = 1000
) -> Dict[str, Any]:
    """
    Evaluates model across dataloader and returns comprehensive metrics.
    """
    model.eval()
    total_char_edits = 0
    total_ref_chars = 0
    total_word_edits = 0
    total_ref_words = 0
    exact_matches = 0
    total_samples = 0
    
    latencies = []
    misrecognized_cases = []
    predictions_log = []

    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= max_eval_samples:
                break
                
            images = batch["images"].to(device)
            texts = batch["texts"]
            
            t0 = time.perf_counter()
            log_probs = model(images)  # [T, B, num_classes]
            lat_batch = (time.perf_counter() - t0) * 1000.0 / images.size(0)
            latencies.append(lat_batch)
            
            # Argmax over vocabulary
            probs = torch.exp(log_probs)
            pred_indices = torch.argmax(probs, dim=2).permute(1, 0)  # [B, T]

            for i in range(images.size(0)):
                if total_samples >= max_eval_samples:
                    break
                    
                seq = pred_indices[i].cpu().numpy().tolist()
                pred_text = ctc_decode_greedy(seq)
                gt_text = normalize_text(texts[i])
                
                # CER
                char_edits = levenshtein(list(gt_text), list(pred_text))
                ref_len = max(1, len(gt_text))
                total_char_edits += char_edits
                total_ref_chars += ref_len
                
                # WER
                gt_words = gt_text.split()
                pred_words = pred_text.split()
                word_edits = levenshtein(gt_words, pred_words)
                total_word_edits += word_edits
                total_ref_words += max(1, len(gt_words))
                
                # Exact match
                is_exact = (gt_text == pred_text)
                if is_exact:
                    exact_matches += 1
                else:
                    if len(misrecognized_cases) < 100:
                        misrecognized_cases.append({
                            "ground_truth": gt_text,
                            "prediction": pred_text,
                            "char_edits": char_edits,
                            "cer": round(char_edits / float(ref_len), 4)
                        })

                predictions_log.append({
                    "gt": gt_text,
                    "pred": pred_text,
                    "exact": is_exact,
                    "cer": round(char_edits / float(ref_len), 4)
                })
                total_samples += 1

    final_cer = total_char_edits / float(total_ref_chars) if total_ref_chars > 0 else 1.0
    final_wer = total_word_edits / float(total_ref_words) if total_ref_words > 0 else 1.0
    exact_acc = (exact_matches / float(total_samples)) * 100.0 if total_samples > 0 else 0.0
    avg_latency_ms = float(np.mean(latencies)) if latencies else 0.0

    return {
        "samples_evaluated": total_samples,
        "cer": round(final_cer, 4),
        "wer": round(final_wer, 4),
        "exact_match_pct": round(exact_acc, 2),
        "average_latency_ms": round(avg_latency_ms, 2),
        "misrecognized_samples": misrecognized_cases,
        "predictions_log": predictions_log
    }

def save_evaluation_artifacts(results: Dict[str, Any], output_dir: str, baseline_cer: float = 0.9854):
    """Save benchmark CSVs and evaluation JSONs."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. evaluation_report.json
    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    # 2. baseline_vs_candidate.csv
    comparison_path = os.path.join(output_dir, "baseline_vs_candidate.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline (10 epochs synthetic)", "Trained Candidate", "Absolute Improvement"])
        writer.writerow(["CER", f"{baseline_cer * 100:.2f}%", f"{results['cer'] * 100:.2f}%", f"{(baseline_cer - results['cer']) * 100:.2f}%"])
        writer.writerow(["WER", "100.00%", f"{results['wer'] * 100:.2f}%", f"{(1.0 - results['wer']) * 100:.2f}%"])
        writer.writerow(["Exact Match", "0.00%", f"{results['exact_match_pct']:.2f}%", f"{results['exact_match_pct']:.2f}%"])
        writer.writerow(["Average Latency (ms)", "~15-30 ms", f"{results['average_latency_ms']:.2f} ms", "N/A"])
        
    # 3. misrecognized_examples.csv
    mis_path = os.path.join(output_dir, "misrecognized_examples.csv")
    with open(mis_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Ground Truth", "Predicted", "Char Edits", "CER"])
        for idx, item in enumerate(results.get("misrecognized_samples", [])):
            writer.writerow([idx + 1, item["ground_truth"], item["prediction"], item["char_edits"], item["cer"]])
            
    print(f"Artifacts exported to {output_dir}:")
    print(f" - {report_path}")
    print(f" - {comparison_path}")
    print(f" - {mis_path}")
