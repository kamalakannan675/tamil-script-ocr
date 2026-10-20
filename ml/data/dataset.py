"""
PyTorch Dataset and Collate utilities for line-level Tamil Handwritten Recognition.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))
from app.ocr.unicode_utils import text_to_indices, normalize_tamil_text

class TamilLineDataset(Dataset):
    def __init__(self, samples: List[Tuple[Image.Image, str]], target_height: int = 32, max_width: int = 768):
        self.samples = samples
        self.target_height = target_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.samples)

    def preprocess_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Convert to grayscale, scale to fixed height 32, normalize to [0, 1]."""
        gray = np.array(pil_img.convert("L"))
        h, w = gray.shape
        if h == 0 or w == 0:
            canvas = np.ones((self.target_height, 64), dtype=np.float32)
            return torch.tensor(canvas).unsqueeze(0)

        scale = self.target_height / float(h)
        new_w = max(16, min(int(w * scale), self.max_width))
        resized = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

        norm_img = resized.astype(np.float32) / 255.0
        # If dark background, invert to light background
        if np.mean(norm_img) < 0.3:
            norm_img = 1.0 - norm_img

        return torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img, raw_text = self.samples[idx]
        norm_text = normalize_tamil_text(raw_text)
        tensor_img = self.preprocess_image(img)
        indices = text_to_indices(norm_text)
        return {
            "image": tensor_img,
            "text": norm_text,
            "target": torch.tensor(indices, dtype=torch.long),
            "width": tensor_img.shape[-1]
        }

def ctc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pad variable-width line images in batch to the maximum width in the batch.
    Pad targets and construct target lengths for PyTorch CTCLoss.
    """
    batch_size = len(batch)
    max_w = max(item["width"] for item in batch)
    
    # Target height is 32
    padded_images = torch.ones((batch_size, 1, 32, max_w), dtype=torch.float32)
    
    targets = []
    target_lengths = []
    input_lengths = []
    raw_texts = []

    for i, item in enumerate(batch):
        img = item["image"]
        w = item["width"]
        padded_images[i, :, :, :w] = img
        
        target = item["target"]
        targets.append(target)
        target_lengths.append(len(target))
        
        # In our CNN architecture, each pooling step halves the width twice (total stride 4)
        input_len = max(1, w // 4)
        input_lengths.append(input_len)
        raw_texts.append(item["text"])

    targets_concat = torch.cat(targets, dim=0) if targets else torch.tensor([], dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)

    return {
        "images": padded_images,
        "targets": targets_concat,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
        "texts": raw_texts
    }
