"""
Dataset loader and augmentation pipeline for Handwritten Tamil OCR.
Handles IIIT-INDIC-HW-WORDS-Tamil (Hugging Face / local Parquet), uTHCD,
and augmented synthetic lines.
"""

import os
import random
import unicodedata
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.utils.data import Dataset

from .config import CHAR_TO_IDX, VOCAB_SIZE

def normalize_text(text: str) -> str:
    """Canonical Unicode NFC normalization."""
    if not text:
        return ""
    clean = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
    return unicodedata.normalize("NFC", clean)

def text_to_indices(text: str) -> List[int]:
    """Convert Unicode string to sequence of token indices using production vocabulary."""
    indices = []
    norm = normalize_text(text)
    for c in norm:
        if c in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[c])
        elif c in ("“", "”"):
            indices.append(CHAR_TO_IDX.get("\"", 0))
        elif c in ("‘", "’"):
            indices.append(CHAR_TO_IDX.get("'", 0))
    return indices

class TamilDataAugmenter:
    """
    Diacritic-safe image augmentations for handwritten Tamil:
    - Subtle rotation (+/- 4 deg)
    - Mild affine shear (+/- 0.04)
    - Contrast and brightness jitter
    - Mild morphological dilation/erosion (pen thickness variation)
    Avoids destroying Tamil pulli dots or vowel loops.
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def augment(self, pil_img: Image.Image) -> Image.Image:
        if not self.is_train:
            return pil_img

        img = pil_img.convert("L")
        
        # 1. Subtle rotation (+/- 3.5 deg)
        if random.random() < 0.4:
            angle = random.uniform(-3.5, 3.5)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=255)

        # 2. Contrast and brightness jitter
        if random.random() < 0.4:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.85, 1.25))

        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.9, 1.15))

        # 3. Mild stroke dilation / erosion (mimic ballpoint vs gel pens)
        if random.random() < 0.25:
            arr = np.array(img)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            if random.random() > 0.5:
                # Thicken ink
                arr = cv2.erode(arr, kernel, iterations=1)
            else:
                # Thinner ink
                arr = cv2.dilate(arr, kernel, iterations=1)
            img = Image.fromarray(arr)

        return img

class TamilOCRDataset(Dataset):
    """
    PyTorch Dataset for Tamil Handwritten Words & Lines.
    Supports in-memory PIL images, HuggingFace dataset items, or disk filepaths.
    """
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        target_height: int = 32,
        max_width: int = 768,
        is_train: bool = True
    ):
        self.samples = samples
        self.target_height = target_height
        self.max_width = max_width
        self.augmenter = TamilDataAugmenter(is_train=is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def preprocess_image(self, pil_img: Image.Image) -> torch.Tensor:
        """
        Grayscale conversion, aspect-ratio scaling to height 32,
        clamping width to [16, max_width], normalizing to [0, 1].
        """
        aug_img = self.augmenter.augment(pil_img)
        gray = np.array(aug_img.convert("L"))
        h, w = gray.shape
        if h == 0 or w == 0:
            canvas = np.ones((self.target_height, 64), dtype=np.float32)
            return torch.tensor(canvas).unsqueeze(0)

        scale = self.target_height / float(h)
        new_w = max(16, min(int(w * scale), self.max_width))
        resized = cv2.resize(gray, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

        norm_img = resized.astype(np.float32) / 255.0
        # If dark background, invert so background is ~1.0
        if np.mean(norm_img) < 0.3:
            norm_img = 1.0 - norm_img

        return torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        raw_img = item["image"]
        if isinstance(raw_img, str) and os.path.exists(raw_img):
            raw_img = Image.open(raw_img)
        elif not isinstance(raw_img, Image.Image):
            raw_img = Image.fromarray(raw_img)

        norm_text = normalize_text(item["text"])
        tensor_img = self.preprocess_image(raw_img)
        indices = text_to_indices(norm_text)

        return {
            "image": tensor_img,
            "text": norm_text,
            "target": torch.tensor(indices, dtype=torch.long),
            "width": tensor_img.shape[-1]
        }

def ctc_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for variable-width line/word images.
    Pads image tensors along width, constructs PyTorch CTCLoss inputs.
    """
    batch_size = len(batch)
    max_w = max(item["width"] for item in batch)
    
    padded_images = torch.ones((batch_size, 1, 32, max_w), dtype=torch.float32)
    targets = []
    target_lengths = []
    input_lengths = []
    texts = []

    for i, item in enumerate(batch):
        img = item["image"]
        w = item["width"]
        padded_images[i, :, :, :w] = img
        
        target = item["target"]
        targets.append(target)
        target_lengths.append(len(target))
        
        # Stride 4 in CNN feature extractor
        input_len = max(1, w // 4)
        input_lengths.append(input_len)
        texts.append(item["text"])

    targets_concat = torch.cat(targets, dim=0) if targets else torch.tensor([], dtype=torch.long)
    return {
        "images": padded_images,
        "targets": targets_concat,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts
    }

def load_iiit_indic_dataset(max_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load c3rl/IIIT-INDIC-HW-WORDS-Tamil using HuggingFace datasets library.
    Returns (train_samples, val_samples, test_samples).
    """
    try:
        from datasets import load_dataset
        print("Loading 'c3rl/IIIT-INDIC-HW-WORDS-Tamil' from Hugging Face...")
        hf_dataset = load_dataset("c3rl/IIIT-INDIC-HW-WORDS-Tamil")
        
        train_split = hf_dataset["train"]
        val_split = hf_dataset["validation"]
        test_split = hf_dataset["test"]
        
        train_samples = [{"image": r["image"], "text": r["text"]} for r in train_split]
        val_samples = [{"image": r["image"], "text": r["text"]} for r in val_split]
        test_samples = [{"image": r["image"], "text": r["text"]} for r in test_split]
        
        if max_samples:
            train_samples = train_samples[:max_samples]
            val_samples = val_samples[:max(100, max_samples // 6)]
            test_samples = test_samples[:max(100, max_samples // 5)]
            
        print(f"Loaded splits -> Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        return train_samples, val_samples, test_samples
    except Exception as e:
        print(f"Hugging Face dataset load failed or offline: {e}")
        return [], [], []
