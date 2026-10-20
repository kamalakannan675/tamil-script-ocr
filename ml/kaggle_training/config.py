"""
Configuration parameters for Kaggle GPU Training of TamilScript OCR models.
"""

import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainingConfig:
    # Environment & Device
    seed: int = 42
    use_amp: bool = True  # Automatic Mixed Precision for fast Tensor Core training
    num_workers: int = 2  # Kaggle standard CPU workers
    
    # Dataset Configuration
    dataset_name: str = "c3rl/IIIT-INDIC-HW-WORDS-Tamil"
    local_dataset_path: str = "/kaggle/input/iiit-indic-hw-words-tamil"
    uthcd_path: str = "/kaggle/input/tamil-hwcr"
    
    # Image Dimensions
    target_height: int = 32
    max_width: int = 768
    in_channels: int = 1  # Grayscale
    
    # Training Hyperparameters
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 5e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    gradient_clip_val: float = 5.0
    patience: int = 7  # Early stopping patience
    
    # Paths (Kaggle friendly)
    output_dir: str = "/kaggle/working"
    checkpoint_dir: str = "/kaggle/working/checkpoints"
    export_dir: str = "/kaggle/working/exports"
    
    # Fallback paths for local testing
    local_output_dir: str = "ml/kaggle_training/output"
    
    def get_checkpoint_dir(self) -> str:
        if os.path.exists("/kaggle"):
            return self.checkpoint_dir
        os.makedirs(self.local_output_dir, exist_ok=True)
        return self.local_output_dir

# Deterministic Tamil 87-character Vocabulary matching production exactly
BLANK_TOKEN = "<blank>"
TAMIL_VOWELS = ["அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ", "ஃ"]
TAMIL_CONSONANTS = [
    "க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம",
    "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன"
]
TAMIL_GRANTHA = ["ஜ", "ஷ", "ஸ", "ஹ", "க்ஷ", "ஸ்ரீ"]
TAMIL_SIGNS = ["்", "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"]
TAMIL_NUMERALS = ["௦", "௧", "௨", "௩", "௪", "௫", "௬", "௭", "௮", "௯", "௰", "௱", "௲"]
ASCII_PUNCTUATION_AND_DIGITS = [
    " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ".", ",", "!", "?", "-", ":", ";", "'", "\"", "(", ")", "/", "%"
]

VOCABULARY: List[str] = [BLANK_TOKEN] + TAMIL_VOWELS + TAMIL_CONSONANTS + TAMIL_GRANTHA + TAMIL_SIGNS + TAMIL_NUMERALS + ASCII_PUNCTUATION_AND_DIGITS
CHAR_TO_IDX = {char: idx for idx, char in enumerate(VOCABULARY)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(VOCABULARY)}
VOCAB_SIZE = len(VOCABULARY)  # Exactly 87
