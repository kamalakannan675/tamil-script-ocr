"""
Tamil Unicode utilities: Vocabulary definition, CTC decoding, Unicode normalization,
and validation of Tamil script characters.
"""

import unicodedata
from typing import List, Dict, Tuple

# Full Tamil Unicode Vocabulary
# CTC Blank is at index 0
BLANK_TOKEN = "<blank>"

TAMIL_VOWELS = [
    "அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ", "ஃ"
]

TAMIL_CONSONANTS = [
    "க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம",
    "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன"
]

TAMIL_GRANTHA = [
    "ஜ", "ஷ", "ஸ", "ஹ", "க்ஷ", "ஸ்ரீ"
]

TAMIL_SIGNS = [
    "்",  # Pulli / Virama
    "ா", "ி", "ீ", "ு", "ூ", "ெ", "ே", "ை", "ொ", "ோ", "ௌ"  # Dependent vowels
]

TAMIL_NUMERALS = [
    "௦", "௧", "௨", "௩", "௪", "௫", "௬", "௭", "௮", "௯", "௰", "௱", "௲"
]

ASCII_PUNCTUATION_AND_DIGITS = [
    " ", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ".", ",", "!", "?", "-", ":", ";", "'", "\"", "(", ")", "/", "%"
]

# Build deterministic vocabulary
VOCABULARY = [BLANK_TOKEN] + TAMIL_VOWELS + TAMIL_CONSONANTS + TAMIL_GRANTHA + TAMIL_SIGNS + TAMIL_NUMERALS + ASCII_PUNCTUATION_AND_DIGITS

# Mapping dictionaries
CHAR_TO_IDX: Dict[str, int] = {char: idx for idx, char in enumerate(VOCABULARY)}
IDX_TO_CHAR: Dict[int, str] = {idx: char for idx, char in enumerate(VOCABULARY)}
VOCAB_SIZE = len(VOCABULARY)

def normalize_tamil_text(text: str) -> str:
    """
    Standardize Tamil text using Unicode NFC normalization.
    Ensures combined forms are canonical, removes \r carriage returns,
    and preserves meaningful newlines.
    """
    if not text:
        return ""
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    clean_chars = [c for c in clean_text if c in ("\n", "\t") or ord(c) >= 32]
    normalized = unicodedata.normalize("NFC", "".join(clean_chars))
    return normalized

def is_tamil_character(char: str) -> bool:
    """Check if a character belongs to the Tamil Unicode block (U+0B80 - U+0BFF)."""
    if not char:
        return False
    cp = ord(char[0])
    return 0x0B80 <= cp <= 0x0BFF

def ctc_decode_greedy(predictions: List[int]) -> str:
    """
    Greedy CTC decoding algorithm:
    1. Collapse consecutive identical tokens.
    2. Remove blank tokens (index 0).
    3. Convert remaining token indices to Unicode string.
    4. Normalize the resulting Tamil Unicode string.
    """
    collapsed = []
    prev_idx = -1
    
    for idx in predictions:
        if idx != prev_idx:
            if idx != 0 and idx in IDX_TO_CHAR:
                collapsed.append(IDX_TO_CHAR[idx])
            prev_idx = idx
            
    raw_text = "".join(collapsed)
    return normalize_tamil_text(raw_text)

def ctc_decode_batch(batch_predictions: List[List[int]]) -> List[str]:
    """Decode a batch of sequence predictions."""
    return [ctc_decode_greedy(seq) for seq in batch_predictions]

def text_to_indices(text: str) -> List[int]:
    """
    Convert a ground-truth Unicode text line into a sequence of token indices.
    Skips unknown characters or replaces them gracefully.
    """
    indices = []
    normalized = unicodedata.normalize("NFC", text)
    for char in normalized:
        if char in CHAR_TO_IDX:
            indices.append(CHAR_TO_IDX[char])
        elif char == "“" or char == "”":
            indices.append(CHAR_TO_IDX.get("\"", 0))
        elif char == "‘" or char == "’":
            indices.append(CHAR_TO_IDX.get("'", 0))
    return indices
