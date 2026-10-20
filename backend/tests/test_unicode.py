import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.ocr.unicode_utils import (
    VOCABULARY,
    VOCAB_SIZE,
    CHAR_TO_IDX,
    IDX_TO_CHAR,
    normalize_tamil_text,
    is_tamil_character,
    ctc_decode_greedy,
    text_to_indices,
)

def test_tamil_vocabulary_integrity():
    assert VOCABULARY[0] == "<blank>"
    assert "அ" in VOCABULARY
    assert "க" in VOCABULARY
    assert "்" in VOCABULARY  # Pulli
    assert "ா" in VOCABULARY  # Vowel sign
    assert "ஜ" in VOCABULARY  # Grantha
    assert VOCAB_SIZE == len(VOCABULARY)
    assert len(CHAR_TO_IDX) == VOCAB_SIZE
    assert len(IDX_TO_CHAR) == VOCAB_SIZE

def test_is_tamil_character():
    assert is_tamil_character("த") is True
    assert is_tamil_character("அ") is True
    assert is_tamil_character("்") is True
    assert is_tamil_character("A") is False
    assert is_tamil_character("1") is False

def test_normalize_tamil_text():
    sample = "தமிழ் \r\n எழுத்துக்கள்"
    normalized = normalize_tamil_text(sample)
    assert "\r" not in normalized
    assert "தமிழ்" in normalized
    assert "\n" in normalized

def test_ctc_greedy_decoding():
    # Sequence with blanks (0) and consecutive duplicates
    indices = [
        CHAR_TO_IDX["த"],
        CHAR_TO_IDX["த"],  # duplicate -> collapse
        0,                 # blank
        CHAR_TO_IDX["ம"],
        CHAR_TO_IDX["ி"],
        0,
        CHAR_TO_IDX["ழ"],
        CHAR_TO_IDX["்"]
    ]
    decoded = ctc_decode_greedy(indices)
    assert decoded == "தமிழ்"

def test_text_to_indices_roundtrip():
    text = "வணக்கம்"
    indices = text_to_indices(text)
    assert len(indices) > 0
    # Every index should be valid
    for idx in indices:
        assert 0 < idx < VOCAB_SIZE
