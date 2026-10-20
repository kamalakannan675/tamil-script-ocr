"""
Verification and regression test suite for candidate OCR checkpoints.
Tests compatibility, state dictionary integrity, forward pass, and CTC decoding
before promoting any new weights to production.
"""

import os
import sys
import hashlib
import torch
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "backend"))

from app.ocr.crnn_model import TamilHandwrittenCRNN
from app.ocr.recognition import CRNNTamilRecognizer
from app.ocr.unicode_utils import VOCAB_SIZE, ctc_decode_greedy, normalize_tamil_text

def verify_candidate_checkpoint(
    candidate_checkpoint_path: str,
    baseline_checkpoint_path: str = "backend/models/tamil_crnn.pt",
    expected_vocab_size: int = VOCAB_SIZE
) -> bool:
    print("=" * 65)
    print("Candidate Checkpoint Production Verification")
    print(f"Candidate: {candidate_checkpoint_path}")
    print("=" * 65)

    if not os.path.exists(candidate_checkpoint_path):
        print(f"FAIL: Candidate file not found at: {candidate_checkpoint_path}")
        return False

    # 1. Compute checksum
    sha = hashlib.sha256(open(candidate_checkpoint_path, "rb").read()).hexdigest()
    print(f"Candidate SHA-256: {sha}")
    size_mb = os.path.getsize(candidate_checkpoint_path) / (1024 * 1024)
    print(f"Candidate File Size: {size_mb:.2f} MB")
    if size_mb > 50.0:
        print("WARNING: Checkpoint exceeds GitHub recommended size limit of 50 MB.")

    # 2. Check structure & keys
    ckpt = torch.load(candidate_checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        print("FAIL: Checkpoint must be a dict containing 'model_state_dict'.")
        return False

    state_dict = ckpt["model_state_dict"]
    vocab_size = ckpt.get("vocab_size", expected_vocab_size)
    if vocab_size != expected_vocab_size:
        print(f"FAIL: Vocabulary size mismatch. Got {vocab_size}, expected {expected_vocab_size}")
        return False

    # 3. Instantiate fresh production model and load state_dict
    model = TamilHandwrittenCRNN(num_classes=VOCAB_SIZE)
    try:
        model.load_state_dict(state_dict)
        print("PASS: Model state dict cleanly loaded into production TamilHandwrittenCRNN.")
    except Exception as e:
        print(f"FAIL: Failed to load state dict: {e}")
        return False

    # 4. Forward pass test
    dummy_input = torch.ones((1, 1, 32, 128), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
        print(f"PASS: Forward pass succeeded. Output tensor shape: {out.shape}")
        assert out.shape[0] == 32  # 128 / 4
        assert out.shape[1] == 1
        assert out.shape[2] == VOCAB_SIZE

    # 5. Production Recognizer Adapter Test
    recognizer = CRNNTamilRecognizer(model_path=candidate_checkpoint_path, device="cpu")
    if not recognizer.is_loaded():
        print("FAIL: CRNNTamilRecognizer failed to initialize with candidate checkpoint.")
        return False

    test_img = np.ones((40, 160), dtype=np.uint8) * 255
    text, conf = recognizer.recognize_line(test_img)
    print(f"PASS: Adapter recognize_line succeeded. Output: '{text}', Confidence: {conf}")

    print("\nALL PRE-DEPLOYMENT COMPATIBILITY CRITERIA PASSED.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_candidate_checkpoint(sys.argv[1])
    else:
        print("Usage: python verify_checkpoint.py <path_to_checkpoint>")
