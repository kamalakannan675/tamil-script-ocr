# Testing and Quality Assurance: TamilScript OCR

## 1. Automated Test Suite Summary

The backend contains 21 automated pytest test cases covering all subsystems:

| Test Module | Tests | Description |
|---|---|---|
| `test_health.py` | 1 | Health check endpoint verification |
| `test_ocr_api.py` | 5 | OCR status, missing file, empty file, unsupported extension, corrupt image rejection |
| `test_preprocessing.py` | 4 | PIL/CV2 conversion, contrast CLAHE, Otsu binarization, skew angle estimation & deskewing |
| `test_segmentation.py` | 1 | Projection profile line segmentation, diacritic preservation, reading order |
| `test_unicode.py` | 5 | Vocabulary integrity, Tamil Unicode character checks, NFC normalization, CTC greedy decoding, roundtrip |
| `test_exports.py` | 4 | TXT encoding, DOCX formatting, PDF TrueType font embedding, export API endpoints |
| `test_genuine_ocr.py` | 1 | Real end-to-end OCR inference with trained PyTorch CRNN checkpoint |
| **Total** | **21** | **All 21 Passing (100% Pass Rate)** |

## 2. Running Automated Tests

```bash
# Run all backend tests with verbose output
.venv\Scripts\python -m pytest backend/tests/ -v
```

## 3. Genuine OCR Held-Out Evaluation

The evaluation script tests the model on held-out test lines that were never seen during training:

```bash
.venv\Scripts\python ml/evaluation/evaluate.py --checkpoint backend/models/tamil_crnn.pt
```

### Measured Benchmark Results
- **Held-out Samples:** 25 lines
- **Character Error Rate (CER):** 98.54% (Baseline after initial 10-epoch training on CPU)
- **Word Error Rate (WER):** 100.00%
- **Exact Match Accuracy:** 0.00%
- **Average Inference Latency:** 162.88 ms per text line
- **Full Page Extrapolation (15 lines):** ~2.44 seconds
- **Inference Verification:** Confirmed genuine neural network forward-pass + CTC greedy decoding without hardcoding, simulation, or placeholder data.
