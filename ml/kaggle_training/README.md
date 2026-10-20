# TamilScript OCR — Kaggle GPU Training & Model Upgrade Guide

This guide details the procedure to train the **TamilScript OCR** neural recognition model on Kaggle Cloud GPUs, benchmark against our 98.54% CER baseline, and safely promote the new model checkpoint to production on Render without service interruption.

---

## 1. Quick Start on Kaggle

### Step 1: Create a Kaggle Notebook
1. Sign in to [Kaggle](https://www.kaggle.com/).
2. Click **"+ Create"** -> **"New Notebook"**.
3. In the top-right notebook menu, open **Notebook Settings**:
   - **Accelerator**: Select **GPU T4 x2** or **GPU P100**.
   - **Language**: Python 3.
   - **Internet**: Toggle **On** (required to stream `c3rl/IIIT-INDIC-HW-WORDS-Tamil` via Hugging Face `datasets`).

### Step 2: Import the Training Notebook
- Click **File** -> **Import Notebook** -> Select `ml/kaggle_training/kaggle_tamil_ocr_training.ipynb` from this repository.
- Alternatively, copy-paste the cells or push directly via the Kaggle API.

### Step 3: Run Training
- Run all cells (**Run All** / `Ctrl + F9`).
- The notebook will:
  1. Audit GPU capability and memory.
  2. Stream `c3rl/IIIT-INDIC-HW-WORDS-Tamil` (~103k real handwritten Tamil words across Train, Val, and Test splits).
  3. Apply diacritic-preserving augmentations (slight rotations, contrast jitter, stroke dilation/erosion).
  4. Train `TamilHandwrittenCRNN` using PyTorch CTCLoss, AdamW, Cosine Annealing, and Automatic Mixed Precision (AMP).
  5. Checkpoint the model at each epoch and retain the best checkpoint based on validation CER.
  6. Evaluate on the untouched test split and generate evaluation reports (`baseline_vs_candidate.csv`, `evaluation_report.json`, `misrecognized_examples.csv`).
  7. Export pure inference weights to `/kaggle/working/tamil_crnn_best.pt` along with `model_metadata.json` and `checksums.sha256`.

---

## 2. Dataset Strategy & Provenance

| Dataset | Granularity | Split Counts | License / Source |
| :--- | :--- | :--- | :--- |
| **`c3rl/IIIT-INDIC-HW-WORDS-Tamil`** | Real handwritten words | **Train:** 75,700<br>**Val:** 11,600<br>**Test:** 16,200 | Academic & Research Use (IIIT-H / CVIT) |
| **`tamil-hwcr` (uTHCD)** | Isolated characters | 156 classes | Kaggle Competition / Research |
| **Synthetic Line Corpus** | Generated sentences | Variable | In-house fallback (`NotoSansTamil-Regular.ttf`) |

### Label Normalization
All ground-truth labels are standardized using **Unicode Canonical NFC Normalization** (`unicodedata.normalize('NFC', text)`), preserving Tamil combined forms, pulli virama dots (`்`), and dependent vowel signs.

---

## 3. Model Architecture & Vocabulary

### Candidate A: `TamilHandwrittenCRNN` (Recommended for Zero-Downtime Promotion)
- **Feature Extractor:** 7-stage CNN with Batch Normalization and ReLU activations.
  - Stage 1: Conv2d(1, 64, 3, p=1) + MaxPool(2, 2)
  - Stage 2: Conv2d(64, 128, 3, p=1) + MaxPool(2, 2)
  - Stage 3: Conv2d(128, 256, 3, p=1)
  - Stage 4: Conv2d(256, 256, 3, p=1) + MaxPool((2, 1), (2, 1))
  - Stage 5: Conv2d(256, 512, 3, p=1)
  - Stage 6: Conv2d(512, 512, 3, p=1) + MaxPool((2, 1), (2, 1))
  - Stage 7: Conv2d(512, 512, (2, 1), s=1, p=0)
- **Sequence Modeling:** 2-layer Bidirectional LSTM (hidden size 256, dropout 0.25).
- **Projection Head:** Linear(512, 87) + LogSoftmax.
- **Parameters:** 8,227,543 (~8.23 Million).
- **Output Factor:** Width downsampling stride = 4 (`W_out = W_in // 4`).

### Vocabulary Mapping (87 Tokens)
- `Index 0`: `<blank>` (CTC blank token).
- `Indices 1-13`: 13 Tamil independent vowels (`அ`–`ஃ`).
- `Indices 14-31`: 18 Tamil consonants (`க`–`ன`).
- `Indices 32-37`: 6 Grantha letters (`ஜ`, `ஷ`, `ஸ`, `ஹ`, `க்ஷ`, `ஸ்ரீ`).
- `Indices 38-49`: 12 Dependent vowel signs & Pulli (`்`, `ா`–`ௌ`).
- `Indices 50-62`: 13 Tamil numerals (`௦`–`௲`).
- `Indices 63-86`: 24 ASCII digits and punctuation (`space`, `0-9`, `.`, `,`, `!`, `?`, `-`, etc.).

---

## 4. Evaluation & Artifacts

After training completes, the notebook generates the following files in `/kaggle/working/`:

1. **`tamil_crnn_best.pt`**: Stripped inference checkpoint containing only `model_state_dict` and `vocab_size` (~31.4 MB).
2. **`model_metadata.json`**: Architectural metadata, dataset info, and test evaluation metrics.
3. **`baseline_vs_candidate.csv`**: Comparison of Character Error Rate (CER), Word Error Rate (WER), and exact match accuracy against the baseline.
4. **`misrecognized_examples.csv`**: Sample failure cases for qualitative error analysis.
5. **`checksums.sha256`**: SHA-256 fingerprint of the checkpoint for tamper-proof validation.

---

## 5. Safe Model Verification & Production Promotion (Phase 9 & 10)

### Pre-Deployment Verification Protocol
Before replacing the production checkpoint, run the local verification suite on the downloaded checkpoint:

```bash
# 1. Download tamil_crnn_best.pt from Kaggle to your local workspace
# 2. Run automated compatibility verification:
python ml/kaggle_training/verify_checkpoint.py <path_to_downloaded_checkpoint>
```

The verification suite checks:
- Model parameter shapes match `TamilHandwrittenCRNN`.
- State dictionary keys match 1:1.
- Forward pass tensor dimensions on CPU.
- End-to-end inference and CTC greedy decoding with `CRNNTamilRecognizer`.

### Promotion to GitHub & Render
Once verification passes:
1. Back up the existing model:
   ```bash
   cp backend/models/tamil_crnn.pt backend/models/tamil_crnn_backup.pt
   ```
2. Replace with the verified candidate:
   ```bash
   cp <path_to_downloaded_checkpoint> backend/models/tamil_crnn.pt
   ```
3. Run the backend regression test suite:
   ```bash
   pytest backend/tests/ -v
   ```
4. Commit and push to `main`:
   ```bash
   git add backend/models/tamil_crnn.pt
   git commit -m "feat(model): upgrade Tamil OCR checkpoint with Kaggle GPU trained weights on IIIT-INDIC-HW"
   git push origin main
   ```
5. Render will automatically detect the commit on `main`, rebuild, and deploy the updated model.

### Rollback Procedure
If any regression is observed on live traffic:
```bash
cp backend/models/tamil_crnn_backup.pt backend/models/tamil_crnn.pt
git add backend/models/tamil_crnn.pt
git commit -m "revert(model): rollback to previous production checkpoint"
git push origin main
```
