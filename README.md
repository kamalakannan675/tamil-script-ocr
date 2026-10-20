# TamilScript OCR (தமிழ் கையெழுத்து உணர்தல்)

A full-stack, production-ready web application for **handwritten Tamil full-page Optical Character Recognition (OCR)**, text editing, and multi-format document export.

![TamilScript OCR](frontend/public/favicon.svg)

---

## Key Features

1. **Genuine Neural OCR Inference:**
   - Powered by a PyTorch **CRNN** (Convolutional Recurrent Neural Network) architecture with bidirectional LSTM sequence modeling and Connectionist Temporal Classification (CTC) decoding.
   - Never simulates recognition or uses hardcoded Tamil strings.
2. **Full-Page Document Processing:**
   - Supports JPG, JPEG, PNG, WebP, and multipage PDFs.
   - Multipage PDF rendering powered by `pypdfium2` without requiring external OS binaries.
3. **Advanced Computer Vision Preprocessing:**
   - Automatic rotational skew estimation and deskewing via OpenCV affine transformation.
   - Contrast Limited Adaptive Histogram Equalization (CLAHE) for ink enhancement.
   - Horizontal projection profiling designed to preserve Tamil upper pulli dots (`்`) and lower vowel signs (`ு`, `ூ`).
4. **Authentic Unicode Reconstruction:**
   - Returns valid, canonically normalized (`NFC`) Unicode Tamil characters (`U+0B80`–`U+0BFF`).
   - Supports 12 vowels, 18 consonants, Aytham (`ஃ`), Grantha characters (`ஜ`, `ஷ`, `ஸ`, `ஹ`, `க்ஷ`), numbers, and punctuation.
5. **Interactive UI & Two-Panel Workspace:**
   - **Left Panel:** Drag-and-drop upload, image preview, zoom, rotation, grayscale/threshold filters, multipage navigation.
   - **Right Panel:** Rich editable text area with Tamil font typography, live character and word counters, copy, and clear confirmation.
6. **Multi-Format Export:**
   - **TXT:** UTF-8 encoded text with preserved line breaks.
   - **DOCX:** Microsoft Word document formatted with Tamil typography.
   - **PDF:** High-fidelity PDF with embedded Google `Noto Sans Tamil` TrueType font.
7. **Complete Reproducible ML Pipeline:**
   - Data generation, PyTorch Dataset, training loop (`train_crnn.py`), and held-out benchmark evaluation (`evaluate.py`).

---

## Technology Stack

- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Lucide icons
- **Backend:** Python 3.11, Flask, PyTorch (CPU-optimized), OpenCV, Pillow, NumPy
- **Document & PDF Processing:** `pypdfium2`, `python-docx`, `reportlab`
- **Testing:** `pytest`, `pytest-mock`
- **Deployment:** Gunicorn WSGI, Render blueprint (`render.yaml`)

---

## Directory Structure

```
tamil-script-ocr/
├── backend/
│   ├── app/
│   │   ├── api/             # API routes (health, ocr, export)
│   │   ├── core/            # Config, error handlers
│   │   ├── ocr/             # Preprocessing, segmentation, CRNN model, Unicode utils
│   │   └── exports/         # TXT, DOCX, and PDF exporters with NotoSansTamil font
│   ├── models/              # Checkpoint directory (tamil_crnn.pt)
│   ├── tests/               # 21 automated pytest test suites
│   ├── requirements.txt     # Pinned Python dependencies
│   ├── wsgi.py              # Production WSGI entrypoint
│   └── run.py               # Local development server
├── frontend/
│   ├── src/
│   │   ├── components/      # UI panels, modals, badges
│   │   ├── services/        # API client
│   │   ├── types/           # TypeScript definitions
│   │   ├── App.tsx          # Main workspace
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── ml/
│   ├── data/                # Dataset loaders & synthetic line generator
│   ├── training/            # PyTorch CRNN-CTC training pipeline
│   ├── evaluation/          # Held-out benchmark evaluation script
│   └── checkpoints/         # Trained model checkpoints
├── docs/
│   ├── architecture.md      # Detailed system architecture
│   ├── model_requirements.md# Vocabulary and model specifications
│   ├── deployment.md        # Render and local deployment instructions
│   └── testing.md           # Testing and benchmark report
├── render.yaml              # Render deployment configuration
├── .env.example             # Environment variable template
└── .gitignore
```

---

## Quickstart & Local Installation

### 1. Backend Setup

```bash
# In project root:
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Start backend server
python backend/run.py
```
Backend will start on `http://127.0.0.1:5000`.

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```
Frontend will be available on `http://localhost:5173`.

---

## Running Automated Tests

```bash
# Run all backend unit and integration tests
.venv\Scripts\python -m pytest backend/tests/ -v
```

---

## Running Model Evaluation

To evaluate the trained checkpoint on held-out test lines:

```bash
.venv\Scripts\python ml/evaluation/evaluate.py --checkpoint backend/models/tamil_crnn.pt
```

---

## Git Initialization

```bash
git init
git add .
git commit -m "Initial commit: TamilScript OCR platform with genuine PyTorch CRNN and full UI"
```

---

## License

This project is licensed under the MIT License. The embedded `NotoSansTamil-Regular.ttf` font is licensed under the SIL Open Font License (OFL).
