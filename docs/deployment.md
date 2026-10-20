# Deployment Guide: TamilScript OCR

This guide documents deploying TamilScript OCR to both local production environments and cloud platforms such as **Render**.

---

## 1. Local Production Deployment (Windows / Linux)

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- CPU with at least 1 GB RAM

### Step 1: Backend Setup
```bash
# From repository root
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Run backend using Gunicorn (WSGI)
gunicorn --chdir backend wsgi:app -w 2 -b 127.0.0.1:5000 --timeout 120
```

### Step 2: Frontend Setup
```bash
cd frontend
npm install
npm run build
```

The production assets will be built in `frontend/dist`. You can serve them using Nginx, Caddy, or Vite preview (`npm run preview`).

---

## 2. Render Cloud Deployment

The repository includes a ready-to-deploy `render.yaml` blueprint defining two interconnected services:

### A. Web Service: Backend (`tamil-script-ocr-backend`)
- **Environment:** Python 3.11
- **Plan:** Free
- **Build Command:**
  ```bash
  pip install --upgrade pip && pip install -r backend/requirements.txt
  ```
- **Start Command:**
  ```bash
  gunicorn --chdir backend wsgi:app -w 2 -b 0.0.0.0:$PORT --timeout 120
  ```
- **Health Check Path:** `/api/health`
- **Environment Variables:**
  - `FLASK_ENV`: `production`
  - `DEVICE`: `cpu`
  - `MAX_CONTENT_LENGTH`: `26214400` (25 MB)
  - `CORS_ORIGINS`: `*`
  - `CRNN_MODEL_PATH`: `backend/models/tamil_crnn.pt`

### B. Static Site: Frontend (`tamil-script-ocr-frontend`)
- **Build Command:**
  ```bash
  cd frontend && npm install && npm run build
  ```
- **Publish Directory:** `frontend/dist`
- **Environment Variables:**
  - `VITE_API_URL`: `https://tamil-script-ocr-backend.onrender.com/api`

---

## 3. Post-Deployment Verification
1. Access `https://<your-backend-domain>/api/health` — should return `{"status": "healthy"}`.
2. Access `https://<your-backend-domain>/api/ocr/status` — should report `{"available": true, "model_name": "TamilHandwrittenCRNN"}`.
3. Upload a sample Tamil handwritten image via the frontend and test TXT, DOCX, and PDF exports.
