import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
ML_CHECKPOINTS_DIR = BASE_DIR.parent / "ml" / "checkpoints"
EXPORTS_DIR = BASE_DIR / "app" / "exports"
FONTS_DIR = EXPORTS_DIR / "fonts"

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "tamil-ocr-secret-key-2026")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 25 * 1024 * 1024))  # 25 MB
    MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", 5000))
    MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", 20))
    MAX_PROCESSING_TIMEOUT = int(os.getenv("MAX_PROCESSING_TIMEOUT", 120))  # seconds
    
    ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_DOCUMENT_EXTENSIONS = {"pdf"}
    ALLOWED_EXTENSIONS = ALLOWED_IMAGE_EXTENSIONS | ALLOWED_DOCUMENT_EXTENSIONS

    # Model configuration
    # Check default model checkpoint locations
    CRNN_MODEL_PATH = os.getenv(
        "CRNN_MODEL_PATH",
        str(MODELS_DIR / "tamil_crnn.pt")
    )
    ALT_CRNN_PATH = str(ML_CHECKPOINTS_DIR / "tamil_crnn.pt")
    
    DEVICE = os.getenv("DEVICE", "cpu")
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
