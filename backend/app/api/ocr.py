import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from ..ocr.engine import TamilOCREngine
from ..core.errors import APIError

ocr_bp = Blueprint("ocr", __name__)
_engine_instance = None

def get_engine():
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = TamilOCREngine()
    return _engine_instance

@ocr_bp.route("/ocr/status", methods=["GET"])
def get_ocr_status():
    """Return genuine model and OCR engine status."""
    engine = get_engine()
    status = engine.get_status()
    return jsonify(status)

@ocr_bp.route("/ocr", methods=["POST"])
def perform_ocr():
    """
    Perform full-page Tamil OCR on uploaded image or PDF document.
    """
    if "file" not in request.files:
        raise APIError("No file was uploaded in request field 'file'.", status_code=400, code="MISSING_FILE")

    uploaded_file = request.files["file"]
    if not uploaded_file or uploaded_file.filename == "":
        raise APIError("Empty filename provided.", status_code=400, code="EMPTY_FILENAME")

    raw_filename = uploaded_file.filename
    safe_name = secure_filename(raw_filename) or "uploaded_document"
    
    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise APIError("Uploaded file is empty.", status_code=400, code="EMPTY_FILE")

    engine = get_engine()
    result = engine.process_document(file_bytes=file_bytes, filename=raw_filename)
    return jsonify(result)

@ocr_bp.route("/ocr/page", methods=["POST"])
def perform_ocr_page():
    """
    Perform single-page inference for image.
    """
    if "file" not in request.files:
        raise APIError("No file uploaded.", status_code=400, code="MISSING_FILE")

    uploaded_file = request.files["file"]
    file_bytes = uploaded_file.read()
    if not file_bytes:
        raise APIError("Uploaded file is empty.", status_code=400, code="EMPTY_FILE")

    import io
    from PIL import Image
    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()
        pil_img = Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        raise APIError(f"Invalid image file: {e}", status_code=400, code="INVALID_IMAGE")

    page_num = request.form.get("page_number", 1, type=int)
    engine = get_engine()
    result = engine.process_image_page(pil_img, page_number=page_num)
    return jsonify({
        "success": True,
        "page": result
    })
