import io
from flask import Blueprint, request, send_file, jsonify
from ..exports.txt_export import generate_txt_bytes
from ..exports.docx_export import generate_docx_bytes
from ..exports.pdf_export import generate_pdf_bytes
from ..core.errors import APIError

export_bp = Blueprint("export", __name__)

@export_bp.route("/export", methods=["POST"])
def export_text():
    """
    Export current edited Tamil text in TXT, DOCX, or PDF format.
    Accepts JSON:
    {
      "text": "...",
      "format": "txt" | "docx" | "pdf",
      "filename": "optional_name"
    }
    """
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    fmt = data.get("format", "txt").lower().strip()
    filename_base = data.get("filename", "tamil_script_ocr_export").strip()

    if not text.strip():
        raise APIError("Cannot export empty text.", status_code=400, code="EMPTY_TEXT")

    # Sanitize filename
    clean_filename = "".join(c for c in filename_base if c.isalnum() or c in ("-", "_")).strip() or "tamil_ocr"

    if fmt == "txt":
        file_bytes = generate_txt_bytes(text)
        mimetype = "text/plain; charset=utf-8"
        download_name = f"{clean_filename}.txt"
    elif fmt == "docx":
        file_bytes = generate_docx_bytes(text, title=f"TamilScript OCR - {clean_filename}")
        mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        download_name = f"{clean_filename}.docx"
    elif fmt == "pdf":
        file_bytes = generate_pdf_bytes(text, title=f"TamilScript OCR - {clean_filename}")
        mimetype = "application/pdf"
        download_name = f"{clean_filename}.pdf"
    else:
        raise APIError(
            f"Unsupported export format '{fmt}'. Supported formats are: txt, docx, pdf",
            status_code=400,
            code="INVALID_FORMAT"
        )

    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype,
        as_attachment=True,
        download_name=download_name
    )
