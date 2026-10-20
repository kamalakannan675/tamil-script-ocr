import pytest
import sys
import os
import io

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.exports.txt_export import generate_txt_bytes
from app.exports.docx_export import generate_docx_bytes
from app.exports.pdf_export import generate_pdf_bytes
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

TAMIL_SAMPLE_TEXT = "தமிழ் மிகத் தொன்மையான செம்மொழி.\nஅகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு."

def test_txt_export():
    txt_bytes = generate_txt_bytes(TAMIL_SAMPLE_TEXT)
    decoded = txt_bytes.decode("utf-8")
    assert "தமிழ்" in decoded
    assert "உலகு" in decoded

def test_docx_export():
    docx_bytes = generate_docx_bytes(TAMIL_SAMPLE_TEXT)
    assert len(docx_bytes) > 500  # Valid zip docx container
    assert docx_bytes.startswith(b"PK")  # ZIP magic bytes

def test_pdf_export():
    pdf_bytes = generate_pdf_bytes(TAMIL_SAMPLE_TEXT)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")  # PDF magic header

def test_export_api_endpoints(client):
    # Test TXT
    res_txt = client.post("/api/export", json={"text": TAMIL_SAMPLE_TEXT, "format": "txt", "filename": "test"})
    assert res_txt.status_code == 200
    assert res_txt.headers.get("Content-Type").startswith("text/plain")

    # Test DOCX
    res_docx = client.post("/api/export", json={"text": TAMIL_SAMPLE_TEXT, "format": "docx", "filename": "test"})
    assert res_docx.status_code == 200
    assert "wordprocessingml" in res_docx.headers.get("Content-Type")

    # Test PDF
    res_pdf = client.post("/api/export", json={"text": TAMIL_SAMPLE_TEXT, "format": "pdf", "filename": "test"})
    assert res_pdf.status_code == 200
    assert res_pdf.headers.get("Content-Type") == "application/pdf"

    # Test empty text error
    res_empty = client.post("/api/export", json={"text": "   ", "format": "txt"})
    assert res_empty.status_code == 400

    # Test invalid format error
    res_invalid = client.post("/api/export", json={"text": TAMIL_SAMPLE_TEXT, "format": "exe"})
    assert res_invalid.status_code == 400
