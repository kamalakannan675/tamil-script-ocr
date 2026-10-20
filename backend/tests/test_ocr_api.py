import pytest
import sys
import os
import io
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_ocr_status_endpoint(client):
    res = client.get("/api/ocr/status")
    assert res.status_code == 200
    data = res.get_json()
    assert "available" in data
    assert "supported_formats" in data
    assert "device" in data
    assert "jpg" in data["supported_formats"]
    assert "pdf" in data["supported_formats"]

def test_ocr_missing_file(client):
    res = client.post("/api/ocr")
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert data["code"] == "MISSING_FILE"

def test_ocr_empty_file(client):
    data = {"file": (io.BytesIO(b""), "")}
    res = client.post("/api/ocr", data=data, content_type="multipart/form-data")
    assert res.status_code == 400

def test_ocr_unsupported_format(client):
    data = {"file": (io.BytesIO(b"fake content"), "script.exe")}
    res = client.post("/api/ocr", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    data_json = res.get_json()
    assert data_json["code"] == "UNSUPPORTED_FORMAT"

def test_ocr_corrupted_image(client):
    data = {"file": (io.BytesIO(b"not a valid image format"), "corrupt.jpg")}
    res = client.post("/api/ocr", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    data_json = res.get_json()
    assert data_json["code"] == "INVALID_IMAGE"
