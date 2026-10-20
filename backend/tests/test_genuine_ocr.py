import pytest
import sys
import os
import io
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from app.ocr.engine import TamilOCREngine
from ml.data.synthetic_generator import render_tamil_line_image

def test_genuine_ocr_pipeline_inference():
    """Verify genuine model inference on an actual Tamil line document."""
    engine = TamilOCREngine()
    status = engine.get_status()
    
    # Verify model loaded
    assert status["available"] is True, f"Model not loaded: {status.get('missing_dependency')}"
    assert status["model_name"] == "TamilHandwrittenCRNN"

    # Generate a realistic Tamil document image
    tamil_phrase = "வணக்கம் தமிழ்"
    pil_img = render_tamil_line_image(tamil_phrase, font_size=24)
    
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Process document
    result = engine.process_document(img_bytes, "test_document.png")

    assert result["success"] is True
    assert result["page_count"] == 1
    assert "total_processing_time_sec" in result
    assert len(result["pages"]) == 1
    
    page = result["pages"][0]
    assert page["line_count"] >= 1
    assert len(page["lines"]) >= 1
    assert "bbox" in page["lines"][0]
    assert "confidence" in page["lines"][0]
    assert isinstance(page["text"], str)

    # Test direct line recognition
    gray_line = np.array(pil_img.convert("L"))
    line_text, conf = engine.recognizer.recognize_line(gray_line)
    assert isinstance(line_text, str)
    assert 0.0 <= conf <= 1.0
