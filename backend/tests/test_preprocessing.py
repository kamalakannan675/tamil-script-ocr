import pytest
import sys
import os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.ocr.preprocessing import (
    pil_to_cv2,
    cv2_to_pil,
    enhance_contrast,
    binarize_image,
    deskew_image,
    estimate_skew_angle,
    preprocess_page,
)

def test_pil_cv2_conversion():
    pil_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    cv_img = pil_to_cv2(pil_img)
    assert cv_img.shape == (100, 100, 3)
    back_pil = cv2_to_pil(cv_img)
    assert back_pil.size == (100, 100)

def test_enhance_contrast_and_binarization():
    gray = np.full((120, 120), 128, dtype=np.uint8)
    enhanced = enhance_contrast(gray)
    assert enhanced.shape == (120, 120)
    binary = binarize_image(enhanced)
    assert binary.shape == (120, 120)
    assert set(np.unique(binary)).issubset({0, 255})

def test_deskew_image():
    img = np.zeros((100, 100), dtype=np.uint8)
    deskewed = deskew_image(img, 5.0)
    assert deskewed.shape == (100, 100)

def test_preprocess_page_pipeline():
    pil_img = Image.new("RGB", (200, 150), color=(250, 250, 250))
    enhanced_gray, binary, skew_angle = preprocess_page(pil_img)
    assert enhanced_gray.shape == (150, 200)
    assert binary.shape == (150, 200)
    assert isinstance(skew_angle, float)
