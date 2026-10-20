"""
Image preprocessing for handwritten Tamil documents:
- Grayscale conversion
- Contrast normalization (CLAHE)
- Noise filtering
- Skew detection and deskewing
- Adaptive thresholding
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional

def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR numpy array."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    np_img = np.array(pil_img)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR/Grayscale image to PIL Image."""
    if len(cv_img.shape) == 2:
        return Image.fromarray(cv_img)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def estimate_skew_angle(gray_img: np.ndarray) -> float:
    """
    Estimate skew angle of the page using Hough lines or minAreaRect on text contours.
    Returns angle in degrees (-45 to 45).
    """
    # Threshold image inverted (text as white pixels)
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find all non-zero coordinate points
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 50:
        return 0.0

    # Find the minimum area bounding box
    angle = cv2.minAreaRect(coords)[-1]
    
    # Normalize OpenCV angle convention
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
        
    # Ignore tiny negligible angles
    if abs(angle) < 0.5:
        return 0.0
    # Guard against extreme erroneous rotations
    if abs(angle) > 30:
        return 0.0
        
    return float(angle)

def deskew_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by the given angle around its center to correct skew."""
    if abs(angle) < 0.2:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        img, m, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed

def enhance_contrast(gray_img: np.ndarray) -> np.ndarray:
    """Enhance document contrast using Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_img)

def binarize_image(gray_img: np.ndarray) -> np.ndarray:
    """Otsu binarization for document analysis."""
    # Apply mild Gaussian blur to reduce high-frequency salt-and-pepper noise
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def preprocess_page(
    image: Image.Image,
    apply_deskew: bool = True,
    apply_clahe: bool = True
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Complete full-page preprocessing pipeline.
    Returns:
    - enhanced_gray: High-contrast grayscale image for recognition
    - binary: Clean binarized image for layout/line segmentation
    - skew_angle: Detected skew angle in degrees
    """
    cv_img = pil_to_cv2(image)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    skew_angle = 0.0
    if apply_deskew:
        skew_angle = estimate_skew_angle(gray)
        if abs(skew_angle) >= 0.5:
            gray = deskew_image(gray, skew_angle)
            
    if apply_clahe:
        enhanced_gray = enhance_contrast(gray)
    else:
        enhanced_gray = gray
        
    binary = binarize_image(enhanced_gray)
    return enhanced_gray, binary, skew_angle
