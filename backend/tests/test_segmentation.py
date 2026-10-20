import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.ocr.segmentation import segment_text_lines, LineSegment

def test_segment_text_lines():
    # Synthetic page with 2 black horizontal text bars on white background
    h, w = 300, 400
    gray = np.full((h, w), 255, dtype=np.uint8)
    binary = np.full((h, w), 255, dtype=np.uint8)

    # Line 1
    gray[50:80, 40:350] = 0
    binary[50:80, 40:350] = 0

    # Line 2
    gray[140:170, 40:350] = 0
    binary[140:170, 40:350] = 0

    lines = segment_text_lines(gray, binary, min_line_height=10)
    assert len(lines) == 2
    
    crop1, seg1 = lines[0]
    crop2, seg2 = lines[1]
    
    assert seg1.line_index == 1
    assert seg2.line_index == 2
    assert seg1.y_min < seg2.y_min  # Reading order: top to bottom
    assert crop1.shape[0] > 0
    assert crop1.shape[1] > 0
