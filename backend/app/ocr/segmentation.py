"""
Page layout analysis and text-line segmentation for handwritten Tamil documents.
Preserves upper and lower Tamil diacritics (pulli dots, vowel loops, consonant conjuncts).
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any

class LineSegment:
    def __init__(self, y_min: int, y_max: int, x_min: int, x_max: int, line_index: int):
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.line_index = line_index

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_index": self.line_index,
            "bbox": [int(self.x_min), int(self.y_min), int(self.width), int(self.height)],
        }

def segment_text_lines(
    gray_img: np.ndarray,
    binary_img: np.ndarray,
    min_line_height: int = 14,
    pad_y: int = 4
) -> List[Tuple[np.ndarray, LineSegment]]:
    """
    Segment the page into individual text lines using horizontal projection
    and connected component clustering, preserving Tamil diacritics.
    
    Returns:
    List of (line_crop_ndarray, LineSegment) in reading order.
    """
    h, w = gray_img.shape[:2]
    
    # Invert binary image: text pixels = 255, background = 0
    inv_bin = cv2.bitwise_not(binary_img)
    
    # Horizontally dilate to connect words within each line into a continuous ribbon
    # Kernel width is proportional to document width to span inter-word gaps
    kernel_w = max(15, int(w * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
    dilated = cv2.dilate(inv_bin, kernel, iterations=1)
    
    # Horizontal projection profile: sum across columns
    proj = np.sum(dilated, axis=1) / 255.0
    
    # Threshold for text line presence
    line_threshold = np.max(proj) * 0.04
    
    in_line = False
    start_y = 0
    candidate_ranges = []
    
    for y in range(h):
        val = proj[y]
        if val > line_threshold and not in_line:
            in_line = True
            start_y = y
        elif val <= line_threshold and in_line:
            in_line = False
            end_y = y
            if (end_y - start_y) >= min_line_height:
                candidate_ranges.append((start_y, end_y))
                
    if in_line and (h - start_y) >= min_line_height:
        candidate_ranges.append((start_y, h))
        
    # If projection profile found no distinct lines (e.g. single line or dense block),
    # fallback to bounding box of all text contours
    if not candidate_ranges:
        coords = np.column_stack(np.where(inv_bin > 0))
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            candidate_ranges.append((max(0, y_min - 5), min(h, y_max + 5)))
        else:
            # Fallback to whole page
            candidate_ranges.append((0, h))

    # Refine horizontal boundaries (x_min, x_max) and extract crops
    segmented_lines = []
    for idx, (y0, y1) in enumerate(candidate_ranges):
        # Expand Y slightly to capture tall pulli dots and low vowel loops
        y0_padded = max(0, y0 - pad_y)
        y1_padded = min(h, y1 + pad_y)
        
        line_strip_bin = inv_bin[y0_padded:y1_padded, :]
        x_coords = np.where(line_strip_bin > 0)[1]
        
        if len(x_coords) > 0:
            x0 = max(0, int(np.min(x_coords)) - 5)
            x1 = min(w, int(np.max(x_coords)) + 5)
        else:
            x0 = 0
            x1 = w
            
        if (x1 - x0) < 15 or (y1_padded - y0_padded) < 10:
            continue
            
        line_crop = gray_img[y0_padded:y1_padded, x0:x1]
        segment_info = LineSegment(
            y_min=y0_padded,
            y_max=y1_padded,
            x_min=x0,
            x_max=x1,
            line_index=idx + 1
        )
        segmented_lines.append((line_crop, segment_info))
        
    # If no lines passed the filter, return the central crop
    if not segmented_lines:
        line_crop = gray_img
        segment_info = LineSegment(0, h, 0, w, 1)
        segmented_lines.append((line_crop, segment_info))
        
    return segmented_lines
