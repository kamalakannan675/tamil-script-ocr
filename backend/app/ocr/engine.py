"""
Master Tamil OCR Engine:
Coordinates document loading, PDF rendering, page-level preprocessing,
text-line segmentation, neural recognition, and Unicode text reconstruction.
"""

import os
import time
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from .preprocessing import preprocess_page, cv2_to_pil
from .segmentation import segment_text_lines
from .recognition import BaseRecognizer, CRNNTamilRecognizer, VisionEncoderDecoderTamilRecognizer
from ..core.config import Config
from ..core.errors import APIError

class TamilOCREngine:
    def __init__(self, config: Config = Config):
        self.config = config
        self.recognizer: Optional[BaseRecognizer] = None
        self._init_recognizer()

    def _init_recognizer(self):
        """Initialize the recognition model adapter from configured path."""
        # Check primary and alternative checkpoint paths
        path_to_try = None
        if os.path.exists(self.config.CRNN_MODEL_PATH):
            path_to_try = self.config.CRNN_MODEL_PATH
        elif os.path.exists(self.config.ALT_CRNN_PATH):
            path_to_try = self.config.ALT_CRNN_PATH

        self.recognizer = CRNNTamilRecognizer(
            model_path=path_to_try,
            device=self.config.DEVICE
        )

    def load_model_checkpoint(self, checkpoint_path: str) -> bool:
        """Dynamically load or switch checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        if isinstance(self.recognizer, CRNNTamilRecognizer):
            return self.recognizer.load_checkpoint(checkpoint_path)
        return False

    def get_status(self) -> Dict[str, Any]:
        """Return genuine status of the OCR engine and model."""
        if not self.recognizer:
            return {
                "available": False,
                "model_name": None,
                "loaded": False,
                "device": self.config.DEVICE,
                "missing_dependency": "Recognizer not initialized",
                "supported_formats": list(self.config.ALLOWED_EXTENSIONS)
            }

        info = self.recognizer.get_info()
        is_loaded = self.recognizer.is_loaded()
        
        return {
            "available": is_loaded,
            "model_name": info.get("name"),
            "architecture": info.get("architecture"),
            "framework": info.get("framework"),
            "loaded": is_loaded,
            "device": info.get("device"),
            "checkpoint_path": info.get("checkpoint_path"),
            "missing_dependency": None if is_loaded else "Trained model checkpoint (.pt) missing or not loaded",
            "supported_formats": sorted(list(self.config.ALLOWED_EXTENSIONS)),
            "max_pdf_pages": self.config.MAX_PDF_PAGES,
            "max_file_size_mb": round(self.config.MAX_CONTENT_LENGTH / (1024 * 1024), 1)
        }

    def load_pdf_pages(self, pdf_bytes: bytes, max_pages: int = 20) -> List[Image.Image]:
        """Render PDF pages to PIL Images using pypdfium2."""
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise APIError("pypdfium2 is required for PDF rendering.", status_code=500, code="DEPENDENCY_MISSING")

        try:
            pdf = pdfium.PdfDocument(pdf_bytes)
            num_pages = len(pdf)
            
            if num_pages == 0:
                raise APIError("The provided PDF file contains no pages.", status_code=400, code="EMPTY_PDF")
            if num_pages > max_pages:
                raise APIError(
                    f"PDF page count ({num_pages}) exceeds allowed limit of {max_pages} pages.",
                    status_code=400,
                    code="PDF_TOO_LONG"
                )

            images = []
            for i in range(num_pages):
                page = pdf[i]
                # Render at 200 DPI (scale ~ 2.77) for high-quality OCR without excessive memory
                bitmap = page.render(scale=2.77)
                pil_image = bitmap.to_pil()
                images.append(pil_image)
            return images
        except APIError:
            raise
        except Exception as e:
            raise APIError(f"Failed to render PDF document: {e}", status_code=400, code="INVALID_PDF")

    def process_image_page(self, pil_image: Image.Image, page_number: int = 1) -> Dict[str, Any]:
        """
        Process a single image page:
        1. Validate dimensions
        2. Preprocess & deskew
        3. Segment text lines
        4. Recognize text with neural model
        5. Reconstruct Tamil Unicode
        """
        if not self.recognizer or not self.recognizer.is_loaded():
            raise APIError(
                "OCR recognition model is not loaded. Cannot process page.",
                status_code=503,
                code="MODEL_UNAVAILABLE"
            )

        w, h = pil_image.size
        if w > self.config.MAX_IMAGE_DIMENSION or h > self.config.MAX_IMAGE_DIMENSION:
            raise APIError(
                f"Image dimensions ({w}x{h}) exceed maximum allowed dimension ({self.config.MAX_IMAGE_DIMENSION}px).",
                status_code=400,
                code="IMAGE_TOO_LARGE"
            )

        start_time = time.time()
        
        # Preprocessing
        enhanced_gray, binary, skew_angle = preprocess_page(pil_image)

        # Line segmentation
        segmented_lines = segment_text_lines(enhanced_gray, binary)
        
        # Recognition
        line_results = []
        recognized_lines_text = []

        for line_crop, seg_info in segmented_lines:
            text, conf = self.recognizer.recognize_line(line_crop)
            clean_text = text.strip()
            
            line_data = {
                "line_index": seg_info.line_index,
                "bbox": [int(seg_info.x_min), int(seg_info.y_min), int(seg_info.width), int(seg_info.height)],
                "text": clean_text,
                "confidence": conf
            }
            line_results.append(line_data)
            if clean_text:
                recognized_lines_text.append(clean_text)

        elapsed = round(time.time() - start_time, 3)
        page_text = "\n".join(recognized_lines_text)

        return {
            "page_number": page_number,
            "width": w,
            "height": h,
            "skew_angle": round(skew_angle, 2),
            "line_count": len(line_results),
            "text": page_text,
            "lines": line_results,
            "processing_time_sec": elapsed
        }

    def process_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Full document processor for images and PDFs.
        Returns combined Tamil Unicode text and page-by-page results.
        """
        start_total = time.time()
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        
        if ext not in self.config.ALLOWED_EXTENSIONS:
            raise APIError(
                f"Unsupported file format '.{ext}'. Supported formats: {', '.join(sorted(self.config.ALLOWED_EXTENSIONS))}",
                status_code=400,
                code="UNSUPPORTED_FORMAT"
            )

        pages_data = []
        if ext == "pdf":
            pil_pages = self.load_pdf_pages(file_bytes, max_pages=self.config.MAX_PDF_PAGES)
            for idx, page_img in enumerate(pil_pages):
                page_res = self.process_image_page(page_img, page_number=idx + 1)
                pages_data.append(page_res)
        else:
            try:
                import io
                pil_img = Image.open(io.BytesIO(file_bytes))
                pil_img.verify()
                # Reopen after verify
                pil_img = Image.open(io.BytesIO(file_bytes))
            except Exception as e:
                raise APIError(f"Corrupted or invalid image file: {e}", status_code=400, code="INVALID_IMAGE")

            page_res = self.process_image_page(pil_img, page_number=1)
            pages_data.append(page_res)

        # Combine text in reading order across pages
        full_text = "\n\n".join(p["text"] for p in pages_data if p["text"]).strip()
        total_time = round(time.time() - start_total, 3)

        return {
            "success": True,
            "filename": filename,
            "page_count": len(pages_data),
            "total_processing_time_sec": total_time,
            "full_text": full_text,
            "pages": pages_data
        }
