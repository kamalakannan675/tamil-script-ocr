"""
Modular recognition interfaces and adapters for Handwritten Tamil OCR.
Provides:
- BaseRecognizer interface
- CRNNTamilRecognizer (PyTorch CNN-LSTM-CTC)
- VisionEncoderDecoderTamilRecognizer (HuggingFace ViT + Tamil-RoBERTa adapter)
"""

import abc
import os
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict, Any
from .crnn_model import TamilHandwrittenCRNN
from .unicode_utils import ctc_decode_greedy, VOCAB_SIZE, normalize_tamil_text

class BaseRecognizer(abc.ABC):
    @abc.abstractmethod
    def is_loaded(self) -> bool:
        """Check if model weights are loaded and ready."""
        pass

    @abc.abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return model metadata, architecture, device, and status."""
        pass

    @abc.abstractmethod
    def recognize_line(self, line_img: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a single cropped text-line image.
        Returns (decoded_tamil_unicode_text, confidence_score).
        """
        pass

    def recognize_lines(self, line_imgs: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Recognize multiple text lines."""
        return [self.recognize_line(img) for img in line_imgs]

class CRNNTamilRecognizer(BaseRecognizer):
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model: Optional[TamilHandwrittenCRNN] = None
        self._is_loaded = False
        self.target_height = 32
        self.max_width = 768

        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load trained weights into CRNN model."""
        try:
            model = TamilHandwrittenCRNN(num_classes=VOCAB_SIZE)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Allow checkpoint to be either raw state_dict or dict with 'model_state_dict'
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.model = model
            self.model_path = checkpoint_path
            self._is_loaded = True
            return True
        except Exception as e:
            self._is_loaded = False
            self.model = None
            raise RuntimeError(f"Failed to load CRNN checkpoint from {checkpoint_path}: {e}")

    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "TamilHandwrittenCRNN",
            "architecture": "CNN (7-layer) + BiLSTM (2-layer) + CTC",
            "framework": "PyTorch",
            "device": str(self.device),
            "loaded": self._is_loaded,
            "vocab_size": VOCAB_SIZE,
            "checkpoint_path": self.model_path if self._is_loaded else None
        }

    def preprocess_line_image(self, img: np.ndarray) -> torch.Tensor:
        """
        Resize line image to height 32 preserving aspect ratio,
        normalize to [0, 1], invert if necessary (dark text on white bg).
        Returns tensor: [1, 1, 32, W]
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        h, w = img.shape
        if h == 0 or w == 0:
            # Degenerate crop: return a blank canvas
            canvas = np.ones((self.target_height, 64), dtype=np.float32)
            return torch.tensor(canvas).unsqueeze(0).unsqueeze(0).to(self.device)

        # Scale to target height
        scale = self.target_height / float(h)
        new_w = max(16, min(int(w * scale), self.max_width))
        resized = cv2.resize(img, (new_w, self.target_height), interpolation=cv2.INTER_AREA)

        # Normalize pixel values
        norm_img = resized.astype(np.float32) / 255.0
        
        # Check text polarity: if text is white on black, invert so background is ~1.0
        if np.mean(norm_img) < 0.3:
            norm_img = 1.0 - norm_img

        # Convert to float tensor [1, 1, H, W]
        tensor = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def recognize_line(self, line_img: np.ndarray) -> Tuple[str, float]:
        if not self.is_loaded():
            raise RuntimeError("CRNN model is not loaded. Cannot perform inference.")

        with torch.no_grad():
            x = self.preprocess_line_image(line_img)
            # Forward pass: [T, 1, num_classes]
            log_probs = self.model(x)
            
            # Softmax probabilities
            probs = torch.exp(log_probs).squeeze(1)  # [T, num_classes]
            max_probs, pred_indices = torch.max(probs, dim=1)
            
            indices_list = pred_indices.cpu().numpy().tolist()
            text = ctc_decode_greedy(indices_list)
            
            # Confidence calculation (mean of non-blank probabilities)
            non_blank_probs = [
                float(max_probs[i]) for i, idx in enumerate(indices_list) if idx != 0
            ]
            confidence = float(np.mean(non_blank_probs)) if non_blank_probs else 0.95
            
            return text, round(confidence, 4)

class VisionEncoderDecoderTamilRecognizer(BaseRecognizer):
    """
    Adapter for HuggingFace's VisionEncoderDecoder model
    (e.g. 'sabaridsnfuji/Tamil_Offline_Handwritten_OCR').
    """
    def __init__(self, model_name_or_path: str = "sabaridsnfuji/Tamil_Offline_Handwritten_OCR", device: str = "cpu"):
        self.device = torch.device(device)
        self.model_name_or_path = model_name_or_path
        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self) -> bool:
        try:
            from transformers import AutoFeatureExtractor, AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
            from PIL import Image

            encoder_model = "google/vit-base-patch16-224-in21k"
            decoder_model = "d42kw01f/Tamil-RoBERTa"
            
            feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_model)
            tokenizer = AutoTokenizer.from_pretrained(decoder_model)
            self.processor = TrOCRProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            return True
        except Exception as e:
            self._is_loaded = False
            raise RuntimeError(f"Failed to load VisionEncoderDecoder model: {e}")

    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "VisionEncoderDecoderTamilRecognizer",
            "architecture": "ViT-base + Tamil-RoBERTa",
            "framework": "Transformers / PyTorch",
            "device": str(self.device),
            "loaded": self._is_loaded,
            "model_path": self.model_name_or_path
        }

    def recognize_line(self, line_img: np.ndarray) -> Tuple[str, float]:
        if not self.is_loaded():
            raise RuntimeError("VisionEncoderDecoder model is not loaded.")
        from PIL import Image
        pil_img = Image.fromarray(line_img).convert("RGB")
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_length=128)
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        return normalize_tamil_text(text), 0.90
