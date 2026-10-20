"""
PyTorch CRNN architecture for line-level Handwritten Tamil Recognition (HTR).
Consists of:
1. Deep CNN feature extractor with batch normalization.
2. Bidirectional LSTM sequence modeling.
3. Linear projection to Tamil vocabulary for CTC loss/decoding.
"""

import torch
import torch.nn as nn
from .unicode_utils import VOCAB_SIZE

class TamilHandwrittenCRNN(nn.Module):
    def __init__(self, num_classes: int = VOCAB_SIZE, hidden_size: int = 256, in_channels: int = 1):
        super(TamilHandwrittenCRNN, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Stage 1: [B, 1, 32, W] -> [B, 64, 16, W/2]
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 2: [B, 64, 16, W/2] -> [B, 128, 8, W/4]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Stage 3: [B, 128, 8, W/4] -> [B, 256, 8, W/4]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Stage 4: [B, 256, 8, W/4] -> [B, 256, 4, W/4]
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Stage 5: [B, 256, 4, W/4] -> [B, 512, 4, W/4]
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Stage 6: [B, 512, 4, W/4] -> [B, 512, 2, W/4]
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            
            # Stage 7: [B, 512, 2, W/4] -> [B, 512, 1, W/4]
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Bidirectional Recurrent Sequence Layers
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.25,
            batch_first=False
        )
        
        # Linear projection to vocabulary
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        Input: [B, 1, 32, W]
        Output: [T, B, num_classes] (Log-probabilities ready for CTC)
        """
        # CNN feature extraction
        conv_out = self.cnn(x)  # [B, 512, 1, W']
        
        # Squeeze height
        conv_out = conv_out.squeeze(2)  # [B, 512, W']
        
        # Permute to time-major: [W', B, 512]
        conv_out = conv_out.permute(2, 0, 1)
        
        # RNN sequence modeling
        rnn_out, _ = self.rnn(conv_out)  # [W', B, 2 * hidden_size]
        
        # Linear projection to classes
        logits = self.fc(rnn_out)  # [W', B, num_classes]
        
        # Log-softmax for CTC loss and decoding
        log_probs = nn.functional.log_softmax(logits, dim=2)
        return log_probs
