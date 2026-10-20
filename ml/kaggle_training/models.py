"""
Candidate Model Architectures for Tamil Handwritten OCR.
Candidate A: Production-compatible 7-layer CNN + 2-layer BiLSTM + CTC (TamilHandwrittenCRNN)
Candidate B: Deeper ResNet-18-style Residual Feature Extractor + 2-layer BiLSTM + CTC (ResNetCRNN)
"""

import torch
import torch.nn as nn
from .config import VOCAB_SIZE

class TamilHandwrittenCRNN(nn.Module):
    """
    Candidate A: Exact match with production architecture.
    7-layer CNN + 2-layer BiLSTM (512 hidden) + CTC projection.
    Parameters: ~8.23 Million
    """
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
        
        # Sequence Modeling
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
        Input: [B, 1, 32, W]
        Output: [T, B, num_classes] (Log-probabilities ready for PyTorch CTCLoss)
        """
        conv_out = self.cnn(x)              # [B, 512, 1, W']
        conv_out = conv_out.squeeze(2)      # [B, 512, W']
        conv_out = conv_out.permute(2, 0, 1)# [W', B, 512]
        
        rnn_out, _ = self.rnn(conv_out)     # [W', B, 2 * hidden_size]
        logits = self.fc(rnn_out)           # [W', B, num_classes]
        return nn.functional.log_softmax(logits, dim=2)


class ResidualBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride=(1, 1)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class ResNetCRNN(nn.Module):
    """
    Candidate B: Deeper ResNet-style backbone with residual skip-connections
    and bidirectional LSTM for complex handwriting recognition.
    """
    def __init__(self, num_classes: int = VOCAB_SIZE, hidden_size: int = 256, in_channels: int = 1):
        super(ResNetCRNN, self).__init__()
        self.num_classes = num_classes
        
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # [B, 64, 16, W/2]
        )
        
        # Residual stages
        self.layer1 = ResidualBlock(64, 128, stride=(2, 2))   # [B, 128, 8, W/4]
        self.layer2 = ResidualBlock(128, 256, stride=(2, 1))  # [B, 256, 4, W/4]
        self.layer3 = ResidualBlock(256, 512, stride=(2, 1))  # [B, 512, 2, W/4]
        self.layer4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # [B, 512, 1, W/4]
        
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=False
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.in_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = out.squeeze(2)          # [B, 512, W']
        out = out.permute(2, 0, 1)    # [W', B, 512]
        
        rnn_out, _ = self.rnn(out)
        logits = self.fc(rnn_out)
        return nn.functional.log_softmax(logits, dim=2)


def get_model(candidate_name: str = "crnn", num_classes: int = VOCAB_SIZE) -> nn.Module:
    """Factory method to instantiate candidate architectures."""
    if candidate_name.lower() in ("crnn", "candidate_a"):
        return TamilHandwrittenCRNN(num_classes=num_classes)
    elif candidate_name.lower() in ("resnet", "resnet_crnn", "candidate_b"):
        return ResNetCRNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown candidate model: {candidate_name}. Choose 'crnn' or 'resnet_crnn'.")
