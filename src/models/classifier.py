"""Convolutional classifier that predicts the input condition of an image.

Input:  an image batch of shape (B, 3, 128, 128) with values in [0, 1].
Output: raw scores (logits) of shape (B, 4), one per condition:
        0 clean, 1 salt-and-pepper, 2 blur, 3 occlusion.

No softmax is applied inside the model:
  - nn.CrossEntropyLoss expects logits (it applies log-softmax itself);
  - in Task 3 the same network initialises the gating network of the
    mixture-of-experts, which applies its own softmax to these logits.

Four blocks each halve the spatial size: 128 -> 64 -> 32 -> 16 -> 8.
Global average pooling then turns the 8x8 feature map into one vector.
"""

import torch
import torch.nn as nn

from src.models.autoencoder import count_parameters  # noqa: F401  (re-exported for convenience)

# Channels of the four blocks for each model size.
CHANNEL_PRESETS = {
    "small": (16, 32, 64, 128),
    "medium": (32, 64, 128, 256),
    "large": (48, 96, 192, 384),
}


def classifier_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Two (conv3x3 -> BatchNorm -> ReLU) layers, then a 2x2 max-pool.

    The convs keep the spatial size; the max-pool halves it.
    The convs have no bias because BatchNorm adds its own shift right after.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CorruptionClassifier(nn.Module):
    """Four conv blocks -> global average pool -> dropout -> linear layer.

    channels:    output channels of the four blocks (see CHANNEL_PRESETS).
    dropout:     probability of dropping a feature before the last layer.
    num_classes: number of conditions to predict.
    """

    def __init__(self, channels=(32, 64, 128, 256), dropout: float = 0.3, num_classes: int = 4):
        super().__init__()
        c1, c2, c3, c4 = channels

        # The first block works at the full 128x128 resolution before any
        # downsampling. Fine detail (single noisy pixels, slightly soft edges
        # from a mild blur) is what separates the classes, and pooling first
        # would throw part of it away.
        self.block1 = classifier_block(3, c1)    # (B, c1, 64, 64)
        self.block2 = classifier_block(c1, c2)   # (B, c2, 32, 32)
        self.block3 = classifier_block(c2, c3)   # (B, c3, 16, 16)
        self.block4 = classifier_block(c3, c4)   # (B, c4,  8,  8)

        self.pool = nn.AdaptiveAvgPool2d(1)      # (B, c4, 1, 1): mean over the 8x8 grid
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c4, num_classes)     # (B, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image batch (B, 3, 128, 128) -> logits (B, num_classes)."""
        h = self.block1(x)          # (B, c1, 64, 64)
        h = self.block2(h)          # (B, c2, 32, 32)
        h = self.block3(h)          # (B, c3, 16, 16)
        h = self.block4(h)          # (B, c4,  8,  8)
        h = self.pool(h)            # (B, c4,  1,  1)
        h = torch.flatten(h, 1)     # (B, c4)
        h = self.dropout(h)         # does nothing in eval mode
        return self.fc(h)           # (B, num_classes), raw logits
