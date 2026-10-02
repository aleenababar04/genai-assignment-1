"""Convolutional autoencoder used for image restoration.

Input:  a corrupted image batch of shape (B, 3, 128, 128) with values in [0, 1].
Output: the restored image batch, same shape, values in [0, 1].

The encoder halves the spatial size four times (128 -> 64 -> 32 -> 16 -> 8)
while the number of channels grows. The bottleneck (latent) is a tensor of
shape (B, latent_channels, 8, 8). The decoder doubles the size four times
to get back to 128x128.

Compression: the input has 3 * 128 * 128 = 49152 values. The latent has
latent_channels * 8 * 8 values:

    latent_channels =  16 ->  1024 values (48x smaller than the input)
    latent_channels =  32 ->  2048 values (24x smaller, the default)
    latent_channels =  64 ->  4096 values (12x smaller)
    latent_channels = 128 ->  8192 values ( 6x smaller)

With skip=False (the default) the decoder sees ONLY the latent, so every
piece of information about the image has to pass through the bottleneck.
With skip=True (used only for an ablation) ONE encoder feature map, the
16x16 one, is also passed to the decoder.
"""

import torch
import torch.nn as nn

NUM_DOWNSAMPLES = 4  # each one halves the height and the width


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    """conv3x3 -> BatchNorm -> ReLU. stride=2 halves the spatial size.

    The conv has no bias because BatchNorm adds its own shift right after.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Halve the spatial size (stride-2 conv), then refine with one more conv."""
    return nn.Sequential(
        conv_block(in_channels, out_channels, stride=2),
        conv_block(out_channels, out_channels),
    )


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Double the spatial size (nearest-neighbour upsampling), then two convs."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv_block(in_channels, out_channels),
        conv_block(out_channels, out_channels),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvAutoencoder(nn.Module):
    """Encoder -> compressed bottleneck -> decoder.

    base_channels:   channels after the first conv (called c below).
    latent_channels: channels of the 8x8 bottleneck.
    dropout:         probability of dropping a whole latent channel in training.
    skip:            if True, pass the 16x16 encoder feature to the decoder.
    """

    def __init__(self, base_channels: int = 32, latent_channels: int = 32,
                 dropout: float = 0.0, skip: bool = False):
        super().__init__()
        c = base_channels
        self.latent_channels = latent_channels
        self.skip = skip

        # Encoder.
        self.stem = conv_block(3, c)              # (B,  c, 128, 128)
        self.down1 = down_block(c, c)             # (B,  c,  64,  64)
        self.down2 = down_block(c, 2 * c)         # (B, 2c,  32,  32)
        self.down3 = down_block(2 * c, 4 * c)     # (B, 4c,  16,  16)
        self.down4 = down_block(4 * c, 8 * c)     # (B, 8c,   8,   8)

        # Bottleneck: a 1x1 conv squeezes 8c channels to latent_channels.
        self.to_latent = nn.Conv2d(8 * c, latent_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout)

        # Decoder.
        self.from_latent = conv_block(latent_channels, 8 * c)  # (B, 8c,   8,   8)
        self.up1 = up_block(8 * c, 4 * c)                      # (B, 4c,  16,  16)
        # With the skip, up2 receives decoder 4c + encoder 4c = 8c channels.
        up2_in_channels = 8 * c if skip else 4 * c
        self.up2 = up_block(up2_in_channels, 2 * c)            # (B, 2c,  32,  32)
        self.up3 = up_block(2 * c, c)                          # (B,  c,  64,  64)
        self.up4 = up_block(c, c)                              # (B,  c, 128, 128)
        self.to_image = nn.Conv2d(c, 3, kernel_size=3, padding=1)  # (B, 3, 128, 128)

    def _encode_with_skip(self, x: torch.Tensor):
        """Run the encoder. Returns (latent, 16x16 encoder feature)."""
        h = self.stem(x)        # (B,  c, 128, 128)
        h = self.down1(h)       # (B,  c,  64,  64)
        h = self.down2(h)       # (B, 2c,  32,  32)
        skip_feature = self.down3(h)   # (B, 4c, 16, 16)
        h = self.down4(skip_feature)   # (B, 8c,  8,  8)
        z = self.to_latent(h)   # (B, latent_channels, 8, 8)
        z = self.dropout(z)     # does nothing in eval mode
        return z, skip_feature

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image (B, 3, 128, 128) -> latent (B, latent_channels, 8, 8)."""
        z, _ = self._encode_with_skip(x)
        return z

    def decode(self, z: torch.Tensor, skip_feature: torch.Tensor | None = None) -> torch.Tensor:
        """Latent (B, latent_channels, 8, 8) -> image (B, 3, 128, 128) in [0, 1].

        `skip_feature` is the 16x16 encoder feature. It is required when the
        model was built with skip=True and ignored otherwise.
        """
        if self.skip and skip_feature is None:
            raise ValueError("This model was built with skip=True, so decode() "
                             "also needs the 16x16 encoder feature (skip_feature).")

        h = self.from_latent(z)  # (B, 8c,   8,   8)
        h = self.up1(h)          # (B, 4c,  16,  16)
        if self.skip:
            # Join along the channel dimension: 4c + 4c = 8c channels.
            h = torch.cat([h, skip_feature], dim=1)
        h = self.up2(h)          # (B, 2c,  32,  32)
        h = self.up3(h)          # (B,  c,  64,  64)
        h = self.up4(h)          # (B,  c, 128, 128)
        h = self.to_image(h)     # (B,  3, 128, 128)
        return torch.sigmoid(h)  # squash every value into [0, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Corrupted image -> restored image."""
        z, skip_feature = self._encode_with_skip(x)
        if self.skip:
            return self.decode(z, skip_feature)
        return self.decode(z)

    def latent_size(self, image_size: int = 128) -> int:
        """Number of values in the bottleneck for one square image."""
        side = image_size // (2 ** NUM_DOWNSAMPLES)  # 128 -> 8
        return self.latent_channels * side * side


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
