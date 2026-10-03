"""Style-conditioned conditional GAN for face-to-sketch generation (Task 4).

This follows pix2pix (Isola et al., "Image-to-Image Translation with
Conditional Adversarial Networks", CVPR 2017), adapted to 128x128 images
and to an extra condition: the FS2K sketch style s in {0, 1, 2}.

    Generator:      y_hat = G(x, s)         U-Net encoder-decoder with skips
    Discriminator:  D(x, y, s) -> logits    PatchGAN, one logit per image patch

The style is a learned categorical embedding (nn.Embedding). The generator
and the discriminator each have their OWN embedding, and both feed it into
the network as extra input channels, so the style really changes what the
networks compute (it is not just a label attached to the interface).

Losses (binary cross-entropy with logits):

    L_D = 0.5 * [BCE(D(x, y, s), 1) + BCE(D(x, G(x, s), s), 0)]
    L_G = BCE(D(x, G(x, s), s), 1) + lambda_L1 * L1(y, G(x, s))

Value ranges: during training photos and sketches are in [-1, 1] and the
generator ends in tanh. GeneratorForExport wraps a trained generator so that
the ONNX graph used by the application takes and returns values in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_STYLES = 3  # the three FS2K sketch styles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_weights(module: nn.Module) -> None:
    """pix2pix initialisation, applied to every submodule of `module`.

    Conv / ConvTranspose / Embedding weights ~ N(0, 0.02),
    BatchNorm weights ~ N(1, 0.02), all biases = 0.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)


def style_map(embedding: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Broadcast a style vector to an image-sized map.

    (B, style_dim) -> (B, style_dim, height, width): every pixel gets a copy.
    """
    return embedding[:, :, None, None].expand(-1, -1, height, width)


def down_block(in_channels: int, out_channels: int, norm: bool = True) -> nn.Sequential:
    """conv4x4 stride 2 -> [BatchNorm] -> LeakyReLU(0.2). Halves H and W.

    The conv has no bias when BatchNorm follows (BatchNorm adds its own shift).
    InstanceNorm2d is a known alternative to BatchNorm for image translation.
    """
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_channels: int, out_channels: int, dropout: float = 0.0) -> nn.Sequential:
    """transposed conv4x4 stride 2 -> BatchNorm -> ReLU -> [Dropout]. Doubles H and W."""
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class StyleUNetGenerator(nn.Module):
    """pix2pix U-Net generator for 128x128 images, conditioned on a style.

    base_channels: channels of the first encoder layer (called c below).
    style_dim:     size of the learned style embedding.
    dropout:       dropout rate of the three innermost decoder layers
                   (pix2pix uses dropout as the generator's noise source).
    out_channels:  channels of the generated sketch (3 or 1).
    num_styles:    number of style categories.
    """

    def __init__(self, base_channels: int = 64, style_dim: int = 16, dropout: float = 0.5,
                 out_channels: int = 3, num_styles: int = NUM_STYLES):
        super().__init__()
        c = base_channels
        self.style_embedding = nn.Embedding(num_styles, style_dim)

        # Encoder: 7 stride-2 convs, 128 -> 1. The input is photo + style map.
        self.down1 = down_block(3 + style_dim, c, norm=False)  # (B,  c, 64, 64) no norm on the first layer
        self.down2 = down_block(c, 2 * c)                       # (B, 2c, 32, 32)
        self.down3 = down_block(2 * c, 4 * c)                   # (B, 4c, 16, 16)
        self.down4 = down_block(4 * c, 8 * c)                   # (B, 8c,  8,  8)
        self.down5 = down_block(8 * c, 8 * c)                   # (B, 8c,  4,  4)
        self.down6 = down_block(8 * c, 8 * c)                   # (B, 8c,  2,  2)
        # Innermost: NO norm. At 1x1 with batch size 1, BatchNorm would see a
        # single value per channel (variance 0), so pix2pix leaves it out.
        self.down7 = down_block(8 * c, 8 * c, norm=False)       # (B, 8c,  1,  1)

        # Decoder: each up layer (except the first) also receives the matching
        # encoder feature through a skip connection, so its input channels are
        # decoder channels + encoder channels.
        self.up1 = up_block(8 * c + style_dim, 8 * c, dropout)  # (B, 8c,  2,  2) bottleneck + style
        self.up2 = up_block(8 * c + 8 * c, 8 * c, dropout)      # (B, 8c,  4,  4) skip: down6
        self.up3 = up_block(8 * c + 8 * c, 8 * c, dropout)      # (B, 8c,  8,  8) skip: down5
        self.up4 = up_block(8 * c + 8 * c, 4 * c)               # (B, 4c, 16, 16) skip: down4
        self.up5 = up_block(4 * c + 4 * c, 2 * c)               # (B, 2c, 32, 32) skip: down3
        self.up6 = up_block(2 * c + 2 * c, c)                   # (B,  c, 64, 64) skip: down2
        self.to_sketch = nn.ConvTranspose2d(c + c, out_channels, kernel_size=4,
                                            stride=2, padding=1)  # (B, out, 128, 128) skip: down1

        init_weights(self)

    def forward(self, photo: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """photo (B, 3, 128, 128) in [-1, 1], style (B,) long -> sketch (B, out, 128, 128) in [-1, 1]."""
        s = self.style_embedding(style)  # (B, style_dim)

        # The style is injected in TWO places:
        # 1) at the input, as a style map next to the photo, so the very first
        #    layers (which make edges and strokes) can depend on the style;
        # 2) at the 1x1 bottleneck, where the network makes its global decision
        #    about the whole image, so the style also steers that decision.
        x = torch.cat([photo, style_map(s, photo.shape[2], photo.shape[3])], dim=1)  # (B, 3+sd, 128, 128)

        d1 = self.down1(x)    # (B,  c, 64, 64)
        d2 = self.down2(d1)   # (B, 2c, 32, 32)
        d3 = self.down3(d2)   # (B, 4c, 16, 16)
        d4 = self.down4(d3)   # (B, 8c,  8,  8)
        d5 = self.down5(d4)   # (B, 8c,  4,  4)
        d6 = self.down6(d5)   # (B, 8c,  2,  2)
        d7 = self.down7(d6)   # (B, 8c,  1,  1)

        h = torch.cat([d7, s[:, :, None, None]], dim=1)  # (B, 8c+sd, 1, 1)
        h = self.up1(h)                                  # (B, 8c,  2,  2)
        h = self.up2(torch.cat([h, d6], dim=1))          # (B, 8c,  4,  4)
        h = self.up3(torch.cat([h, d5], dim=1))          # (B, 8c,  8,  8)
        h = self.up4(torch.cat([h, d4], dim=1))          # (B, 4c, 16, 16)
        h = self.up5(torch.cat([h, d3], dim=1))          # (B, 2c, 32, 32)
        h = self.up6(torch.cat([h, d2], dim=1))          # (B,  c, 64, 64)
        h = self.to_sketch(torch.cat([h, d1], dim=1))    # (B, out, 128, 128)
        return torch.tanh(h)                             # values in [-1, 1]


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

class PatchDiscriminator(nn.Module):
    """pix2pix 70x70 PatchGAN, conditioned on the photo and the style.

    Input: concat(photo, sketch, style map) along the channels.
    Output: a map of logits (no sigmoid); each logit says whether one patch
    of the photo-sketch pair looks real.

    For a 128x128 input the spatial sizes are
        128 -> 64 (C64, s2) -> 32 (C128, s2) -> 16 (C256, s2)
            -> 15 (C512, s1) -> 14 (logit conv, s1),
    so the output is (B, 1, 14, 14). With a 4x4 kernel and padding 1, a
    stride-1 conv shrinks the size by 1.

    Receptive field of one output logit (walk back from the output,
    rf = rf * stride + (kernel - stride)):
        logit conv (s1): 4;  C512 (s1): 4 + 3 = 7;  C256 (s2): 7*2 + 2 = 16;
        C128 (s2): 16*2 + 2 = 34;  C64 (s2): 34*2 + 2 = 70  -> 70x70 pixels.
    """

    def __init__(self, base_channels: int = 64, style_dim: int = 16, in_photo_channels: int = 3,
                 in_sketch_channels: int = 3, num_styles: int = NUM_STYLES):
        super().__init__()
        c = base_channels
        # The discriminator's OWN style embedding (not shared with G).
        self.style_embedding = nn.Embedding(num_styles, style_dim)
        in_channels = in_photo_channels + in_sketch_channels + style_dim

        self.net = nn.Sequential(
            # C64: no norm on the first layer.
            nn.Conv2d(in_channels, c, kernel_size=4, stride=2, padding=1),           # (B,  c, 64, 64)
            nn.LeakyReLU(0.2, inplace=True),
            # C128
            nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1, bias=False),     # (B, 2c, 32, 32)
            nn.BatchNorm2d(2 * c),
            nn.LeakyReLU(0.2, inplace=True),
            # C256
            nn.Conv2d(2 * c, 4 * c, kernel_size=4, stride=2, padding=1, bias=False), # (B, 4c, 16, 16)
            nn.BatchNorm2d(4 * c),
            nn.LeakyReLU(0.2, inplace=True),
            # C512 (stride 1)
            nn.Conv2d(4 * c, 8 * c, kernel_size=4, stride=1, padding=1, bias=False), # (B, 8c, 15, 15)
            nn.BatchNorm2d(8 * c),
            nn.LeakyReLU(0.2, inplace=True),
            # One logit per patch (stride 1), no sigmoid: BCEWithLogits adds it.
            nn.Conv2d(8 * c, 1, kernel_size=4, stride=1, padding=1),                 # (B,  1, 14, 14)
        )

        init_weights(self)

    def forward(self, photo: torch.Tensor, sketch: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """photo (B, 3, H, W), sketch (B, C, H, W), style (B,) long -> logits (B, 1, H', W')."""
        s = self.style_embedding(style)                      # (B, style_dim)
        s_map = style_map(s, photo.shape[2], photo.shape[3])  # (B, style_dim, H, W)
        x = torch.cat([photo, sketch, s_map], dim=1)         # (B, 3 + C + style_dim, H, W)
        return self.net(x)


# ---------------------------------------------------------------------------
# Export wrapper
# ---------------------------------------------------------------------------

class GeneratorForExport(nn.Module):
    """Wraps a trained generator for ONNX/app use: [0,1] in -> [0,1] out.

    photo (B, 3, 128, 128) float in [0, 1], style (B,) int64 in {0, 1, 2}
    -> sketch (B, C, 128, 128) float in [0, 1].
    """

    def __init__(self, generator: StyleUNetGenerator):
        super().__init__()
        self.generator = generator

    def forward(self, photo: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = photo * 2 - 1                # [0, 1] -> [-1, 1] (training range)
        y = self.generator(x, style)     # [-1, 1] (tanh)
        return (y + 1) / 2               # [-1, 1] -> [0, 1]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    """Return (loss, real_loss, fake_loss), all scalar tensors.

    real_loss = BCE(D(x, y, s), 1), fake_loss = BCE(D(x, G(x, s), s), 0).
    loss = 0.5 * (real_loss + fake_loss): pix2pix halves D's loss so that D
    learns more slowly relative to G.
    """
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    loss = 0.5 * (real_loss + fake_loss)
    return loss, real_loss, fake_loss


def generator_loss(fake_logits: torch.Tensor, fake: torch.Tensor, target: torch.Tensor,
                   l1_weight: float):
    """Return (loss, adv_loss, l1_loss), all scalar tensors.

    adv_loss = BCE(D(x, G(x, s), s), 1): G wants D to call its sketches real.
    l1_loss  = mean |y - G(x, s)|: keeps the sketch close to the real one.
    loss     = adv_loss + l1_weight * l1_loss.
    """
    adv_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    l1_loss = F.l1_loss(fake, target)
    loss = adv_loss + l1_weight * l1_loss
    return loss, adv_loss, l1_loss
