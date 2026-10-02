"""Training loss for image restoration (Tasks 1-3).

    L = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))

Images are float tensors of shape (B, 3, H, W) with values in [0, 1].
"""

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def restoration_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float):
    """Return (loss, l1, ssim_value), all scalar tensors.

    `l1` is the mean absolute pixel error (lower is better).
    `ssim_value` is the mean structural similarity (1 means identical), so
    `1 - ssim_value` is used as the second error term.
    `alpha` balances the two terms: 1 gives pure L1, 0 gives pure 1 - SSIM.
    """
    l1 = F.l1_loss(pred, target)
    # data_range=1.0 because pixel values lie in [0, 1].
    ssim_value = ssim(pred, target, data_range=1.0, size_average=True)
    loss = alpha * l1 + (1 - alpha) * (1 - ssim_value)
    return loss, l1, ssim_value
