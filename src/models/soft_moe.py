"""Soft mixture-of-experts restoration system (Task 3).

Task 2 picked ONE branch with an argmax, which cannot be differentiated.
Here a gating network G gives every branch a weight and the output is a
weighted average of ALL branches:

    w = softmax(G(x) / tau),   w = [w0, w1, w2, w3]

    x_hat = w0 * x                  (identity / clean branch)
          + w1 * A_salt(x)
          + w2 * A_blur(x)
          + w3 * A_occlusion(x)

tau is the temperature: a small tau makes w close to one-hot (almost hard
routing), a large tau spreads the weight over the branches.

Training loss:

    L_MoE = l1 * L1 + ls * (1 - SSIM) + lc * L_CE + lb * L_balance
    L_balance = sum_k (wbar_k - 1/4)^2,  wbar_k = mean weight of branch k in the batch

G starts from the trained Task 2 classifier and the experts from the
trained Task 2 specialists (the training script loads them).

Images are float tensors of shape (B, 3, H, W) with values in [0, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim

# Name of each branch, in the same order as the routing weights
# (index = condition label: 0 clean, 1 salt_pepper, 2 blur, 3 occlusion).
BRANCH_NAMES = ["identity", "salt_pepper", "blur", "occlusion"]


class SoftMoERestorer(nn.Module):
    """Gate + three specialist autoencoders + identity branch, mixed softly.

    gate:             any module mapping (B, 3, H, W) -> logits (B, 4).
    salt_expert:      restores salt-and-pepper images (branch 1).
    blur_expert:      restores blurred images (branch 2).
    occlusion_expert: restores occluded images (branch 3).
    tau:              softmax temperature (must be > 0).
    Branch 0 (clean) has no network: it is the input itself.
    """

    def __init__(self, gate: nn.Module, salt_expert: nn.Module, blur_expert: nn.Module,
                 occlusion_expert: nn.Module, tau: float = 1.0):
        super().__init__()
        self.gate = gate
        # ModuleList position k-1 holds the expert for branch k (1, 2, 3).
        self.experts = nn.ModuleList([salt_expert, blur_expert, occlusion_expert])
        # A buffer (not a parameter): it is saved in the state dict and baked
        # into the ONNX graph as a constant, but the optimiser never changes it.
        self.register_buffer("tau", torch.tensor(1.0))
        self.set_tau(tau)
        self.experts_frozen = False

    def set_tau(self, value: float) -> None:
        """Change the softmax temperature (in place, so the buffer is kept)."""
        if value <= 0:
            raise ValueError(f"tau must be positive, got {value}.")
        with torch.no_grad():
            self.tau.fill_(float(value))

    def gate_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Raw gate scores G(x): (B, 3, H, W) -> (B, 4)."""
        return self.gate(x)

    def forward(self, x: torch.Tensor):
        """Returns (x_hat (B, 3, H, W), weights (B, 4), logits (B, 4))."""
        logits = self.gate_logits(x)                       # (B, 4)
        weights = torch.softmax(logits / self.tau, dim=1)  # (B, 4), each row sums to 1

        # Every branch processes the WHOLE batch (needed for the gradient of
        # every weight, and keeps the ONNX graph free of data-dependent ifs).
        branches = [x] + [expert(x) for expert in self.experts]  # 4 x (B, 3, H, W)
        stacked = torch.stack(branches, dim=1)                   # (B, 4, 3, H, W)

        # Weighted sum over the branch axis. weights[:, :, None, None, None]
        # has shape (B, 4, 1, 1, 1) so it broadcasts over channels and pixels.
        # No clamp is needed: each expert ends in a sigmoid and the identity
        # branch is the input, so every branch is in [0, 1]; the weights are
        # >= 0 and sum to 1 (a convex combination), so x_hat is in [0, 1] too.
        x_hat = (weights[:, :, None, None, None] * stacked).sum(dim=1)  # (B, 3, H, W)
        return x_hat, weights, logits

    # ------------------------------------------------------------------
    # Freezing (warm-up: only the gate trains)
    # ------------------------------------------------------------------

    def freeze_experts(self) -> None:
        """Stop the experts from learning: no gradients, BatchNorm stats fixed."""
        for p in self.experts.parameters():
            p.requires_grad = False
        # eval mode: BatchNorm uses (and does not update) its running stats.
        self.experts.eval()
        self.experts_frozen = True

    def unfreeze_experts(self) -> None:
        """Let the experts learn again (joint fine-tuning)."""
        for p in self.experts.parameters():
            p.requires_grad = True
        self.experts_frozen = False
        # Follow the mode of the whole model again.
        self.experts.train(self.training)

    def train(self, mode: bool = True):
        """Like nn.Module.train, but frozen experts always stay in eval mode.

        nn.Module.train(True) switches EVERY submodule to train mode, which
        would let frozen experts update their BatchNorm running statistics.
        """
        super().train(mode)
        if self.experts_frozen:
            self.experts.eval()
        return self

    # ------------------------------------------------------------------
    # Parameter groups (separate learning rates in the optimiser)
    # ------------------------------------------------------------------

    def gate_parameters(self):
        """Parameters of the gating network."""
        return self.gate.parameters()

    def expert_parameters(self):
        """Parameters of the three experts."""
        return self.experts.parameters()


# ---------------------------------------------------------------------------
# Losses and analysis helpers
# ---------------------------------------------------------------------------

def balance_loss(weights: torch.Tensor) -> torch.Tensor:
    """sum_k (mean_b weights[b, k] - 1/K)^2 for weights of shape (B, K).

    Zero when, on average over the batch, every branch gets 1/K of the
    weight. It grows when the gate collapses onto a few branches.
    """
    num_branches = weights.shape[1]
    mean_weights = weights.mean(dim=0)                      # (K,) wbar_k
    return ((mean_weights - 1.0 / num_branches) ** 2).sum()


def moe_loss(x_hat: torch.Tensor, target: torch.Tensor, logits: torch.Tensor,
             weights: torch.Tensor, labels: torch.Tensor, tau,
             l1_weight: float, ssim_weight: float, ce_weight: float, balance_weight: float):
    """Return (loss, parts). `loss` is a scalar tensor, `parts` a dict of floats.

    loss = l1_weight * L1 + ssim_weight * (1 - SSIM)
         + ce_weight * CE + balance_weight * L_balance
    """
    l1 = F.l1_loss(x_hat, target)
    # data_range=1.0 because pixel values lie in [0, 1].
    ssim_value = ssim(x_hat, target, data_range=1.0, size_average=True)
    # Cross-entropy on logits / tau, i.e. on exactly the distribution that
    # gives the routing weights. This pushes the ACTUAL weight of the true
    # branch (the known corruption label) towards 1, not just the untempered
    # classifier probabilities.
    ce = F.cross_entropy(logits / tau, labels)
    balance = balance_loss(weights)

    loss = (l1_weight * l1 + ssim_weight * (1 - ssim_value)
            + ce_weight * ce + balance_weight * balance)
    parts = {
        "l1": l1.item(),
        "ssim": ssim_value.item(),
        "ce": ce.item(),
        "balance": balance.item(),
    }
    return loss, parts


def routing_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Entropy (natural log) of each image's routing weights: (B, K) -> (B,).

    0 means all weight on one branch; log(K) means equal weights.
    """
    # clamp avoids log(0) = -inf; 0 * log(tiny) is still 0.
    return -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
