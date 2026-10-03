"""Tests for the soft mixture-of-experts restorer (run on CPU with small models)."""

import math

import pytest
import torch
import torch.nn as nn

from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier
from src.models.soft_moe import (
    BRANCH_NAMES,
    SoftMoERestorer,
    balance_loss,
    moe_loss,
    routing_entropy,
)


def make_batch(batch_size: int = 2) -> torch.Tensor:
    """A random image batch with values in [0, 1]."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(batch_size, 3, 128, 128, generator=generator)


def make_experts():
    torch.manual_seed(0)
    return [ConvAutoencoder(base_channels=8, latent_channels=8) for _ in range(3)]


def make_model(tau: float = 1.0) -> SoftMoERestorer:
    torch.manual_seed(0)
    gate = CorruptionClassifier(channels=(4, 8, 8, 8), dropout=0.0)
    return SoftMoERestorer(gate, *make_experts(), tau=tau)


class FixedGate(nn.Module):
    """Stub gate that returns the same logits for every image."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)  # (B, 4)


def branch_outputs(model: SoftMoERestorer, x: torch.Tensor):
    """Output of each of the four branches, computed separately."""
    with torch.no_grad():
        return [x] + [expert(x) for expert in model.experts]


# ---------------------------------------------------------------------------
# Forward pass and mixing
# ---------------------------------------------------------------------------

def test_branch_names():
    assert BRANCH_NAMES == ["identity", "salt_pepper", "blur", "occlusion"]


def test_output_shapes_and_ranges():
    model = make_model()
    x = make_batch(3)
    x_hat, weights, logits = model(x)
    assert x_hat.shape == (3, 3, 128, 128)
    assert weights.shape == (3, 4)
    assert logits.shape == (3, 4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-6)
    assert x_hat.min().item() >= 0.0
    assert x_hat.max().item() <= 1.0


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_huge_logit_selects_one_branch(k):
    logits = [0.0, 0.0, 0.0, 0.0]
    logits[k] = 1e4
    model = SoftMoERestorer(FixedGate(logits), *make_experts()).eval()
    x = make_batch()
    with torch.no_grad():
        x_hat, weights, _ = model(x)
    expected = branch_outputs(model, x)[k]
    assert torch.allclose(x_hat, expected, atol=1e-5)
    if k == 0:
        assert torch.allclose(x_hat, x, atol=1e-5)  # identity branch returns the input


def test_equal_logits_give_mean_of_branches():
    model = SoftMoERestorer(FixedGate([0.5, 0.5, 0.5, 0.5]), *make_experts()).eval()
    x = make_batch()
    with torch.no_grad():
        x_hat, weights, _ = model(x)
    assert torch.allclose(weights, torch.full((2, 4), 0.25), atol=1e-6)
    expected = torch.stack(branch_outputs(model, x)).mean(dim=0)
    assert torch.allclose(x_hat, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def test_larger_tau_gives_higher_entropy():
    gate = FixedGate([2.0, -1.0, 0.5, 0.0])
    model = SoftMoERestorer(gate, *make_experts()).eval()
    x = make_batch()
    with torch.no_grad():
        model.set_tau(0.5)
        _, sharp, _ = model(x)
        model.set_tau(5.0)
        _, soft, _ = model(x)
    assert (routing_entropy(soft) > routing_entropy(sharp)).all()


def test_tau_in_state_dict_and_set_tau():
    model = make_model(tau=2.0)
    assert "tau" in model.state_dict()
    assert model.state_dict()["tau"].item() == pytest.approx(2.0)
    model.set_tau(0.3)
    assert model.tau.item() == pytest.approx(0.3)
    # tau is a buffer, not a trainable parameter.
    assert all(p is not model.tau for p in model.parameters())
    with pytest.raises(ValueError):
        model.set_tau(0.0)


def test_routing_entropy_limits():
    one_hot = torch.eye(4)
    uniform = torch.full((2, 4), 0.25)
    assert torch.allclose(routing_entropy(one_hot), torch.zeros(4), atol=1e-6)
    assert torch.allclose(routing_entropy(uniform), torch.full((2,), math.log(4)), atol=1e-6)


# ---------------------------------------------------------------------------
# Freezing and parameter groups
# ---------------------------------------------------------------------------

def test_freeze_experts():
    model = make_model()
    model.freeze_experts()
    model.train()

    assert all(not p.requires_grad for p in model.expert_parameters())
    assert all(not expert.training for expert in model.experts)
    assert model.gate.training

    # BatchNorm running stats of a frozen expert must not change in train mode.
    bn = model.experts[0].stem[1]
    running_mean_before = bn.running_mean.clone()

    x = make_batch()
    x_hat, weights, logits = model(x)
    x_hat.mean().backward()

    assert torch.equal(bn.running_mean, running_mean_before)
    assert all(p.grad is None for p in model.expert_parameters())
    gate_grads = [p.grad for p in model.gate_parameters()]
    assert all(g is not None for g in gate_grads)
    assert any(g.abs().sum().item() > 0 for g in gate_grads)


def test_unfreeze_experts():
    model = make_model()
    model.freeze_experts()
    model.train()
    model.unfreeze_experts()

    assert all(p.requires_grad for p in model.expert_parameters())
    assert all(expert.training for expert in model.experts)

    x_hat, _, _ = model(make_batch())
    x_hat.mean().backward()
    assert all(p.grad is not None for p in model.expert_parameters())

    # After unfreezing, model.eval() / model.train() control the experts again.
    model.eval()
    assert all(not expert.training for expert in model.experts)
    model.train()
    assert all(expert.training for expert in model.experts)


def test_parameter_groups_are_disjoint_and_complete():
    model = make_model()
    gate_ids = {id(p) for p in model.gate_parameters()}
    expert_ids = {id(p) for p in model.expert_parameters()}
    all_ids = {id(p) for p in model.parameters()}
    assert gate_ids and expert_ids
    assert gate_ids.isdisjoint(expert_ids)
    assert gate_ids | expert_ids == all_ids


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def test_balance_loss_zero_when_balanced():
    weights = torch.eye(4)  # one image per branch: mean weight is exactly 1/4
    assert balance_loss(weights).item() == pytest.approx(0.0, abs=1e-8)


def test_balance_loss_full_collapse():
    weights = torch.zeros(4, 4)
    weights[:, 2] = 1.0  # every image puts all its weight on branch 2
    value = balance_loss(weights).item()
    assert value > 0
    assert value == pytest.approx(3 * (1 / 4) ** 2 + (3 / 4) ** 2)


def test_moe_loss_scalar_with_grad_and_parts():
    model = make_model()
    x = make_batch(4)
    target = make_batch(4).flip(0)
    labels = torch.tensor([0, 1, 2, 3])
    x_hat, weights, logits = model(x)
    loss, parts = moe_loss(x_hat, target, logits, weights, labels, model.tau,
                           l1_weight=1.0, ssim_weight=0.5, ce_weight=0.1, balance_weight=0.1)
    assert loss.dim() == 0
    assert loss.requires_grad
    assert set(parts) == {"l1", "ssim", "ce", "balance"}
    assert all(isinstance(v, float) for v in parts.values())
    loss.backward()


def test_moe_loss_reduces_to_l1():
    model = make_model()
    x = make_batch(4)
    target = make_batch(4).flip(0)
    labels = torch.tensor([0, 1, 2, 3])
    x_hat, weights, logits = model(x)
    loss, parts = moe_loss(x_hat, target, logits, weights, labels, model.tau,
                           l1_weight=2.0, ssim_weight=0.0, ce_weight=0.0, balance_weight=0.0)
    expected = 2.0 * torch.nn.functional.l1_loss(x_hat, target)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-6)
    assert parts["l1"] == pytest.approx(expected.item() / 2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# ONNX export of the full pipeline
# ---------------------------------------------------------------------------

def export_onnx(model: SoftMoERestorer, x: torch.Tensor, path: str) -> None:
    # dynamo=False selects the legacy (TorchScript) exporter, which does not
    # need the extra onnxscript package.
    torch.onnx.export(
        model, x, path,
        opset_version=17,
        input_names=["input"],
        output_names=["output", "weights", "logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
            "weights": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )


def test_onnx_export_matches_pytorch(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    model = make_model(tau=1.0).eval()
    x = make_batch(2)
    path = str(tmp_path / "soft_moe.onnx")
    export_onnx(model, x, path)
    onnx.checker.check_model(onnx.load(path))

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = [t.numpy() for t in model(x)]
    actual = session.run(["output", "weights", "logits"], {"input": x.numpy()})
    for a, e in zip(actual, expected):
        assert a.shape == e.shape
        assert abs(a - e).max() < 1e-4

    # The batch axis is dynamic: a different batch size must also work.
    out, w, lg = session.run(None, {"input": make_batch(3).numpy()})
    assert out.shape == (3, 3, 128, 128)
    assert w.shape == (3, 4) and lg.shape == (3, 4)

    # tau is baked into the graph: exporting with another tau changes the weights.
    model.set_tau(5.0)
    path_hot = str(tmp_path / "soft_moe_tau5.onnx")
    export_onnx(model, x, path_hot)
    session_hot = ort.InferenceSession(path_hot, providers=["CPUExecutionProvider"])
    weights_hot = session_hot.run(["weights"], {"input": x.numpy()})[0]
    with torch.no_grad():
        expected_hot = model(x)[1].numpy()
    assert abs(weights_hot - expected_hot).max() < 1e-4
    assert abs(weights_hot - actual[1]).max() > 1e-4
