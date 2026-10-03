"""Tests for the style-conditioned pix2pix cGAN (run on CPU with small models)."""

import pytest
import torch

from src.models.autoencoder import count_parameters
from src.models.cgan import (
    NUM_STYLES,
    GeneratorForExport,
    PatchDiscriminator,
    StyleUNetGenerator,
    discriminator_loss,
    generator_loss,
)

C = 8  # small base_channels keeps the tests fast


def make_photo(batch_size: int = 2, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """A random photo batch with values in [low, high]."""
    generator = torch.Generator().manual_seed(0)
    return low + (high - low) * torch.rand(batch_size, 3, 128, 128, generator=generator)


def make_style(batch_size: int = 2, value: int | None = None) -> torch.Tensor:
    """Style labels (B,) int64; all equal to `value`, or 0, 1, 2, 0, ... if None."""
    if value is not None:
        return torch.full((batch_size,), value, dtype=torch.long)
    return torch.arange(batch_size, dtype=torch.long) % NUM_STYLES


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("out_channels", [3, 1])
def test_generator_shape_and_range(out_channels):
    G = StyleUNetGenerator(base_channels=C, out_channels=out_channels)
    out = G(make_photo(), make_style())
    assert out.shape == (2, out_channels, 128, 128)
    assert out.dtype == torch.float32
    assert out.min().item() >= -1.0
    assert out.max().item() <= 1.0


def test_generator_style_changes_output():
    G = StyleUNetGenerator(base_channels=C).eval()  # eval: dropout off, deterministic
    photo = make_photo()
    with torch.no_grad():
        out0 = G(photo, make_style(value=0))
        out0_again = G(photo, make_style(value=0))
        out2 = G(photo, make_style(value=2))
    assert torch.equal(out0, out0_again)
    assert (out0 - out2).abs().max().item() > 0


def test_generator_embedding_weights_change_output():
    G = StyleUNetGenerator(base_channels=C).eval()
    photo, style = make_photo(), make_style(value=1)
    with torch.no_grad():
        before = G(photo, style)
        G.style_embedding.weight.add_(1.0)  # change ONLY the embedding
        after = G(photo, style)
    assert (before - after).abs().max().item() > 0


def test_generator_batch_size_one_in_train_mode():
    G = StyleUNetGenerator(base_channels=C).train()
    out = G(make_photo(1), make_style(1))
    assert out.shape == (1, 3, 128, 128)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

def test_discriminator_output_shape():
    D = PatchDiscriminator(base_channels=C)
    logits = D(make_photo(), make_photo(), make_style())
    assert logits.shape == (2, 1, 14, 14)  # documented in PatchDiscriminator


def test_discriminator_one_channel_sketch():
    D = PatchDiscriminator(base_channels=C, in_sketch_channels=1)
    logits = D(make_photo(), make_photo()[:, :1], make_style())
    assert logits.shape == (2, 1, 14, 14)


def test_discriminator_style_changes_logits():
    D = PatchDiscriminator(base_channels=C).eval()
    photo, sketch = make_photo(), make_photo()
    with torch.no_grad():
        logits0 = D(photo, sketch, make_style(value=0))
        logits2 = D(photo, sketch, make_style(value=2))
    assert (logits0 - logits2).abs().max().item() > 0


def test_discriminator_embedding_is_separate():
    G = StyleUNetGenerator(base_channels=C).eval()
    D = PatchDiscriminator(base_channels=C)
    assert G.style_embedding is not D.style_embedding
    assert G.style_embedding.weight.data_ptr() != D.style_embedding.weight.data_ptr()

    photo, style = make_photo(), make_style()
    with torch.no_grad():
        before = G(photo, style)
        D.style_embedding.weight.add_(1.0)  # must not affect G
        after = G(photo, style)
    assert torch.equal(before, after)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

def test_generator_loss_gradients_reach_generator():
    G = StyleUNetGenerator(base_channels=C)
    D = PatchDiscriminator(base_channels=C)
    photo, target, style = make_photo(), make_photo(), make_style()

    fake = G(photo, style)
    loss, _, _ = generator_loss(D(photo, fake, style), fake, target, l1_weight=100.0)
    loss.backward()

    assert G.style_embedding.weight.grad is not None
    assert G.style_embedding.weight.grad.abs().sum().item() > 0
    for name, p in G.named_parameters():
        assert p.grad is not None, name


def test_discriminator_loss_gradients_reach_discriminator():
    G = StyleUNetGenerator(base_channels=C)
    D = PatchDiscriminator(base_channels=C)
    photo, real, style = make_photo(), make_photo(), make_style()

    with torch.no_grad():
        fake = G(photo, style)
    loss, _, _ = discriminator_loss(D(photo, real, style), D(photo, fake, style))
    loss.backward()

    assert D.style_embedding.weight.grad is not None
    assert D.style_embedding.weight.grad.abs().sum().item() > 0
    for name, p in D.named_parameters():
        assert p.grad is not None, name
    # fake was computed without gradients, so G gets none.
    assert all(p.grad is None for p in G.parameters())


# ---------------------------------------------------------------------------
# Loss values
# ---------------------------------------------------------------------------

def test_discriminator_loss_values():
    real_logits = torch.full((2, 1, 14, 14), 10.0)
    fake_logits = torch.full((2, 1, 14, 14), -10.0)
    loss, real_loss, fake_loss = discriminator_loss(real_logits, fake_logits)
    assert loss.item() < 1e-3
    assert torch.allclose(loss, 0.5 * (real_loss + fake_loss))

    # Swap: real called fake and fake called real -> both parts large.
    loss_bad, real_bad, fake_bad = discriminator_loss(fake_logits, real_logits)
    assert real_bad.item() > 5 and fake_bad.item() > 5
    assert loss_bad.item() > 5

    # The two parts are reported separately.
    _, real_only, fake_only = discriminator_loss(real_logits, real_logits)
    assert real_only.item() < 1e-3
    assert fake_only.item() > 5


def test_generator_loss_values():
    fake_logits = torch.randn(2, 1, 14, 14)
    fake, target = make_photo(), -make_photo()

    loss, adv, l1 = generator_loss(fake_logits, fake, target, l1_weight=0.0)
    assert torch.allclose(loss, adv)
    assert l1.item() > 0

    loss, adv, l1 = generator_loss(fake_logits, fake, fake.clone(), l1_weight=100.0)
    assert l1.item() == 0.0
    assert torch.allclose(loss, adv)

    loss, adv, l1 = generator_loss(fake_logits, fake, target, l1_weight=100.0)
    assert torch.allclose(loss, adv + 100.0 * l1)


# ---------------------------------------------------------------------------
# Export wrapper and ONNX
# ---------------------------------------------------------------------------

def test_export_wrapper_range_and_mapping():
    G = StyleUNetGenerator(base_channels=C).eval()
    wrapper = GeneratorForExport(G).eval()
    photo01, style = make_photo(low=0.0, high=1.0), make_style()
    with torch.no_grad():
        out = wrapper(photo01, style)
        expected = (G(photo01 * 2 - 1, style) + 1) / 2
    assert out.shape == (2, 3, 128, 128)
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0
    assert torch.allclose(out, expected)


def test_onnx_export_matches_pytorch(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    wrapper = GeneratorForExport(StyleUNetGenerator(base_channels=C)).eval()
    photo, style = make_photo(2, low=0.0, high=1.0), make_style()
    path = str(tmp_path / "generator.onnx")

    # dynamo=False selects the legacy (TorchScript) exporter.
    torch.onnx.export(
        wrapper, (photo, style), path,
        opset_version=17,
        input_names=["photo", "style"],
        output_names=["sketch"],
        dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}, "sketch": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    photo3 = make_photo(3, low=0.0, high=1.0)  # another batch size checks the dynamic axis
    for value in range(NUM_STYLES):
        style3 = make_style(3, value=value)
        onnx_out = session.run(["sketch"], {"photo": photo3.numpy().astype("float32"),
                                            "style": style3.numpy().astype("int64")})[0]
        with torch.no_grad():
            torch_out = wrapper(photo3, style3).numpy()
        assert onnx_out.shape == (3, 3, 128, 128)
        assert abs(onnx_out - torch_out).max() < 1e-4


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------

def test_parameter_counts(capsys):
    counts = {}
    for c in (32, 64):
        counts[c] = (count_parameters(StyleUNetGenerator(base_channels=c, style_dim=16)),
                     count_parameters(PatchDiscriminator(base_channels=c, style_dim=16)))
    with capsys.disabled():
        for c, (g, d) in counts.items():
            print(f"\nbase_channels={c}, style_dim=16: G={g:,} D={d:,}")
    assert counts[32][0] < counts[64][0]
    assert counts[32][1] < counts[64][1]
