"""Tests for the convolutional autoencoder (run on CPU with small models)."""

import pytest
import torch
import torch.nn.functional as F

from src.models.autoencoder import ConvAutoencoder, count_parameters

INPUT_VALUES = 3 * 128 * 128  # number of values in one input image


def make_batch(batch_size: int = 2) -> torch.Tensor:
    """A random image batch with values in [0, 1]."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(batch_size, 3, 128, 128, generator=generator)


@pytest.mark.parametrize("skip", [False, True])
def test_output_shape_range_and_dtype(skip):
    model = ConvAutoencoder(base_channels=8, latent_channels=16, skip=skip)
    out = model(make_batch())
    assert out.shape == (2, 3, 128, 128)
    assert out.dtype == torch.float32
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


@pytest.mark.parametrize("latent_channels", [16, 32, 64])
def test_encode_shape(latent_channels):
    model = ConvAutoencoder(base_channels=8, latent_channels=latent_channels)
    z = model.encode(make_batch())
    assert z.shape == (2, latent_channels, 8, 8)


def test_decode_of_encode_equals_forward():
    """Without the skip, the output depends on the input only through the latent."""
    model = ConvAutoencoder(base_channels=8, latent_channels=16).eval()
    x = make_batch()
    with torch.no_grad():
        assert torch.equal(model.decode(model.encode(x)), model(x))


@pytest.mark.parametrize("latent_channels", [16, 32, 64, 128])
def test_bottleneck_is_compressed(latent_channels):
    model = ConvAutoencoder(base_channels=8, latent_channels=latent_channels)
    assert model.latent_size() == latent_channels * 8 * 8
    assert model.latent_size() < INPUT_VALUES
    # The method must agree with the real latent tensor.
    z = model.encode(make_batch())
    assert z[0].numel() == model.latent_size()


def test_decode_needs_skip_feature_when_skip_is_on():
    model = ConvAutoencoder(base_channels=8, latent_channels=16, skip=True).eval()
    x = make_batch()
    with torch.no_grad():
        z = model.encode(x)
        with pytest.raises(ValueError):
            model.decode(z)
        # With the skip feature, decode gives the same result as forward.
        z, skip_feature = model._encode_with_skip(x)
        assert torch.equal(model.decode(z, skip_feature), model(x))


@pytest.mark.parametrize("skip", [False, True])
def test_gradients_flow(skip):
    model = ConvAutoencoder(base_channels=8, latent_channels=16, skip=skip)
    x = make_batch()
    loss = F.l1_loss(model(x), x)
    loss.backward()

    first_conv = model.stem[0]
    last_conv = model.to_image
    for conv in (first_conv, last_conv):
        assert conv.weight.grad is not None
        assert conv.weight.grad.abs().sum().item() > 0.0


def test_dropout_is_off_in_eval_mode():
    model = ConvAutoencoder(base_channels=8, latent_channels=16, dropout=0.5).eval()
    x = make_batch()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_dropout_is_on_in_train_mode():
    model = ConvAutoencoder(base_channels=8, latent_channels=16, dropout=0.5).train()
    x = make_batch()
    with torch.no_grad():
        assert not torch.equal(model.encode(x), model.encode(x))


def test_parameter_count_grows_with_base_channels():
    small = count_parameters(ConvAutoencoder(base_channels=8))
    medium = count_parameters(ConvAutoencoder(base_channels=16))
    large = count_parameters(ConvAutoencoder(base_channels=32))
    assert 0 < small < medium < large


def test_onnx_export_matches_pytorch(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    model = ConvAutoencoder(base_channels=8, latent_channels=16).eval()
    x = make_batch()
    path = str(tmp_path / "autoencoder.onnx")

    # dynamo=False selects the legacy (TorchScript) exporter, which does not
    # need the extra onnxscript package.
    torch.onnx.export(
        model, x, path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(x).numpy()
    actual = session.run(["output"], {"input": x.numpy()})[0]
    assert abs(actual - expected).max() < 1e-4

    # The batch axis is dynamic: a different batch size must also work.
    one_image = x[:1].numpy()
    assert session.run(["output"], {"input": one_image})[0].shape == (1, 3, 128, 128)
