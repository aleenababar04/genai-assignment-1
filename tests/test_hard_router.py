"""Tests for the hard-routing restorer (CPU, tiny stand-in models, no real data)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.autoencoder import ConvAutoencoder
from src.models.hard_router import (EXPERT_NAMES, HardRoutedRestorer, make_restore_fn,
                                    routing_records)

SIZE = 16  # small images keep the fake-model tests fast


class TagClassifier(nn.Module):
    """Fake classifier: reads the class "tag" stored in pixel [0, 0, 0].

    An image whose pixel [0, 0, 0] equals k / 10 is classified as class k.
    """

    def forward(self, x):
        tags = (x[:, 0, 0, 0] * 10).round().long()   # (B,)
        return 5.0 * F.one_hot(tags, num_classes=4).float()


class ConstantExpert(nn.Module):
    """Fake expert k: returns an image filled with k / 10 and counts its calls."""

    def __init__(self, k):
        super().__init__()
        self.value = k / 10
        self.calls = 0
        self.images_seen = 0

    def forward(self, x):
        self.calls += 1
        self.images_seen += len(x)
        return torch.full_like(x, self.value)


def make_router():
    return HardRoutedRestorer(TagClassifier(), ConstantExpert(1), ConstantExpert(2), ConstantExpert(3))


def make_batch(tags, size=SIZE):
    """Random images in [0.5, 1] whose pixel [0, 0, 0] holds the tag k / 10."""
    generator = torch.Generator().manual_seed(0)
    x = 0.5 + 0.5 * torch.rand(len(tags), 3, size, size, generator=generator)
    x[:, 0, 0, 0] = torch.tensor(tags, dtype=torch.float32) / 10
    return x


def expected_output(x, routes):
    """What the router should return for the fake experts."""
    out = x.clone()
    for i, r in enumerate(routes):
        if r != 0:
            out[i] = r / 10
    return out


class FakeDataset:
    """Minimal dataset: (corrupted, clean, label) with label stored as the tag."""

    def __init__(self, tags):
        self.images = make_batch(tags)
        self.entries = [{"label": t} for t in tags]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.images[index], self.images[index], self.entries[index]["label"]


# ---------------------------------------------------------------------------
# Predicted routing
# ---------------------------------------------------------------------------

def test_expert_names():
    assert EXPERT_NAMES == ["identity", "salt_pepper", "blur", "occlusion"]


def test_predicted_mode_uses_classifier_choice():
    router = make_router()
    tags = [0, 1, 2, 3, 2, 0]
    x = make_batch(tags)
    x_hat, routes, probs = router(x)
    assert routes.tolist() == tags
    assert routes.dtype == torch.long
    assert torch.equal(x_hat, expected_output(x, tags))


def test_clean_route_is_bit_identical_identity():
    router = make_router()
    x = make_batch([0, 0, 0])
    x_hat, _, _ = router(x)
    assert torch.equal(x_hat, x)
    assert x_hat is not x  # a copy, not the input itself
    # No expert was touched for an all-clean batch.
    assert all(expert.calls == 0 for expert in router.experts)


def test_expert_not_called_when_nothing_routed_to_it():
    router = make_router()
    router(make_batch([1, 1, 0, 3]))
    salt, blur, occ = router.experts
    assert salt.calls == 1 and salt.images_seen == 2
    assert blur.calls == 0 and blur.images_seen == 0
    assert occ.calls == 1 and occ.images_seen == 1


def test_probs_are_softmax_rows():
    router = make_router()
    _, _, probs = router(make_batch([0, 1, 2, 3]))
    assert probs.shape == (4, 4)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4))
    assert (probs >= 0).all()


def test_route_matches_argmax_of_probs():
    router = make_router()
    routes, probs = router.route(make_batch([3, 2, 1, 0]))
    assert routes.tolist() == [3, 2, 1, 0]
    assert torch.equal(routes, probs.argmax(dim=1))


# ---------------------------------------------------------------------------
# Oracle routing
# ---------------------------------------------------------------------------

def test_oracle_mode_overrides_classifier_but_keeps_probs():
    router = make_router()
    tags = [0, 1, 2, 3]
    oracle = [3, 0, 1, 2]  # disagrees with the classifier everywhere
    x = make_batch(tags)
    x_hat, routes, probs = router(x, torch.tensor(oracle))
    assert routes.tolist() == oracle
    assert torch.equal(x_hat, expected_output(x, oracle))
    # probs still describe the classifier's own opinion.
    assert probs.argmax(dim=1).tolist() == tags


def test_mixed_batch_keeps_original_order():
    router = make_router()
    tags = [3, 0, 1, 3, 2, 0, 1, 2]
    x = make_batch(tags)
    x_hat, _, _ = router(x)
    for i, t in enumerate(tags):
        if t == 0:
            assert torch.equal(x_hat[i], x[i])
        else:
            assert torch.all(x_hat[i] == t / 10)


# ---------------------------------------------------------------------------
# make_restore_fn
# ---------------------------------------------------------------------------

def test_oracle_restore_fn_walks_through_entries():
    router = make_router()
    true_labels = [1, 2, 3, 0, 0, 2, 1]
    entries = [{"label": t} for t in true_labels]
    restore_fn = make_restore_fn(router, "oracle", entries)

    # The classifier would say "clean" for all of them; oracle must ignore that.
    sizes = [3, 1, 3]
    start = 0
    for size in sizes:
        x = make_batch([0] * size)
        out = restore_fn(x)
        expected = expected_output(x, true_labels[start:start + size])
        assert torch.equal(out, expected)
        start += size

    # Running past the end of the entries is an error.
    with pytest.raises(RuntimeError):
        restore_fn(make_batch([0]))


def test_predicted_restore_fn_ignores_entries():
    router = make_router()
    tags = [2, 0, 3]
    x = make_batch(tags)
    wrong_entries = [{"label": 1}] * 3
    out = make_restore_fn(router, "predicted", wrong_entries)(x)
    assert torch.equal(out, expected_output(x, tags))
    out = make_restore_fn(router, "predicted")(x)
    assert torch.equal(out, expected_output(x, tags))


def test_make_restore_fn_bad_arguments():
    router = make_router()
    with pytest.raises(ValueError):
        make_restore_fn(router, "random")
    with pytest.raises(ValueError):
        make_restore_fn(router, "oracle")


# ---------------------------------------------------------------------------
# routing_records
# ---------------------------------------------------------------------------

def test_routing_records_are_aligned():
    tags = [0, 1, 2, 3, 3, 2, 1]
    dataset = FakeDataset(tags)
    records = routing_records(make_router(), dataset, torch.device("cpu"), batch_size=3)
    assert records["true"].shape == (7,)
    assert records["pred"].shape == (7,)
    assert records["probs"].shape == (7, 4)
    assert records["true"].tolist() == tags
    assert records["pred"].tolist() == tags  # the fake classifier is perfect
    assert np.allclose(records["probs"].sum(axis=1), 1.0)


def test_routing_records_find_misrouted_images():
    dataset = FakeDataset([0, 1, 2, 3])
    dataset.entries[2]["label"] = 1  # pretend item 2 is really salt-and-pepper
    records = routing_records(make_router(), dataset, torch.device("cpu"))
    misrouted = np.nonzero(records["true"] != records["pred"])[0]
    assert misrouted.tolist() == [2]


# ---------------------------------------------------------------------------
# Real autoencoders, eval mode and no_grad
# ---------------------------------------------------------------------------

def test_real_autoencoder_experts_shapes_eval_no_grad():
    torch.manual_seed(0)
    experts = [ConvAutoencoder(base_channels=8, latent_channels=8) for _ in range(3)]
    router = HardRoutedRestorer(TagClassifier(), *experts)
    router.eval()
    tags = [0, 1, 2, 3]
    x = make_batch(tags, size=128)
    with torch.no_grad():
        x_hat, routes, probs = router(x)
    assert x_hat.shape == (4, 3, 128, 128)
    assert routes.shape == (4,)
    assert probs.shape == (4, 4)
    assert not x_hat.requires_grad
    assert torch.equal(x_hat[0], x[0])                  # clean bypass
    assert x_hat.min().item() >= 0.0 and x_hat.max().item() <= 1.0
    # Each expert output matches running that expert alone on its image.
    with torch.no_grad():
        for k, expert in enumerate(experts, start=1):
            assert torch.allclose(x_hat[k], expert(x[k:k + 1])[0], atol=1e-6)
