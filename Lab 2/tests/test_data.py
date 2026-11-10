"""TV1: kiểm thử src/lab_2/data.py. Chạy: python -m pytest tests/test_data.py -q"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import Subset, TensorDataset

from lab_2.data import build_transform, describe, make_loader, raw_samples, split_indices


def test_split_has_requested_sizes_and_no_shared_index():
    train, val = split_indices(100, train_size=70, val_size=20, seed=42)
    assert (len(train), len(val)) == (70, 20)
    assert not set(train) & set(val)


def test_split_is_reproducible_and_depends_on_seed():
    assert split_indices(100, 70, 20, 42) == split_indices(100, 70, 20, 42)
    assert split_indices(100, 70, 20, 42) != split_indices(100, 70, 20, 7)


def test_split_rejects_more_samples_than_available():
    with pytest.raises(ValueError):
        split_indices(100, 90, 20, 42)


def test_test_subset_is_shuffled_not_the_first_images():
    indices, _ = split_indices(100, 10, 0, 42)
    assert len(indices) == 10 and indices != list(range(10))


def test_eval_transform_resizes_and_normalizes_with_imagenet_values(config):
    tensor = build_transform(config, train=False)(Image.new("RGB", (32, 32), (255, 255, 255)))
    assert tensor.shape == (3, 224, 224)
    assert tensor[0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)


def test_train_transform_is_random_but_eval_transform_is_not(config):
    image = Image.effect_noise((32, 32), 80).convert("RGB")
    train, evaluate = build_transform(config, True), build_transform(config, False)
    assert any(not torch.equal(train(image), train(image)) for _ in range(5))
    assert torch.equal(evaluate(image), evaluate(image))


def test_loader_uses_batch_size_and_shuffle_is_reproducible(config):
    config["data"]["num_workers"] = 0
    dataset = TensorDataset(torch.arange(10))
    first = [batch[0].tolist() for batch in make_loader(dataset, 4, config, shuffle=True)]
    second = [batch[0].tolist() for batch in make_loader(dataset, 4, config, shuffle=True)]
    assert first == second and [len(batch) for batch in first] == [4, 4, 2]
    assert [batch[0].tolist() for batch in make_loader(dataset, 10, config)] == [list(range(10))]


def fake_subset():
    source = SimpleNamespace(data=np.zeros((4, 32, 32, 3), dtype=np.uint8), targets=[0, 1, 1, 2])
    return Subset(source, [3, 1, 2])


def test_raw_samples_returns_original_images_with_labels():
    samples = raw_samples(fake_subset(), 2)
    assert [label for _, label in samples] == [2, 1]
    assert samples[0][0].size == (32, 32)


def test_describe_counts_each_class():
    report = describe(fake_subset())
    assert report["samples"] == 3
    assert (report["class_counts"]["automobile"], report["class_counts"]["bird"]) == (2, 1)
    assert report["class_counts"]["cat"] == 0
