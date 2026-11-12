"""TV1: kiểm thử src/lab_2/data.py. Chạy: python -m pytest tests/test_data.py -q"""
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch.utils.data import Subset, TensorDataset

from lab_2.data import build_transform, describe, make_loader, raw_samples, split_indices
from lab_2 import data


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


@pytest.mark.parametrize("sizes", [(-1, 0, 0), (10, -1, 2), (10, 2, -1),
                                  (10, 1.5, 2), (10, True, 2), ("10", 1, 2)])
def test_split_rejects_invalid_sizes(sizes):
    with pytest.raises(ValueError):
        split_indices(*sizes, seed=42)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_raw_samples_rejects_invalid_count(count):
    with pytest.raises(ValueError):
        raw_samples(fake_subset(), count)


def test_empty_splits_and_sample_limits():
    assert split_indices(0, 0, 0, 42) == ([], [])
    assert raw_samples(fake_subset(), 0) == []
    assert len(raw_samples(fake_subset(), 100)) == 3


def test_load_splits_and_main_use_correct_sources_and_preserve_report(config, tmp_path, monkeypatch, capsys):
    class FakeCIFAR10:
        def __init__(self, root, train, download, transform):
            self.root, self.train, self.download, self.transform = root, train, download, transform
            size = 100 if train else 20
            self.data = np.zeros((size, 32, 32, 3), dtype=np.uint8)
            self.targets = [i % 10 for i in range(size)]

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return self.transform(Image.fromarray(self.data[index])), self.targets[index]

    monkeypatch.setattr(data.datasets, "CIFAR10", FakeCIFAR10)
    config["data"].update(data_dir=str(tmp_path / "data"), train_size=7, val_size=2, test_size=2, num_workers=0)
    config["paths"]["reports_dir"] = str(tmp_path / "reports")
    train, val, test = data.load_splits(config)
    assert [len(subset) for subset in (train, val, test)] == [7, 2, 2]
    assert train.dataset.train and val.dataset.train and not test.dataset.train
    assert train.dataset is not val.dataset
    assert all(subset.dataset.download for subset in (train, val, test))
    assert all(subset.dataset.root == tmp_path / "data" for subset in (train, val, test))
    assert not set(train.indices) & set(val.indices)
    assert all(0 <= i < len(subset.dataset) for subset in (train, val, test) for i in subset.indices)
    assert all(subset[0][0].shape == (3, 224, 224) for subset in (train, val, test))
    assert [subset.indices for subset in data.load_splits(config)] == [train.indices, val.indices, test.indices]
    assert all(sum(describe(subset)["class_counts"].values()) == len(subset) for subset in (train, val, test))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report_path = tmp_path / "reports/metrics.json"
    report_path.parent.mkdir()
    report_path.write_text('{"model": {"preserved": true}}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["lab_2.data", "--config", str(path)])
    data.main()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model"] == {"preserved": True}
    assert report["data"] == {name: describe(subset) for name, subset in [("train", train), ("val", val), ("test", test)]}
    assert "Shape ảnh sau tiền xử lý: (3, 224, 224)" in capsys.readouterr().out
