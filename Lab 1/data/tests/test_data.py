"""Integration checks on prepared FashionMNIST (run prepare_data.py first).

Spec: docs/requirements/3_1_prepare_data.md
Spec: docs/requirements/3_2_transforms_dataloader.md
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline import create_dataloaders
from train import build_config


def test_full_split_counts_order_and_reproducibility():
    config = build_config("run_001")
    train, val, test = create_dataloaders(config)
    assert [len(x.dataset) for x in (train, val, test)] == [54000, 6000, 10000]
    train_ids, val_ids = set(train.dataset.indices), set(val.dataset.indices)
    assert not train_ids & val_ids
    assert train_ids | val_ids == set(range(60000))
    first = next(iter(train))
    repeated = next(iter(create_dataloaders(config)[0]))
    assert all(torch.equal(a, b) for a, b in zip(first, repeated))
    for loader in (train, val, test):
        count = 0
        targets = []
        for images, labels in loader:
            assert images.dtype == torch.float32 and images.shape[1:] == (1, 28, 28)
            assert images.min() >= 0 and images.max() <= 1
            assert labels.dtype == torch.int64 and labels.min() >= 0 and labels.max() <= 9
            count += len(labels)
            targets.append(labels)
        assert count == len(loader.dataset)
        if loader is test:
            assert torch.equal(torch.cat(targets), test.dataset.targets)
        if loader is val:
            assert torch.equal(torch.cat(targets), val.dataset.dataset.targets[val.dataset.indices])


def test_smoke_config_and_invalid_options():
    config = build_config("smoke_001")
    assert [len(x.dataset) for x in create_dataloaders(config)] == [512, 256, 256]
    for update in ({"batch_size": 0}, {"transforms": ["Normalize"]},
                   {"limits": {"train": 0, "val": 1, "test": 1}}):
        with pytest.raises(ValueError):
            create_dataloaders(update)
