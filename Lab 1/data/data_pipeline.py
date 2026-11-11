"""Reproducible train/validation/test loaders for FashionMNIST.

Spec: docs/requirements/3_2_transforms_dataloader.md
Spec: docs/lab_1_description.md
Spec: docs/team_plan_integrity.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from run_tracking import LAB_ROOT, record_artifact, write_json

SPECS = ["docs/requirements/3_2_transforms_dataloader.md", "docs/lab_1_description.md",
         "docs/team_plan_integrity.md"]


def split_indices():
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).tolist()
    return {"seed": 42, "train": order[:54000], "val": order[54000:]}


def create_dataloaders(config):
    """Accept flat data config or the nested config produced by train.build_config."""
    data = config.get("data", config)
    seed = config.get("training", {}).get("seed", config.get("seed", 42))
    batch_size = data.get("batch_size", 64)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if data.get("transforms", ["ToTensor"]) != ["ToTensor"]:
        raise ValueError("Only transforms=['ToTensor'] is supported")
    path = LAB_ROOT / "data" / "reports" / "split_indices.json"
    if not path.is_file():
        raise FileNotFoundError("Run python data/prepare_data.py first (missing split_indices.json)")
    indices = json.loads(path.read_text(encoding="utf-8"))
    if indices != split_indices():
        raise ValueError("Split indices do not match the fixed 54000/6000 split with seed 42")
    try:
        original = datasets.FashionMNIST(str(LAB_ROOT / "data"), train=True,
                                         transform=transforms.ToTensor(), download=False)
        test = datasets.FashionMNIST(str(LAB_ROOT / "data"), train=False,
                                     transform=transforms.ToTensor(), download=False)
    except RuntimeError as exc:
        raise RuntimeError("Missing/invalid data; run python data/prepare_data.py first") from exc
    if len(original) != 60000 or len(test) != 10000:
        raise ValueError("Unexpected FashionMNIST dataset size")
    splits = {"train": Subset(original, indices["train"]),
              "val": Subset(original, indices["val"]), "test": test}
    limits = data.get("limits")
    if limits is not None:
        if set(limits) != set(splits):
            raise ValueError("limits must contain train, val and test")
        for name, dataset in splits.items():
            limit = limits[name]
            if type(limit) is not int or not 1 <= limit <= len(dataset):
                raise ValueError(f"Invalid {name} limit: {limit}")
            splits[name] = Subset(dataset, range(limit))
    return tuple(DataLoader(dataset, batch_size=batch_size, shuffle=name == "train",
                            generator=torch.Generator().manual_seed(seed),
                            drop_last=False, num_workers=0)
                 for name, dataset in splits.items())


def check_batch():
    loaders = create_dataloaders({"batch_size": 64, "seed": 42, "transforms": ["ToTensor"]})
    images, labels = next(iter(loaders[0]))
    if images.shape != (64, 1, 28, 28) or images.dtype != torch.float32:
        raise ValueError("Invalid image batch")
    if labels.shape != (64,) or labels.dtype != torch.int64 or not ((labels >= 0) & (labels <= 9)).all():
        raise ValueError("Invalid label batch")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("Invalid pixel values")
    report = {"batch_size_requested": 64, "image_shape": list(images.shape),
              "image_dtype": str(images.dtype), "label_shape": list(labels.shape),
              "label_dtype": str(labels.dtype), "pixel_min": images.min().item(),
              "pixel_max": images.max().item()}
    path = LAB_ROOT / "data" / "reports" / "batch_check.json"
    write_json(path, report)
    record_artifact(path, specs=SPECS, inputs=[LAB_ROOT / "data" / "reports" / "split_indices.json"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args()
    print(check_batch())
