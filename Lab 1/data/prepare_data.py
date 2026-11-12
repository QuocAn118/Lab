"""Download and inspect the original FashionMNIST dataset.

Spec: docs/requirements/3_1_prepare_data.md
Spec: docs/lab_1_description.md
Spec: docs/team_plan_integrity.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

from run_tracking import LAB_ROOT, record_artifact, write_json

SPECS = ["docs/requirements/3_1_prepare_data.md", "docs/lab_1_description.md",
         "docs/team_plan_integrity.md"]
RAW_FILES = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]


def prepare_data():
    root, outputs = LAB_ROOT / "data", LAB_ROOT / "data" / "reports"
    root.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    train = FashionMNIST(root=str(root), train=True, download=True)
    test = FashionMNIST(root=str(root), train=False, download=True)
    for dataset, expected in ((train, 60000), (test, 10000)):
        if len(dataset) != expected or tuple(dataset.data.shape) != (expected, 28, 28):
            raise ValueError("Unexpected FashionMNIST image count or shape")
        if len(dataset.classes) != 10 or set(dataset.targets.tolist()) != set(range(10)):
            raise ValueError("Unexpected FashionMNIST labels")
        if dataset.classes != train.classes:
            raise ValueError("Train/test class mappings differ")
    raw = [root / "FashionMNIST" / "raw" / name for name in RAW_FILES]
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw}
    from data_pipeline import split_indices
    indices = split_indices()
    summary = {"dataset": "FashionMNIST", "train_samples": len(train),
               "test_samples": len(test), "num_classes": 10, "image_shape": [28, 28],
               "classes": train.classes, "sha256": hashes,
               "split_samples": {key: len(value) for key, value in indices.items()
                                 if key in ("train", "val")},
               "overlap_train_val": len(set(indices["train"]) & set(indices["val"]))}
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        image, label = train[i]
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(train.classes[label])
        ax.axis("off")
    fig.tight_layout()
    try:
        fig.savefig(outputs / "data_samples.png")
    finally:
        plt.close(fig)
    write_json(outputs / "split_indices.json", indices)
    write_json(outputs / "data_summary.json", summary)
    for name in ("data_summary.json", "data_samples.png", "split_indices.json"):
        record_artifact(outputs / name, specs=SPECS, inputs=raw)
    return summary


if __name__ == "__main__":
    print(prepare_data())
