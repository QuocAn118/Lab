
"""Evaluate a completed FashionMNIST run on the full test set.

Usage:
    python src/evaluate.py --run-id run_001

Outputs:
    outputs/<run_id>/metrics.json
    outputs/<run_id>/predictions.csv
"""

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from torch import nn


# Resolve Lab 1 and register source directories.
LAB_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = LAB_ROOT / "src"
DATA_DIR = LAB_ROOT / "data"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DATA_DIR))

from model_io import load_model
from data_pipeline import create_dataloaders


EXPECTED_TEST_SAMPLES = 10_000
NUM_CLASSES = 10


def validate_run_id(run_id):
    """Validate run ID and prevent path traversal."""
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id
    ):
        raise ValueError(f"Invalid run_id: {run_id}")

    return run_id


def check_required_files(run_id):
    """Check required configuration, checkpoint and split files."""
    validate_run_id(run_id)

    config_path = LAB_ROOT / "outputs" / run_id / "config.json"
    checkpoint_path = (
        LAB_ROOT / "checkpoints" / run_id / "final_epoch.pth"
    )
    split_path = LAB_ROOT / "data" / "reports" / "split_indices.json"

    required = [
        config_path,
        checkpoint_path,
        split_path,
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing required files or directories:\n" + details
        )

    return config_path, checkpoint_path


def read_config(config_source):
    """Read and validate an experiment configuration."""
    if isinstance(config_source, dict):
        config = config_source
    else:
        with Path(config_source).open("r", encoding="utf-8") as file:
            config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")

    classes = config.get("classes")

    if not isinstance(classes, list) or len(classes) != NUM_CLASSES:
        raise ValueError(
            "config.classes must contain exactly 10 class names"
        )

    if not all(
        isinstance(name, str) and name.strip()
        for name in classes
    ):
        raise ValueError(
            "Every class name must be a non-empty string"
        )

    if not isinstance(config.get("model"), dict):
        raise ValueError("config.model is missing or invalid")

    if not isinstance(config.get("data"), dict):
        raise ValueError("config.data is missing or invalid")

    return config


def evaluate(run_id):
    """Evaluate the checkpoint on all 10,000 FashionMNIST test samples."""
    run_id = validate_run_id(run_id)

    config_path, checkpoint_path = check_required_files(run_id)
    config = read_config(config_path)

    # Keep the original config unchanged.
    # Remove smoke-test limits for full test evaluation.
    eval_config = copy.deepcopy(config)
    eval_config["data"]["limits"] = None

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"[EVALUATE] Run ID: {run_id}")
    print(f"[EVALUATE] Checkpoint: {checkpoint_path}")
    print(f"[EVALUATE] Device: {device}")

    # Load the trained model.
    model = load_model(run_id, device)

    model.eval()

    # Create data loaders using the full dataset.
    _, _, test_loader = create_dataloaders(eval_config)

    if len(test_loader.dataset) != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, "
            f"got {len(test_loader.dataset)}"
        )

    classes = config["classes"]
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    predictions = []

    # No gradient calculation during evaluation.
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
                raise ValueError(
                    "Expected model output shape (batch_size, 10), "
                    f"got {tuple(logits.shape)}"
                )

            if not torch.isfinite(logits).all().item():
                raise FloatingPointError(
                    "Model produced NaN or Inf logits"
                )

            loss = loss_fn(logits, labels)

            if not math.isfinite(loss.item()) or loss.item() < 0:
                raise FloatingPointError(
                    f"Invalid evaluation loss: {loss.item()}"
                )

            predicted = logits.argmax(dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            num_correct += (predicted == labels).sum().item()

            true_values = labels.cpu().tolist()
            predicted_values = predicted.cpu().tolist()

            for true_label, predicted_label in zip(
                true_values, predicted_values
            ):
                if not 0 <= true_label < NUM_CLASSES:
                    raise ValueError(
                        f"Invalid true label: {true_label}"
                    )

                if not 0 <= predicted_label < NUM_CLASSES:
                    raise ValueError(
                        f"Invalid predicted label: {predicted_label}"
                    )

                predictions.append({
                    "sample_index": num_samples,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "true_name": classes[true_label],
                    "predicted_name": classes[predicted_label],
                })

                num_samples += 1

    # Verify all test samples were evaluated.
    if num_samples != EXPECTED_TEST_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SAMPLES} evaluated samples, "
            f"got {num_samples}"
        )

    test_loss = total_loss / num_samples
    test_accuracy = num_correct / num_samples

    if not math.isfinite(test_loss) or test_loss < 0:
        raise FloatingPointError("Invalid final evaluation loss")

    if not 0 <= test_accuracy <= 1:
        raise FloatingPointError("Invalid final evaluation accuracy")

    metrics = {
        "run_id": run_id,
        "checkpoint": f"checkpoints/{run_id}/final_epoch.pth",
        "split": "test",
        "num_samples": num_samples,
        "num_correct": num_correct,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "device": str(device),
    }

    # Save reports.
    output_dir = LAB_ROOT / "outputs" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.csv"

    metrics_tmp = output_dir / "metrics.json.tmp"
    predictions_tmp = output_dir / "predictions.csv.tmp"

    try:
        with metrics_tmp.open("w", encoding="utf-8") as file:
            json.dump(
                metrics,
                file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            file.write("\n")

        with predictions_tmp.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:
            fieldnames = [
                "sample_index",
                "true_label",
                "predicted_label",
                "true_name",
                "predicted_name",
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(predictions)

        os.replace(metrics_tmp, metrics_path)
        os.replace(predictions_tmp, predictions_path)

    finally:
        metrics_tmp.unlink(missing_ok=True)
        predictions_tmp.unlink(missing_ok=True)

    print(f"[EVALUATE] Samples: {num_samples}")
    print(f"[EVALUATE] Correct: {num_correct}")
    print(f"[EVALUATE] Test loss: {test_loss:.6f}")
    print(f"[EVALUATE] Test accuracy: {test_accuracy:.6f}")
    print(f"[EVALUATE] Metrics: {metrics_path}")
    print(f"[EVALUATE] Predictions: {predictions_path}")

    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a FashionMNIST experiment "
            "on the full test set"
        )
    )

    parser.add_argument(
        "--run-id",
        required=True,
        help="Experiment ID, for example run_001",
    )

    args = parser.parse_args(argv)
    evaluate(args.run_id)


if __name__ == "__main__":
    main()
