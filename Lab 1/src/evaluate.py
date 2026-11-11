"""Đánh giá best trên test và tạo artifact trước khi chốt Run.

Spec: docs/requirements/3_5_evaluate.md
Spec: docs/lab_1_description.md
"""
import csv
import math
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torchvision

from demo import render_samples
from model_io import load_model
from run_tracking import write_json


def evaluate_model(model, loader, device):
    model.eval()
    total_loss, labels, predictions = 0.0, [], []
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            if tuple(logits.shape) != (len(targets), 10) or not torch.isfinite(logits).all():
                raise FloatingPointError("Test logits phải hữu hạn và có shape [B,10]")
            total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            labels.extend(targets.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    if not labels:
        raise ValueError("Test loader rỗng")
    if not math.isfinite(total_loss):
        raise FloatingPointError("Test loss không hữu hạn")
    correct = sum(a == b for a, b in zip(labels, predictions))
    return {"num_samples": len(labels), "num_correct": correct,
            "test_loss": total_loss / len(labels), "test_accuracy": correct / len(labels),
            "labels": labels, "predictions": predictions}


def plot_loss(rows, path):
    if not rows:
        raise ValueError("History rỗng")
    epochs = [int(row["epoch"]) for row in rows]
    acc_epoch = int(max(rows, key=lambda row: float(row["val_accuracy"]))["epoch"])
    loss_epoch = int(min(rows, key=lambda row: float(row["val_loss"]))["epoch"])
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        for key, label in [("train_loss", "Train loss"), ("val_loss", "Validation loss")]:
            ax.plot(epochs, [float(row[key]) for row in rows], marker="o", label=label)
        ax.axvline(acc_epoch, color="green", linestyle=":", label="Best val accuracy")
        ax.axvline(loss_epoch, color="orange", linestyle="--", label="Best val loss")
        ax.set(xlabel="Epoch", ylabel="Cross-entropy loss", title="FashionMNIST MLP")
        ax.set_xticks(epochs)
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def evaluate_run(config, run, test_loader, summary):
    if run.manifest.get("status") != "running":
        raise ValueError("Chỉ đánh giá trong Run đang mở; không sửa run đã chốt")
    device = config["training"]["device"]
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model = load_model(run.run_id, device, checkpoint="best")
    if cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = evaluate_model(model, test_loader, device)
    if cuda:
        torch.cuda.synchronize()
    test_seconds = time.perf_counter() - start
    peak_bytes = max(summary["peak_gpu_bytes"], torch.cuda.max_memory_allocated()) if cuda else None
    classes = config["classes"]
    labels, predictions = result.pop("labels"), result.pop("predictions")
    metrics = {"run_id": run.run_id, "checkpoint": f"checkpoints/{run.run_id}/best.pth",
               "split": "test", "device": device, **result}
    write_json(run.out_dir / "metrics.json", metrics)
    with (run.out_dir / "predictions.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_index", "true_label", "predicted_label", "true_name", "predicted_name"])
        for i, (label, prediction) in enumerate(zip(labels, predictions)):
            writer.writerow([i, label, prediction, classes[label], classes[prediction]])
    per_class = {}
    for i, name in enumerate(classes):
        count = labels.count(i)
        correct = sum(label == i and prediction == i for label, prediction in zip(labels, predictions))
        per_class[name] = correct / count if count else None
    with (run.out_dir / "history.csv").open(encoding="utf-8") as stream:
        history = list(csv.DictReader(stream))
    checkpoint = run.ckpt_dir / "best.pth"
    performance = {
        "accuracy": {"test_accuracy": result["test_accuracy"], "test_loss": result["test_loss"],
                     "best_val_accuracy": summary["best_val_accuracy"],
                     "best_val_accuracy_epoch": summary["best_val_accuracy_epoch"],
                     "best_val_loss": summary["best_val_loss"],
                     "best_val_loss_epoch": summary["best_val_loss_epoch"], "per_class_accuracy": per_class},
        "timing": {"training_started_at": summary["started_at"], "training_finished_at": summary["finished_at"],
                   "train_seconds": summary["train_seconds"],
                   "epoch_seconds": [float(row["epoch_seconds"]) for row in history],
                   "test_seconds": test_seconds, "inference_ms_per_image": test_seconds * 1000 / len(labels)},
        "resources": {"device": device, "device_name": torch.cuda.get_device_name() if cuda else platform.processor(),
                      "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
                      "num_parameters": sum(p.numel() for p in model.parameters()),
                      "checkpoint_bytes": checkpoint.stat().st_size,
                      "checkpoint_mb": checkpoint.stat().st_size / 1_000_000,
                      "peak_gpu_memory_mb": peak_bytes / 1_000_000 if cuda else None},
    }
    write_json(run.out_dir / "performance.json", performance)
    plot_loss(history, run.out_dir / "loss_curve.png")
    render_samples(model, test_loader.dataset, classes, run.out_dir / "prediction_samples.png")
    run.note(f"best test accuracy {result['test_accuracy']:.6f} on {result['num_samples']} samples")
    return metrics
