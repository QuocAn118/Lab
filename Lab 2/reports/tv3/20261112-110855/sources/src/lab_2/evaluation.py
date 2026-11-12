"""Bước 7: chọn mốc `best`, đánh giá trên tập test và đo hiệu năng. Đặc tả: docs/evaluation.md."""
import os
import shutil
import statistics
import time

import matplotlib

# `!python` trên Colab thừa hưởng backend inline của notebook, còn script chỉ cần ghi file ảnh.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch import nn

from lab_2.common import (
    argument_parser,
    ci_half_width,
    load_config,
    peak_ram_gb,
    project_path,
    read_report,
    update_report,
)
from lab_2.data import CLASS_NAMES, build_transform, load_splits, make_loader, raw_samples
from lab_2.model import load_checkpoint
from lab_2.training import select_best_run

LATENCY_SAMPLES = 200
LATENCY_WARMUP_RUNS = 5
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D")]


def evaluate_model(model, loader, device):
    model.eval()
    total_loss, predictions, labels = 0.0, [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            if not torch.isfinite(outputs).all():
                raise FloatingPointError("Logits đánh giá không hữu hạn")
            total_loss += nn.functional.cross_entropy(outputs, targets, reduction="sum").item()
            predictions += outputs.argmax(dim=1).tolist()
            labels += targets.tolist()
    if not labels:
        raise ValueError("Loader đánh giá rỗng")
    correct = sum(p == t for p, t in zip(predictions, labels))
    return {
        "loss": total_loss / len(labels),
        "accuracy": correct / len(labels),
        "predictions": predictions,
        "labels": labels,
    }


def kpi_status(value, target, higher_is_better):
    if value is None:
        return "Không đo được"
    met = value >= target if higher_is_better else value <= target
    return "Đạt" if met else "Không đạt"


def completed_runs(config, report):
    """Kết quả của mọi lần chạy trong cấu hình; dừng nếu còn thiếu lần nào."""
    runs = report.get("training", {}).get("runs", {})
    names = [run["name"] for run in config["runs"]]
    missing = [name for name in names if name not in runs]
    if missing:
        raise SystemExit(f"Chưa chạy đủ. Còn thiếu: {missing}")
    return {name: runs[name] for name in names}


def measure_latency_ms(model, transform, samples, device):
    """Tiền xử lý và suy luận từng ảnh. Vài lần đầu chạy nháp vì thường chậm bất thường."""
    if not samples:
        raise ValueError("Cần ảnh để đo latency")
    model.eval()
    def run_once(image):
        with torch.no_grad():
            model(transform(image).unsqueeze(0).to(device))
        if device == "cuda":
            torch.cuda.synchronize()

    for image, _ in samples[:LATENCY_WARMUP_RUNS]:
        run_once(image)
    timings = []
    for image, _ in samples:
        started = time.perf_counter()
        run_once(image)
        timings.append((time.perf_counter() - started) * 1000)
    return {"mean": statistics.mean(timings), "p95": float(np.percentile(timings, 95))}


def style_axis(axis, title, xlabel, ylabel):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, color=INK, loc="left")
    axis.set_xlabel(xlabel, color=MUTED)
    axis.set_ylabel(ylabel, color=MUTED)
    axis.grid(color=GRID, linewidth=1)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)


def plot_line(axis, x, y, color, marker, label):
    axis.plot(
        x, y, color=color, linewidth=2, marker=marker, markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2, label=label,
    )


def plot_loss_curve(run_name, run, path):
    epochs = [entry["epoch"] for entry in run["epochs"]]
    figure, axis = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    style_axis(axis, f"Loss theo epoch của {run_name}", "Epoch", "Loss")
    plot_line(axis, epochs, [e["train_loss"] for e in run["epochs"]], *SERIES[0], "Train loss")
    plot_line(axis, epochs, [e["val_loss"] for e in run["epochs"]], *SERIES[1], "Validation loss")
    axis.set_xticks(epochs)
    max_loss = max(max(e["train_loss"], e["val_loss"]) for e in run["epochs"])
    axis.set_ylim(0, max(0.01, max_loss * 1.1))
    axis.legend(frameon=False, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_val_accuracy(runs, path):
    figure, axis = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    style_axis(axis, "Validation accuracy theo epoch của các lần chạy", "Epoch", "Validation accuracy")
    for (name, run), (color, marker) in zip(runs.items(), SERIES):
        plot_line(
            axis, [e["epoch"] for e in run["epochs"]], [e["val_accuracy"] for e in run["epochs"]],
            color, marker, name,
        )
    axis.xaxis.get_major_locator().set_params(integer=True)
    axis.legend(frameon=False, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix, path):
    positions = range(len(CLASS_NAMES))
    figure, axis = plt.subplots(figsize=(9, 8), facecolor=SURFACE)
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    axis.set_xticks(positions, CLASS_NAMES, color=MUTED, rotation=45, ha="right")
    axis.set_yticks(positions, CLASS_NAMES, color=MUTED)
    axis.set_xlabel("Dự đoán", color=MUTED)
    axis.set_ylabel("Nhãn thật", color=MUTED)
    axis.set_title("Confusion matrix trên tập test", color=INK, loc="left")
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    for row in positions:
        for column in positions:
            value = matrix[row, column]
            axis.text(
                column, row, str(value), ha="center", va="center",
                color="white" if value > matrix.max() / 2 else INK,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def copy_best(config, run_name):
    """Ghi vào file tạm rồi mới thay thế, để lỗi giữa chừng không làm hỏng mốc cũ."""
    checkpoints = project_path(config["paths"]["checkpoints_dir"])
    destination = checkpoints / "best.pth"
    temporary = destination.with_name("best.pth.tmp")
    shutil.copyfile(checkpoints / run_name / "best_val_acc.pth", temporary)
    os.replace(temporary, destination)
    return destination


def comparison_table(runs):
    return [{
        "run": name,
        "purpose": run["purpose"],
        "model": run["model"],
        "learning_rate": run["learning_rate"],
        "batch_size": run["batch_size"],
        "epochs_run": run["epochs_run"],
        "best_epoch": run["best_epoch"],
        "val_accuracy": run["best_val_accuracy"],
        "val_loss": run["best_val_accuracy_loss"],
        "seconds_per_epoch": run["mean_train_seconds_per_epoch"],
    } for name, run in runs.items()]


def main():
    config = load_config(argument_parser(__doc__).parse_args().config)
    report = read_report(config)
    if "evaluation" in report:
        raise SystemExit("Đã có đánh giá test. Giữ nguyên kết quả và dùng thư mục reports/checkpoints mới cho thí nghiệm khác.")
    runs = completed_runs(config, report)
    if any(run.get("config") != config for run in runs.values()):
        raise SystemExit("Config hiện tại khác config đã train. Dùng lại đúng config của cả 4 run.")
    best_name = select_best_run(runs)
    best_run = runs[best_name]

    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    model, info = load_checkpoint(copy_best(config, best_name), device)
    _, _, test = load_splits(config)
    result = evaluate_model(
        model, make_loader(test, config["training"]["eval_batch_size"], config), device
    )
    labels, predictions = result["labels"], result["predictions"]
    class_ids = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(labels, predictions, labels=class_ids)
    macro_f1 = float(f1_score(labels, predictions, labels=class_ids, average="macro", zero_division=0))
    per_class = recall_score(labels, predictions, labels=class_ids, average=None, zero_division=0)
    latency = measure_latency_ms(
        model, build_transform(config, train=False), raw_samples(test, LATENCY_SAMPLES), device
    )

    # Mục tiêu tốc độ đặt cho GPU T4, nên số đo trên CPU không được tính là đạt hay không đạt.
    trained_on_gpu = best_run["device"] != "cpu"
    vram_values = [run["peak_vram_gb"] for run in runs.values() if run["peak_vram_gb"] is not None]
    if gpu:
        vram_values.append(torch.cuda.max_memory_allocated() / 1024**3)
    ram = max(peak_ram_gb(), *(run["peak_ram_gb"] for run in runs.values()))
    kpi = config["kpi"]
    rows = [
        ("Validation Accuracy", best_run["best_val_accuracy"], kpi["accuracy"], True),
        ("Test Accuracy", result["accuracy"], kpi["accuracy"], True),
        ("Test Macro F1", macro_f1, kpi["macro_f1"], True),
        ("Thời gian train (giây/epoch)",
         best_run["mean_train_seconds_per_epoch"] if trained_on_gpu else None,
         kpi["seconds_per_epoch"], False),
        ("Độ trễ suy luận (ms/ảnh)", latency["mean"] if gpu else None, kpi["latency_ms"], False),
        ("VRAM cao nhất (GB)", max(vram_values, default=None), kpi["peak_vram_gb"], False),
        ("RAM cao nhất (GB)", ram, kpi["peak_ram_gb"], False),
    ]
    kpi_table = [
        {"metric": name, "value": value, "target": target,
         "status": kpi_status(value, target, higher)}
        for name, value, target, higher in rows
    ]

    figures = project_path(config["paths"]["reports_dir"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(best_name, best_run, figures / "loss_curve_best.png")
    plot_val_accuracy(runs, figures / "val_accuracy_by_run.png")
    plot_confusion_matrix(matrix, figures / "confusion_matrix.png")

    half_width = ci_half_width(result["accuracy"], len(labels))
    update_report(config, "evaluation", {
        "device": device,
        "best_run": best_name,
        "best_epoch": info["epoch"],
        "test_samples": len(labels),
        "test_metrics": {
            "accuracy": result["accuracy"],
            "accuracy_ci95_half_width": half_width,
            "macro_f1": macro_f1,
            "loss": result["loss"],
        },
        "comparison": comparison_table(runs),
        "kpi": kpi_table,
        "per_class_accuracy": dict(zip(CLASS_NAMES, per_class.tolist())),
        "confusion_matrix": {"labels": CLASS_NAMES, "rows_true_columns_pred": matrix.tolist()},
        "latency_ms": latency,
    })
    print(f"Lần chạy tốt nhất theo validation: {best_name}, epoch {info['epoch']}")
    for row in kpi_table:
        print(f"{row['metric']}: {row['value']} | mục tiêu {row['target']} | {row['status']}")
    print(f"Test Accuracy {result['accuracy']:.3f} ± {half_width:.3f} (khoảng tin cậy 95%)")


if __name__ == "__main__":
    main()
