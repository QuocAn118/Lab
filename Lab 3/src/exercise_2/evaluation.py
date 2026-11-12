# Sources: docs/evaluation.md; docs/lab_3_description.md; docs/08_implementation.md §10
"""Exercise 2, bước 7: đánh giá mốc `best` trên tập test và đo hiệu năng. Đặc tả: docs/evaluation.md."""
import functools
import statistics
import tempfile
import time

import evaluate
import matplotlib

# `!python` trên Colab thừa hưởng backend inline của notebook, còn script chỉ cần ghi file ảnh.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from exercise_2.common import (
    ci_half_width,
    config_from_args,
    peak_ram_gb,
    project_path,
    read_report,
    update_report,
)
from exercise_2.data import clean_text, load_splits, tokenize_dataset

LATENCY_WARMUP_RUNS = 5
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"


@functools.cache
def accuracy_metric():
    return evaluate.load("accuracy")


def compute_classification_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_metric().compute(predictions=preds, references=labels)["accuracy"],
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_classification_metrics(np.argmax(logits, axis=-1), labels)


def kpi_status(value, target, higher_is_better):
    if value is None:
        return "Không đo được"
    met = value >= target if higher_is_better else value <= target
    return "Đạt" if met else "Không đạt"


def measure_latency_ms(model, tokenizer, texts, max_length, device):
    """Thời gian tokenize và suy luận từng mẫu. Vài lần đầu chạy nháp vì thường chậm bất thường."""
    def run_once(text):
        inputs = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
        if device == "cuda":
            torch.cuda.synchronize()

    for text in texts[:LATENCY_WARMUP_RUNS]:
        run_once(text)
    timings = []
    for text in texts:
        started = time.perf_counter()
        run_once(text)
        timings.append((time.perf_counter() - started) * 1000)
    return {"mean": statistics.mean(timings), "p95": float(np.percentile(timings, 95))}


def plot_loss_curve(log_history, path):
    train = [(entry["step"], entry["loss"]) for entry in log_history if "loss" in entry]
    val = [(entry["step"], entry["eval_loss"]) for entry in log_history if "eval_loss" in entry]
    figure, axis = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)
    if train:
        axis.plot(*zip(*train), color=TRAIN_COLOR, linewidth=2, label="Train loss")
    axis.plot(
        *zip(*val), color=VAL_COLOR, linewidth=2, marker="o", markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2, label="Validation loss",
    )
    axis.set_title("Loss theo số bước huấn luyện", color=INK, loc="left")
    axis.set_xlabel("Bước", color=MUTED)
    axis.set_ylabel("Loss", color=MUTED)
    axis.set_ylim(bottom=0)
    axis.grid(color=GRID, linewidth=1)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.legend(frameon=False, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_confusion_matrix(matrix, path):
    labels = ["NEGATIVE", "POSITIVE"]
    figure, axis = plt.subplots(figsize=(5, 4.5), facecolor=SURFACE)
    axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    axis.set_xticks([0, 1], labels, color=MUTED)
    axis.set_yticks([0, 1], labels, color=MUTED)
    axis.set_xlabel("Dự đoán", color=MUTED)
    axis.set_ylabel("Nhãn thật", color=MUTED)
    axis.set_title("Confusion matrix trên tập test", color=INK, loc="left")
    axis.tick_params(length=0)
    for side in axis.spines.values():
        side.set_visible(False)
    for row in range(2):
        for column in range(2):
            value = matrix[row, column]
            axis.text(
                column, row, str(value), ha="center", va="center", fontsize=16,
                color="white" if value > matrix.max() / 2 else INK,
            )
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def t4_kpi_value(value, device_name):
    return value if "T4" in device_name.upper().split() else None


def main():
    config = config_from_args(__doc__)
    best_dir = project_path(config["paths"]["checkpoints_dir"]) / "best"
    if not best_dir.exists():
        raise SystemExit(f"Không thấy {best_dir}. Hãy chạy exercise_2.training trước.")
    report = read_report(config)
    if "evaluation" in report:
        raise FileExistsError("Đã có kết quả evaluation. Dùng config phiên mới.")
    training_report = report["training"]

    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    device_name = torch.cuda.get_device_name(0) if gpu else "cpu"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device).eval()
    _, _, test = load_splits(config)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as output_dir:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_eval_batch_size=config["training"]["batch_size"],
                report_to="none",
            ),
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
        )
        prediction = trainer.predict(tokenize_dataset(test, tokenizer, config))

    preds = np.argmax(prediction.predictions, axis=-1)
    matrix = confusion_matrix(prediction.label_ids, preds, labels=[0, 1])
    metrics = {key.removeprefix("test_"): value for key, value in prediction.metrics.items()}
    latency = measure_latency_ms(
        model, tokenizer, [clean_text(text) for text in test["text"]],
        config["data"]["max_length"], device,
    )
    half_width = ci_half_width(metrics["accuracy"], len(test))
    vram_values = [training_report["peak_vram_gb"]]
    if gpu:
        vram_values.append(torch.cuda.max_memory_allocated() / 1024**3)
    peak_vram = max(value for value in vram_values if value is not None) if gpu else None

    best_epoch = next(
        epoch for epoch in training_report["epochs"]
        if epoch["epoch"] == training_report["best_epoch"]
    )
    kpi = config["kpi"]
    # Mục tiêu tốc độ tham chiếu T4, tên GPU của từng phép đo quyết định đối chiếu.
    rows = [
        ("Validation Accuracy", best_epoch["val_accuracy"], kpi["accuracy"], True),
        ("Test Accuracy", metrics["accuracy"], kpi["accuracy"], True),
        ("Test Macro F1", metrics["macro_f1"], kpi["macro_f1"], True),
        ("Thời gian train (giây/epoch)",
         t4_kpi_value(training_report["mean_train_seconds_per_epoch"], training_report["device"]),
         kpi["seconds_per_epoch"], False),
        ("Độ trễ suy luận (ms/mẫu)", t4_kpi_value(latency["mean"], device_name), kpi["latency_ms"], False),
        ("VRAM cao nhất (GB)", peak_vram, kpi["peak_vram_gb"], False),
        ("RAM cao nhất (GB)", max(training_report["peak_ram_gb"], peak_ram_gb()),
         kpi["peak_ram_gb"], False),
    ]
    kpi_table = [
        {"metric": name, "value": value, "target": target,
         "status": kpi_status(value, target, higher)}
        for name, value, target, higher in rows
    ]

    figures = project_path(config["paths"]["reports_dir"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(training_report["log_history"], figures / "loss_curve.png")
    plot_confusion_matrix(matrix, figures / "confusion_matrix.png")

    update_report(config, "evaluation", {
        "device": device,
        "device_name": device_name,
        "speed_kpi_reference": "NVIDIA T4",
        "checkpoint": "best",
        "test_samples": len(test),
        "test_metrics": metrics,
        "accuracy_ci95_half_width": half_width,
        "confusion_matrix": {
            "labels": ["NEGATIVE", "POSITIVE"],
            "rows_true_columns_pred": matrix.tolist(),
        },
        "latency_ms": latency,
        "kpi": kpi_table,
    })
    for row in kpi_table:
        print(f"{row['metric']}: {row['value']} | mục tiêu {row['target']} | {row['status']}")
    print(f"Test Accuracy {metrics['accuracy']:.3f} ± {half_width:.3f} (khoảng tin cậy 95%)")


if __name__ == "__main__":
    main()
