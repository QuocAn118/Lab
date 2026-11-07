"""TV4: kiểm thử src/lab_2/evaluation.py. Chạy: python -m pytest tests/test_evaluation.py -q"""
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab_2.evaluation import (
    comparison_table,
    completed_runs,
    evaluate_model,
    kpi_status,
    measure_latency_ms,
    plot_confusion_matrix,
    plot_loss_curve,
    plot_val_accuracy,
)


class Fixed(nn.Module):
    def forward(self, inputs):
        return inputs


def fake_run():
    return {
        "purpose": "Mô hình gốc", "model": "resnet18", "learning_rate": 0.001, "batch_size": 64,
        "epochs_run": 2, "best_epoch": 2, "best_val_accuracy": 0.8, "best_val_accuracy_loss": 0.5,
        "mean_train_seconds_per_epoch": 60.0,
        "epochs": [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 0.9, "val_accuracy": 0.7},
            {"epoch": 2, "train_loss": 0.6, "val_loss": 0.5, "val_accuracy": 0.8},
        ],
    }


def test_evaluate_model_computes_accuracy_loss_and_predictions():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    labels = torch.tensor([0, 1, 1, 0])
    result = evaluate_model(Fixed(), DataLoader(TensorDataset(logits, labels), batch_size=3), "cpu")
    assert result["accuracy"] == 0.75
    assert result["loss"] == pytest.approx(nn.functional.cross_entropy(logits, labels).item())
    assert result["predictions"] == [0, 1, 0, 0]
    assert result["labels"] == [0, 1, 1, 0]


def test_kpi_status_handles_both_directions_and_missing_value():
    assert kpi_status(0.9, 0.8, True) == "Đạt"
    assert kpi_status(0.7, 0.8, True) == "Không đạt"
    assert kpi_status(10, 25, False) == "Đạt"
    assert kpi_status(None, 25, False) == "Không đo được"


def test_evaluation_stops_and_names_the_missing_runs():
    config = {"runs": [{"name": "a"}, {"name": "b"}]}
    with pytest.raises(SystemExit, match="b"):
        completed_runs(config, {"training": {"runs": {"a": {}}}})
    report = {"training": {"runs": {"a": {"x": 1}, "b": {"x": 2}}}}
    assert list(completed_runs(config, report)) == ["a", "b"]


def test_latency_returns_mean_and_p95():
    samples = [(torch.zeros(1), 0)] * 10
    latency = measure_latency_ms(Fixed(), lambda image: image, samples, "cpu")
    assert 0 <= latency["mean"] and 0 <= latency["p95"]


def test_comparison_table_has_one_row_per_run():
    row = comparison_table({"a": fake_run()})[0]
    assert (row["run"], row["val_accuracy"], row["seconds_per_epoch"]) == ("a", 0.8, 60.0)


def test_plots_write_image_files(tmp_path):
    plot_loss_curve("a", fake_run(), tmp_path / "loss.png")
    plot_val_accuracy({"a": fake_run()}, tmp_path / "accuracy.png")
    plot_confusion_matrix(np.eye(10, dtype=int), tmp_path / "matrix.png")
    names = ("loss.png", "accuracy.png", "matrix.png")
    assert all((tmp_path / name).stat().st_size > 0 for name in names)
