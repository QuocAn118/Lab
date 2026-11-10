"""Cả nhóm: ghép các file với nhau, chỉ đạt khi TV1 đến TV5 viết xong.

Chạy: python -m pytest tests/test_integration.py -q
"""
import torch
from torch.utils.data import TensorDataset

from lab_2.common import load_config, read_report
from lab_2.data import CLASS_NAMES
from lab_2.demo import plot_probabilities
from lab_2.evaluation import copy_best
from lab_2.model import build_model, load_checkpoint
from lab_2.training import train_run


def test_train_run_writes_marks_log_and_report_and_stops_early(tmp_path, monkeypatch):
    """Chạy thật vòng lặp với dữ liệu giả và model không tải trọng số."""
    dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,)))
    monkeypatch.setattr("lab_2.training.load_splits", lambda config: (dataset, dataset, dataset))
    monkeypatch.setattr(
        "lab_2.training.build_model",
        lambda name, dropout: build_model(name, dropout, pretrained=False),
    )
    config = load_config()
    config["paths"] = {"checkpoints_dir": str(tmp_path / "ckpt"), "reports_dir": str(tmp_path / "rep")}
    config["data"]["num_workers"] = 0
    # Ngưỡng không thể vượt nên mọi epoch đều không cải thiện, phải dừng đúng ở epoch 4.
    config["training"].update(max_epochs=6, early_stopping_threshold=2.0, log_every_batches=1)

    summary = train_run(config, config["runs"][0])

    assert (summary["epochs_run"], summary["stop_epoch"]) == (4, 4)
    run_dir = tmp_path / "ckpt" / "resnet18_baseline"
    assert sorted(item.name for item in run_dir.iterdir()) == [
        "best_val_acc.pth", "best_val_loss.pth", "final_epoch.pth",
    ]
    assert load_checkpoint(run_dir / "final_epoch.pth")[1]["epoch"] == 4
    assert list((tmp_path / "rep" / "runs" / "resnet18_baseline").glob("events.out.tfevents*"))
    assert read_report(config)["training"]["runs"]["resnet18_baseline"]["epochs_run"] == 4

    best = copy_best(config, "resnet18_baseline")
    assert best.name == "best.pth" and best.exists()


def test_demo_chart_uses_shared_style(tmp_path):
    result = {
        "image": "Ảnh test số 1",
        "prediction": "cat",
        "probabilities": {name: 0.1 for name in CLASS_NAMES},
    }
    plot_probabilities([result], tmp_path / "demo.png")
    assert (tmp_path / "demo.png").stat().st_size > 0
