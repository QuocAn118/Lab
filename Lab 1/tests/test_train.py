"""Kiểm tra src/train.py bằng DataLoader tensor ngẫu nhiên, không cần dữ liệu FashionMNIST.

Spec: docs/requirements/3_4_train.md
Spec: docs/lab_1_description.md §6.1

Chạy từ thư mục Lab 1: python -m pytest tests/test_train.py -q
"""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model_io  # noqa: E402
import run_tracking  # noqa: E402
import train  # noqa: E402

CHECKPOINTS = ["final_epoch.pth", "best_val_loss.pth", "best_val_acc.pth",
               "best_val_pause.pth", "best.pth"]


@pytest.fixture(autouse=True)
def tmp_lab(tmp_path, monkeypatch):
    """Ghi outputs/ và checkpoints/ vào thư mục tạm thay cho thư mục Lab 1 thật."""
    monkeypatch.setattr(run_tracking, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(run_tracking, "CHECKPOINTS", tmp_path / "checkpoints")
    monkeypatch.setattr(model_io, "project_root", tmp_path)
    return tmp_path


def fake_loaders(sizes=(512, 256, 256)):
    generator = torch.Generator().manual_seed(0)
    loaders = []
    for i, n in enumerate(sizes):
        images = torch.rand(n, 1, 28, 28, generator=generator)
        labels = torch.randint(0, 10, (n,), generator=generator)
        loaders.append(DataLoader(TensorDataset(images, labels), batch_size=64, shuffle=i == 0))
    return tuple(loaders)


def run_training(run_id, epochs=2, lr=0.001, sizes=(512, 256, 256)):
    config = train.build_config(run_id, epochs=epochs, learning_rate=lr, device="cpu")
    with run_tracking.Run(run_id, purpose="test", config=config, specs=train.SPECS) as run:
        summary = train.train(config, run, fake_loaders(sizes))
    return run, summary


def test_writes_history_checkpoints_and_reload_check():
    run, summary = run_training("smoke_001", epochs=2)

    with (run.out_dir / "history.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == train.HISTORY_COLUMNS
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    for r in rows:
        assert 0 <= float(r["train_accuracy"]) <= 1 and 0 <= float(r["val_accuracy"]) <= 1
        assert float(r["train_loss"]) >= 0 and float(r["epoch_seconds"]) > 0

    index = json.loads((run.ckpt_dir / "checkpoint_index.json").read_text(encoding="utf-8"))
    assert sorted(index) == sorted(CHECKPOINTS)
    for name in CHECKPOINTS:
        assert (run.ckpt_dir / name).is_file()
        assert {"epoch", "val_loss", "val_accuracy", "saved_at", "elapsed_train_seconds"} <= set(index[name])
    assert index["best_val_pause.pth"]["early_stopped"] is False
    assert index["final_epoch.pth"]["epoch"] == 2
    assert (run.ckpt_dir / "best.pth").read_bytes() == (run.ckpt_dir / "best_val_acc.pth").read_bytes()
    assert index["best.pth"]["epoch"] == index["best_val_acc.pth"]["epoch"]

    reload = json.loads((run.out_dir / "reload_check.json").read_text(encoding="utf-8"))
    assert reload["passed"] is True
    assert summary["epochs_run"] == 2
    assert not (run.out_dir / "train_summary.json").exists()


def test_early_stops_after_two_epochs_without_enough_gain():
    # lr = 0: trọng số không đổi nên val accuracy không tăng; epoch 2 và 3 không đủ 0.01 -> dừng ở epoch 3.
    run, summary = run_training("smoke_002", epochs=10, lr=0.0)

    index = json.loads((run.ckpt_dir / "checkpoint_index.json").read_text(encoding="utf-8"))
    assert summary["epochs_run"] == 3 and summary["early_stopped"] is True
    assert index["best_val_pause.pth"]["early_stopped"] is True
    assert index["best_val_pause.pth"]["epoch"] == 3
    assert index["best_val_acc.pth"]["epoch"] == 1  # hòa thì giữ checkpoint sớm hơn


def test_patience_counts_gain_against_best_so_far():
    assert train.next_stale_count(None, 0.50, stale=0) == 0
    assert train.next_stale_count(0.50, 0.51, stale=1) == 0  # đúng 0.01 vẫn tính là cải thiện
    assert train.next_stale_count(0.50, 0.505, stale=0) == 1
    assert train.next_stale_count(0.50, 0.40, stale=1) == 2


def test_rejects_existing_run_id():
    run, _ = run_training("smoke_003", epochs=1)
    before = (run.out_dir / "history.csv").read_bytes()
    with pytest.raises(FileExistsError):
        run_training("smoke_003", epochs=1)
    assert (run.out_dir / "history.csv").read_bytes() == before


def test_smoke_run_rejects_full_size_loaders():
    with pytest.raises(ValueError, match="limits"):
        run_training("smoke_004", epochs=1, sizes=(600, 256, 256))


def test_config_for_experiments():
    base = train.build_config("run_001", device="cpu")
    assert base["model"]["hidden_dims"] == [512, 512]
    assert base["training"]["epochs"] == 10 and base["data"]["limits"] is None
    assert base["training"]["optimizer"] == "SGD" and base["training"]["momentum"] == 0.9
    # Đúng các trường config.json trong 3_4_train.md mục 5, cộng data.limits cho smoke run.
    assert set(base["data"]) == {"transforms", "batch_size", "num_workers", "train_shuffle",
                                 "test_shuffle", "drop_last", "limits"}
    assert set(base["training"]) == {"epochs", "seed", "loss", "optimizer", "learning_rate",
                                     "momentum", "device"}

    adam = train.build_config("run_004", optimizer="Adam", device="cpu")
    assert adam["training"]["momentum"] is None

    smoke = train.build_config("smoke_001", device="cpu")
    assert smoke["training"]["epochs"] == 1 and smoke["data"]["limits"] == train.SMOKE_LIMITS
