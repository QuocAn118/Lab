"""TV3: kiểm tra orchestration bằng dependency thay thế, không đóng vai code TV1/2/4/5.

Không thay đổi test nghiệm thu gốc. Chưa thay thế smoke test pretrained thật.
"""
import copy
import json
import sys
from types import ModuleType

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from lab_2 import training


@pytest.fixture
def harness(tmp_path, monkeypatch, config):
    config = copy.deepcopy(config)
    config["paths"] = {"checkpoints_dir": str(tmp_path / "checkpoints"),
                       "reports_dir": str(tmp_path / "reports")}
    config["training"].update(max_epochs=6, early_stopping_threshold=2, log_every_batches=1)
    dataset = TensorDataset(torch.randn(8, 2), torch.arange(8) % 3)
    untouched_test = object()
    report = {"data": {"keep": True}, "training": {"runs": {"other": {"keep": True}}}}
    report_path = tmp_path / "metrics.json"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "project_path", lambda path: path)
    monkeypatch.setattr(training, "load_splits", lambda cfg: (dataset, dataset, untouched_test))
    monkeypatch.setattr(training, "make_loader", lambda ds, bs, cfg, shuffle=False:
                        DataLoader(ds, batch_size=bs, shuffle=shuffle))
    monkeypatch.setattr(training, "build_model", lambda name, dropout: nn.Linear(2, 3))
    monkeypatch.setattr(training, "set_train_mode", lambda model: model.train())
    monkeypatch.setattr(training, "trainable_parameters", lambda model: list(model.parameters()))
    monkeypatch.setattr(training, "peak_ram_gb", lambda: 0.1)
    monkeypatch.setattr(training, "read_report", lambda cfg: copy.deepcopy(report))

    def update(cfg, section, value):
        report[section] = value
        report_path.write_text(json.dumps(report), encoding="utf-8")

    def save(model, path, **info):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), **info}, path)

    monkeypatch.setattr(training, "update_report", update)
    monkeypatch.setattr(training, "save_checkpoint", save)
    evaluation = ModuleType("lab_2.evaluation")
    results = iter([{"accuracy": 0.7, "loss": 0.8}, {"accuracy": 0.7, "loss": 0.6},
                    {"accuracy": 0.69, "loss": 0.7}, {"accuracy": 0.68, "loss": 0.9}])

    def evaluate(model, loader, device):
        assert loader.dataset is dataset
        return next(results)

    evaluation.evaluate_model = evaluate
    monkeypatch.setitem(sys.modules, "lab_2.evaluation", evaluation)
    return config, report, tmp_path


def test_training_writes_checkpoints_events_and_preserves_other_runs(harness):
    config, report, root = harness
    run = config["runs"][0]
    log = root / "reports/runs" / run["name"]
    sibling = log.parent / "other"
    sibling.mkdir(parents=True)
    (sibling / "keep.log").write_text("keep", encoding="utf-8")
    summary = training.train_run(config, run)
    assert (summary["epochs_run"], summary["stop_epoch"]) == (4, 4)
    assert summary["best_epoch"] == summary["best_val_loss_epoch"] == 2
    assert summary["best_val_accuracy_loss"] == 0.6
    assert summary["device"] == "cpu" and summary["peak_vram_gb"] is None
    assert report["training"]["runs"]["other"] == {"keep": True}
    assert report["data"] == {"keep": True}
    assert (sibling / "keep.log").exists()
    ckpt = root / "checkpoints" / run["name"]
    assert sorted(p.name for p in ckpt.iterdir()) == ["best_val_acc.pth", "best_val_loss.pth", "final_epoch.pth"]
    assert torch.load(ckpt / "final_epoch.pth", weights_only=True)["epoch"] == 4
    assert torch.load(ckpt / "best_val_acc.pth", weights_only=True)["epoch"] == 2
    events = EventAccumulator(str(log)).Reload()
    assert set(events.Tags()["scalars"]) == {"Loss/train", "Loss/val", "Accuracy/val", "LearningRate"}
    assert [event.step for event in events.Scalars("Accuracy/val")] == [1, 2, 3, 4]


def test_normal_completion_and_scheduler(harness):
    config, _, _ = harness
    config["training"].update(max_epochs=4, early_stopping_patience=10, scheduler_patience=0)
    summary = training.train_run(config, config["runs"][0])
    assert summary["stop_epoch"] is None
    assert summary["epochs"][-1]["learning_rate"] < summary["learning_rate"]


def test_log_path_cannot_escape_or_delete_siblings(harness):
    config, _, root = harness
    sentinel = root / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    for name in ("../outside", "..", "a/b", "a\\b", "C:outside"):
        with pytest.raises(ValueError):
            training.train_run(config, {**config["runs"][0], "name": name})
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_empty_loader_and_nonfinite_loss_are_rejected(monkeypatch):
    monkeypatch.setattr(training, "set_train_mode", lambda model: model.train())
    model = nn.Linear(2, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = (nn.CrossEntropyLoss(), optimizer, "cpu", None, 10, 0)
    with pytest.raises(ValueError):
        training.train_one_epoch(model, [], *args)
    with pytest.raises(FloatingPointError):
        training.train_one_epoch(model, [(torch.full((1, 2), float("nan")), torch.tensor([0]))], *args)


def test_empty_run_selection_is_rejected():
    with pytest.raises(ValueError):
        training.select_best_run({})


def test_existing_log_is_preserved(harness):
    config, _, root = harness
    run = config["runs"][0]
    log = root / "reports/runs" / run["name"]
    log.mkdir(parents=True)
    original = log / "original.log"
    original.write_bytes(b"preserve completed experiment")
    with pytest.raises(FileExistsError):
        training.train_run(config, run)
    assert original.read_bytes() == b"preserve completed experiment"
