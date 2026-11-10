"""TV3: kiểm thử logic src/lab_2/training.py. Chạy: python -m pytest tests/test_training.py -q"""
import pytest
import torch
from torch import nn

from lab_2.training import EarlyStopping, find_run, is_better, select_best_run, train_one_epoch


def test_is_better_prefers_accuracy_then_lower_loss():
    current = {"accuracy": 0.80, "loss": 0.5}
    assert is_better({"accuracy": 0.81, "loss": 0.9}, current)
    assert is_better({"accuracy": 0.80, "loss": 0.4}, current)
    assert not is_better({"accuracy": 0.80, "loss": 0.6}, current)
    assert is_better(current, None)


def test_select_best_run_uses_validation_results_only():
    runs = {
        "a": {"best_val_accuracy": 0.80, "best_val_accuracy_loss": 0.5},
        "b": {"best_val_accuracy": 0.85, "best_val_accuracy_loss": 0.7},
        "c": {"best_val_accuracy": 0.85, "best_val_accuracy_loss": 0.6},
    }
    assert select_best_run(runs) == "c"


def test_early_stopping_stops_after_patience_without_enough_improvement():
    stopper = EarlyStopping(patience=3, threshold=0.01)
    assert [stopper.update(a) for a in (0.70, 0.705, 0.71, 0.71)] == [False, False, False, True]


def test_early_stopping_resets_when_improvement_exceeds_threshold():
    stopper = EarlyStopping(patience=2, threshold=0.01)
    assert [stopper.update(a) for a in (0.70, 0.70, 0.75, 0.75, 0.75)] == [
        False, False, False, False, True,
    ]


def test_find_run_returns_the_run_or_lists_valid_names(config):
    assert find_run(config, "densenet121")["model"] == "densenet121"
    with pytest.raises(SystemExit, match="resnet18_baseline"):
        find_run(config, "khong_co")


def test_train_one_epoch_returns_mean_loss_and_logs_every_n_batches(monkeypatch):
    monkeypatch.setattr("lab_2.training.set_train_mode", lambda model: model.train())
    logged = []

    class Writer:
        def add_scalar(self, tag, value, step):
            logged.append((tag, step))

    model = nn.Linear(2, 3)
    inputs, labels = torch.randn(4, 2), torch.tensor([0, 1, 2, 0])
    criterion = nn.CrossEntropyLoss()
    expected = criterion(model(inputs), labels).item()
    # lr = 0 nên trọng số không đổi và loss của mọi batch đều bằng nhau.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    loss, step = train_one_epoch(
        model, [(inputs, labels)] * 4, criterion, optimizer, "cpu", Writer(), 2, 0
    )

    assert loss == pytest.approx(expected)
    assert step == 4
    assert logged == [("Loss/train", 2), ("Loss/train", 4)]
