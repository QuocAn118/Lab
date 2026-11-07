"""Kiểm thử logic Lab 2, không cần GPU, không tải dữ liệu hay trọng số pre-trained."""
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lab_2.common import ci_half_width, load_config, read_report, update_report
from lab_2.data import build_transform, split_indices
from lab_2.evaluation import evaluate_model, kpi_status
from lab_2.model import (
    SUPPORTED_MODELS,
    build_model,
    head_of,
    load_checkpoint,
    save_checkpoint,
    set_train_mode,
)
from lab_2.training import EarlyStopping, is_better, select_best_run


def build(name="resnet18", unfreeze=()):
    return build_model(name, dropout=0.2, unfreeze=unfreeze, pretrained=False)


def test_split_has_requested_sizes_and_no_shared_index():
    train, val = split_indices(100, train_size=70, val_size=20, seed=42)
    assert (len(train), len(val)) == (70, 20)
    assert not set(train) & set(val)


def test_split_is_reproducible_and_depends_on_seed():
    assert split_indices(100, 70, 20, 42) == split_indices(100, 70, 20, 42)
    assert split_indices(100, 70, 20, 42) != split_indices(100, 70, 20, 7)


def test_split_rejects_more_samples_than_available():
    with pytest.raises(ValueError):
        split_indices(100, 90, 20, 42)


def test_eval_transform_resizes_and_normalizes_with_imagenet_values():
    config = load_config()
    tensor = build_transform(config, train=False)(Image.new("RGB", (32, 32), (255, 255, 255)))
    assert tensor.shape == (3, 224, 224)
    assert tensor[0, 0, 0].item() == pytest.approx((1 - 0.485) / 0.229, abs=1e-4)


def test_train_transform_is_random_but_eval_transform_is_not():
    config = load_config()
    image = Image.effect_noise((32, 32), 80).convert("RGB")
    train, evaluate = build_transform(config, True), build_transform(config, False)
    assert any(not torch.equal(train(image), train(image)) for _ in range(5))
    assert torch.equal(evaluate(image), evaluate(image))


@pytest.mark.parametrize("name", SUPPORTED_MODELS)
def test_model_outputs_ten_classes_and_only_head_is_trainable(name):
    model = build(name).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 10)
    head_parameters = {id(p) for p in head_of(model).parameters()}
    for parameter in model.parameters():
        assert parameter.requires_grad == (id(parameter) in head_parameters)


def test_head_has_dropout_from_config():
    dropouts = [m for m in head_of(build()).modules() if isinstance(m, nn.Dropout)]
    assert [m.p for m in dropouts] == [0.2]


def test_unfreeze_opens_only_the_named_block_and_head():
    model = build(unfreeze=["layer4"])
    assert all(p.requires_grad for p in model.layer4.parameters())
    assert all(p.requires_grad for p in head_of(model).parameters())
    assert not any(p.requires_grad for p in model.layer3.parameters())


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build("vgg16")


def test_train_mode_keeps_frozen_batchnorm_statistics_unchanged():
    model = build()
    set_train_mode(model)
    before = model.bn1.running_mean.clone()
    model(torch.randn(4, 3, 224, 224))
    assert torch.equal(model.bn1.running_mean, before)
    assert head_of(model).training


def test_train_mode_lets_unfrozen_batchnorm_update():
    model = build(unfreeze=["layer4"])
    set_train_mode(model)
    batchnorm = model.layer4[0].bn1
    before = batchnorm.running_mean.clone()
    model(torch.randn(4, 3, 224, 224))
    assert not torch.equal(batchnorm.running_mean, before)
    assert not model.bn1.training


def test_checkpoint_round_trip_restores_weights_and_leaves_no_temp_file(tmp_path):
    model = build()
    path = tmp_path / "run" / "best_val_acc.pth"
    save_checkpoint(model, path, name="resnet18", dropout=0.2, unfreeze=[], epoch=3,
                    val_accuracy=0.8, val_loss=0.5)
    restored, info = load_checkpoint(path)
    assert torch.equal(head_of(restored)[1].weight, head_of(model)[1].weight)
    assert (info["epoch"], info["val_accuracy"]) == (3, 0.8)
    assert [item.name for item in path.parent.iterdir()] == ["best_val_acc.pth"]


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


def test_evaluate_model_computes_accuracy_loss_and_predictions():
    class Fixed(nn.Module):
        def forward(self, inputs):
            return inputs

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


def test_ci_half_width_for_ten_thousand_samples():
    assert ci_half_width(0.8, 10000) == pytest.approx(0.00784, abs=1e-4)


def test_report_keeps_sections_written_by_other_steps(tmp_path):
    config = {"paths": {"reports_dir": str(tmp_path)}}
    assert read_report(config) == {}
    update_report(config, "data", {"a": 1})
    update_report(config, "training", {"b": 2})
    assert read_report(config) == {"data": {"a": 1}, "training": {"b": 2}}


def test_config_runs_are_consistent():
    config = load_config()
    names = [run["name"] for run in config["runs"]]
    assert len(names) == len(set(names))
    for position, run in enumerate(config["runs"]):
        assert run["model"] in SUPPORTED_MODELS
        if "init_from" in run:
            assert run["init_from"] in names[:position]
