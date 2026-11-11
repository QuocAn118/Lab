import sys
from pathlib import Path

import pytest
import torch
from torch import nn

# Cho phép import model.py nằm cùng thư mục với test.py
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import (
    SUPPORTED_MODELS,
    build_model,
    count_parameters,
    head_of,
    load_checkpoint,
    save_checkpoint,
    set_train_mode,
    trainable_parameters,
)


@pytest.mark.parametrize("name", ["resnet18", "densenet121"])
def test_build_model_output_shape(name):
    model = build_model(name, pretrained=False)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 64, 64))

    assert output.shape == (2, 10)


@pytest.mark.parametrize("name", ["resnet18", "densenet121"])
def test_classifier_head_has_dropout(name):
    model = build_model(name, pretrained=False)
    head = head_of(model)

    assert isinstance(head, nn.Sequential)
    assert isinstance(head[0], nn.Dropout)
    assert head[0].p == pytest.approx(0.2)
    assert isinstance(head[1], nn.Linear)
    assert head[1].out_features == 10


def test_supported_models():
    assert SUPPORTED_MODELS == ("resnet18", "densenet121")


def test_invalid_model_name():
    with pytest.raises(ValueError):
        build_model("invalid_model", pretrained=False)


def test_resnet18_frozen_parameter_count():
    model = build_model("resnet18", pretrained=False)
    counts = count_parameters(model)

    assert counts == {
        "total": 11_181_642,
        "trainable": 5_130,
    }


def test_densenet121_frozen_parameter_count():
    model = build_model("densenet121", pretrained=False)
    counts = count_parameters(model)

    assert counts == {
        "total": 6_964_106,
        "trainable": 10_250,
    }


def test_resnet18_unfreeze_layer4_parameter_count():
    model = build_model(
        "resnet18",
        unfreeze=("layer4",),
        pretrained=False,
    )
    counts = count_parameters(model)

    assert counts == {
        "total": 11_181_642,
        "trainable": 8_398_858,
    }


def test_frozen_feature_extractor():
    model = build_model("resnet18", pretrained=False)

    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    )

    assert all(
        parameter.requires_grad
        for parameter in head_of(model).parameters()
    )


def test_unfreeze_layer4():
    model = build_model(
        "resnet18",
        unfreeze=("layer4",),
        pretrained=False,
    )

    assert all(
        parameter.requires_grad
        for parameter in model.layer4.parameters()
    )

    assert all(
        not parameter.requires_grad
        for parameter in model.layer3.parameters()
    )


def test_set_train_mode_keeps_frozen_batchnorm_in_eval():
    model = build_model("resnet18", pretrained=False)
    set_train_mode(model)

    assert model.training is False
    assert head_of(model).training is True
    assert model.bn1.training is False

    running_mean_before = model.bn1.running_mean.clone()

    with torch.no_grad():
        model(torch.randn(2, 3, 64, 64))

    assert torch.equal(
        running_mean_before,
        model.bn1.running_mean,
    )


def test_set_train_mode_unfrozen_layer4_batchnorm():
    model = build_model(
        "resnet18",
        unfreeze=("layer4",),
        pretrained=False,
    )
    set_train_mode(model)

    assert model.bn1.training is False
    assert model.layer4[0].bn1.training is True
    assert head_of(model).training is True


def test_checkpoint_save_and_load(tmp_path):
    model = build_model(
        "resnet18",
        unfreeze=("layer4",),
        pretrained=False,
    )

    checkpoint_path = tmp_path / "model_checkpoint.pt"

    save_checkpoint(
        model,
        checkpoint_path,
        name="resnet18",
        dropout=0.2,
        unfreeze=("layer4",),
        epoch=3,
        val_accuracy=0.85,
        val_loss=0.42,
    )

    loaded_model, info = load_checkpoint(
        checkpoint_path,
        device="cpu",
    )

    assert isinstance(loaded_model, nn.Module)
    assert info["name"] == "resnet18"
    assert info["epoch"] == 3
    assert info["val_accuracy"] == pytest.approx(0.85)
    assert info["val_loss"] == pytest.approx(0.42)
    assert "state_dict" not in info

    original_state = model.state_dict()
    loaded_state = loaded_model.state_dict()

    for key in original_state:
        assert torch.equal(original_state[key], loaded_state[key])

    assert not checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    ).exists()