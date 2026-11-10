"""TV2: kiểm thử src/lab_2/model.py, không tải trọng số. Chạy: python -m pytest tests/test_model.py -q"""
import pytest
import torch
from torch import nn

from lab_2.model import (
    SUPPORTED_MODELS,
    build_model,
    count_parameters,
    head_of,
    load_checkpoint,
    save_checkpoint,
    set_train_mode,
    trainable_parameters,
)


def build(name="resnet18", unfreeze=()):
    return build_model(name, dropout=0.2, unfreeze=unfreeze, pretrained=False)


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


@pytest.mark.parametrize("name, unfreeze, total, trainable", [
    ("resnet18", (), 11181642, 5130),
    ("densenet121", (), 6964106, 10250),
    ("resnet18", ("layer4",), 11181642, 8398858),
])
def test_parameter_counts_match_docs(name, unfreeze, total, trainable):
    model = build(name, unfreeze)
    assert count_parameters(model) == {"total": total, "trainable": trainable}
    assert sum(p.numel() for p in trainable_parameters(model)) == trainable


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


def test_config_only_uses_supported_models(config):
    assert all(run["model"] in SUPPORTED_MODELS for run in config["runs"])
