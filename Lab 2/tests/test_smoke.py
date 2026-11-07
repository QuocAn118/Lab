"""Smoke test theo docs/training.md: 2 batch 64 ảnh 224x224 với dữ liệu và trọng số thật."""
import pytest
import torch
from torch import nn

from lab_2.common import load_config
from lab_2.data import load_splits, make_loader
from lab_2.model import build_model, head_of, set_train_mode, trainable_parameters

BATCH_SIZE = 64
NUM_BATCHES = 2


@pytest.mark.parametrize("name, unfreeze", [
    ("resnet18", ()), ("densenet121", ()), ("resnet18", ("layer4",)),
])
def test_two_batches_train_without_shape_loss_gradient_or_memory_errors(name, unfreeze):
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train, _, _ = load_splits(config)
    loader = make_loader(train, BATCH_SIZE, config, shuffle=True)
    model = build_model(name, config["model"]["dropout"], unfreeze=unfreeze).to(device)
    set_train_mode(model)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=1e-3)
    frozen_name, frozen = next(
        (n, p) for n, p in model.named_parameters() if not p.requires_grad
    )
    frozen_before = frozen.clone()

    batches = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        assert images.shape == (BATCH_SIZE, 3, 224, 224)
        outputs = model(images)
        assert outputs.shape == (BATCH_SIZE, 10)
        loss = nn.functional.cross_entropy(outputs, labels)
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        gradient = head_of(model)[1].weight.grad
        assert gradient is not None and gradient.abs().sum() > 0
        optimizer.step()
        batches += 1
        if batches == NUM_BATCHES:
            break
    assert batches == NUM_BATCHES
    assert torch.equal(frozen, frozen_before), f"Tầng đóng băng {frozen_name} đã bị thay đổi"

    if device == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        assert vram_gb <= config["kpi"]["peak_vram_gb"]
        print(f"Smoke test {name} {unfreeze}: VRAM {vram_gb:.2f} GB")
