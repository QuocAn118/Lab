# Sources: docs/training.md; docs/plan_progress_status.md
"""Smoke test theo docs/training.md: 2 batch, mỗi batch 16 mẫu, chạy trước khi train thật."""
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from exercise_2.common import load_config
from exercise_2.data import load_splits, tokenize_dataset
from exercise_2.model import load_model, load_tokenizer

BATCH_SIZE = 16
NUM_BATCHES = 2


def test_two_batches_train_without_shape_loss_or_gradient_errors():
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(config)
    train, _, _ = load_splits(config)
    tokenized = tokenize_dataset(
        train.select(range(BATCH_SIZE * NUM_BATCHES)), tokenizer, config
    )
    loader = DataLoader(
        tokenized, batch_size=BATCH_SIZE, collate_fn=DataCollatorWithPadding(tokenizer)
    )
    model = load_model(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])

    batches = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = model(**batch)
        assert outputs.logits.shape == (BATCH_SIZE, 2)
        assert torch.isfinite(outputs.loss)
        optimizer.zero_grad()
        outputs.loss.backward()
        gradient = model.classifier.weight.grad
        assert gradient is not None and gradient.abs().sum() > 0
        optimizer.step()
        batches += 1
    assert batches == NUM_BATCHES

    if device == "cuda":
        vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        assert vram_gb <= config["kpi"]["peak_vram_gb"]
        print(f"Smoke test VRAM: {vram_gb:.2f} GB")
