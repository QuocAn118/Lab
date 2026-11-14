# Sources: docs/docs demo/training.md; docs/docs demo/lab_3_description.md; docs/08_implementation.md §10
"""Exercise 2, bước 5 và 6: khai báo tham số, tạo Trainer và fine-tune. Đặc tả: docs/docs demo/training.md."""
import importlib.metadata
import shutil
import tempfile
import time

import torch
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from exercise_2.common import config_from_args, peak_ram_gb, project_path, read_report, update_report
from exercise_2.data import load_splits, tokenize_dataset
from exercise_2.evaluation import compute_metrics
from exercise_2.model import load_model, load_tokenizer

LIBRARIES = ["torch", "transformers", "datasets", "evaluate", "accelerate", "scikit-learn"]


def is_better(candidate, current):
    """Quy tắc chọn mốc `best`: accuracy cao hơn; nếu bằng nhau thì loss thấp hơn."""
    if current is None:
        return True
    if candidate["accuracy"] != current["accuracy"]:
        return candidate["accuracy"] > current["accuracy"]
    return candidate["loss"] < current["loss"]


def save_checkpoint(model, tokenizer, destination):
    """Ghi vào thư mục tạm rồi mới thay thế, để lỗi giữa chừng không làm hỏng mốc cũ."""
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.rmtree(temporary, ignore_errors=True)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    shutil.rmtree(destination, ignore_errors=True)
    temporary.rename(destination)


class CheckpointCallback(TrainerCallback):
    """Sau mỗi epoch: ghi nhật ký và cập nhật các mốc final_epoch, best_val_loss, best_val_acc."""

    def __init__(self, tokenizer, checkpoints_dir):
        self.tokenizer = tokenizer
        self.checkpoints_dir = checkpoints_dir
        self.epochs = []
        self.best_by_accuracy = None
        self.best_loss = None
        self.logged_until = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_started = time.perf_counter()

    def on_epoch_end(self, args, state, control, **kwargs):
        self.train_seconds = time.perf_counter() - self.epoch_started

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        candidate = {"accuracy": metrics["eval_accuracy"], "loss": metrics["eval_loss"]}
        flags = ["final_epoch"]
        # Trainer nạp lại model tốt nhất trước on_train_end, nên final_epoch phải lưu ngay tại đây.
        save_checkpoint(model, self.tokenizer, self.checkpoints_dir / "final_epoch")
        if self.best_loss is None or candidate["loss"] < self.best_loss:
            self.best_loss = candidate["loss"]
            save_checkpoint(model, self.tokenizer, self.checkpoints_dir / "best_val_loss")
            flags.append("best_val_loss")
        if is_better(candidate, self.best_by_accuracy):
            self.best_by_accuracy = candidate
            save_checkpoint(model, self.tokenizer, self.checkpoints_dir / "best_val_acc")
            flags.append("best_val_acc")

        train_losses = [
            entry["loss"] for entry in state.log_history[self.logged_until:] if "loss" in entry
        ]
        self.logged_until = len(state.log_history)
        self.epochs.append({
            "epoch": round(state.epoch),
            "train_loss": sum(train_losses) / len(train_losses) if train_losses else None,
            "val_loss": metrics["eval_loss"],
            "val_accuracy": metrics["eval_accuracy"],
            "val_precision": metrics["eval_precision"],
            "val_recall": metrics["eval_recall"],
            "val_f1": metrics["eval_f1"],
            "val_macro_f1": metrics["eval_macro_f1"],
            "train_seconds": self.train_seconds,
            "flags": flags,
        })


def build_training_args(config, output_dir):
    training = config["training"]
    return TrainingArguments(
        output_dir=output_dir,
        seed=config["seed"],
        data_seed=config["seed"],
        learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["batch_size"],
        per_device_eval_batch_size=training["batch_size"],
        num_train_epochs=training["max_epochs"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        warmup_steps=training["warmup_steps"],
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=training["logging_steps"],
        report_to="none",
        fp16=torch.cuda.is_available(),
    )


def main():
    config = config_from_args(__doc__)
    checkpoints_dir = project_path(config["paths"]["checkpoints_dir"])
    if checkpoints_dir.exists() and any(checkpoints_dir.iterdir()):
        raise FileExistsError(f"Đã có checkpoint tại {checkpoints_dir}. Dùng config phiên mới.")
    report_path = project_path(config["paths"]["reports_dir"]) / "metrics.json"
    if report_path.exists() and "training" in read_report(config):
        raise FileExistsError(f"Đã có kết quả training tại {report_path}. Dùng config phiên mới.")
    set_seed(config["seed"])
    gpu = torch.cuda.is_available()
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    else:
        print("CẢNH BÁO: không có GPU. Vẫn chạy được nhưng các KPI tốc độ và VRAM sẽ không đo được.")

    tokenizer = load_tokenizer(config)
    train, val, _ = load_splits(config)  # Tập test chỉ được dùng trong evaluation.py.
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    callback = CheckpointCallback(tokenizer, checkpoints_dir)
    training = config["training"]

    # Checkpoint nội bộ của Trainer để ở ổ tạm: chỉ cần cho load_best_model_at_end, không cần lên Drive.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as trainer_dir:
        trainer = Trainer(
            model=load_model(config),
            args=build_training_args(config, trainer_dir),
            train_dataset=tokenize_dataset(train, tokenizer, config),
            eval_dataset=tokenize_dataset(val, tokenizer, config),
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[
                callback,
                EarlyStoppingCallback(
                    early_stopping_patience=training["early_stopping_patience"],
                    early_stopping_threshold=training["early_stopping_threshold"],
                ),
            ],
        )
        trainer.train()

    best_dir = checkpoints_dir / "best"
    shutil.rmtree(best_dir, ignore_errors=True)
    shutil.copytree(checkpoints_dir / "best_val_acc", best_dir)

    epochs_run = len(callback.epochs)
    update_report(config, "training", {
        "device": torch.cuda.get_device_name(0) if gpu else "cpu",
        "library_versions": {name: importlib.metadata.version(name) for name in LIBRARIES},
        "epochs": callback.epochs,
        "epochs_run": epochs_run,
        "max_epochs": training["max_epochs"],
        "stopped_early": epochs_run < training["max_epochs"],
        "best_epoch": max(
            callback.epochs, key=lambda epoch: (epoch["val_accuracy"], -epoch["val_loss"])
        )["epoch"],
        "mean_train_seconds_per_epoch": sum(e["train_seconds"] for e in callback.epochs) / epochs_run,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if gpu else None,
        "peak_ram_gb": peak_ram_gb(),
        "log_history": trainer.state.log_history,
    })
    for epoch in callback.epochs:
        print(epoch)
    print(f"Đã chạy {epochs_run}/{training['max_epochs']} epoch. Mốc tốt nhất: {best_dir}")


if __name__ == "__main__":
    main()
