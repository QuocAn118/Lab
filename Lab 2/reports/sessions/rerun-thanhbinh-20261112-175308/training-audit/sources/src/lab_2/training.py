"""Bước 6 và Exercise 1–3: train, checkpoint, early stopping, TensorBoard."""
from datetime import datetime
from importlib.metadata import version
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from lab_2.common import argument_parser, load_config, peak_ram_gb, project_path, read_report, update_report
from lab_2.data import load_splits, make_loader
from lab_2.model import build_model, save_checkpoint, set_train_mode, trainable_parameters

LIBRARIES = ["torch", "torchvision", "tensorboard", "scikit-learn"]


def is_better(candidate, current):
    return current is None or candidate["accuracy"] > current["accuracy"] or (
        candidate["accuracy"] == current["accuracy"] and candidate["loss"] < current["loss"]
    )


def select_best_run(runs):
    if not runs:
        raise ValueError("Chưa có kết quả validation")
    return max(runs, key=lambda name: (runs[name]["best_val_accuracy"], -runs[name]["best_val_accuracy_loss"]))


class EarlyStopping:
    def __init__(self, patience, threshold):
        if patience < 1 or threshold < 0:
            raise ValueError("Patience phải dương và threshold không âm")
        self.patience, self.threshold = patience, threshold
        self.best, self.waited = None, 0

    def update(self, accuracy):
        if not math.isfinite(accuracy):
            raise ValueError("Validation accuracy không hữu hạn")
        if self.best is None or round(accuracy - self.best, 10) > self.threshold:
            self.best, self.waited = accuracy, 0
            return False
        self.waited += 1
        return self.waited >= self.patience


def find_run(config, name):
    for run in config["runs"]:
        if run["name"] == name:
            return run
    raise SystemExit(f"Run không hợp lệ. Chọn: {[run['name'] for run in config['runs']]}")


def train_one_epoch(model, loader, criterion, optimizer, device, writer, log_every, step):
    if log_every < 1:
        raise ValueError("log_every phải dương")
    set_train_mode(model)
    losses = []
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Train loss không hữu hạn")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        step += 1
        if step % log_every == 0:
            writer.add_scalar("Loss/train", loss.item(), step)
    if not losses:
        raise ValueError("Loader train rỗng")
    return statistics.mean(losses), step


def train_run(config, run):
    from lab_2.evaluation import evaluate_model

    settings = config["training"]
    if settings["max_epochs"] < 1 or settings["log_every_batches"] < 1:
        raise ValueError("max_epochs và log_every_batches phải dương")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run["name"]):
        raise ValueError("Tên run chỉ được gồm chữ, số, _ và -")
    checkpoint_parent = Path(project_path(config["paths"]["checkpoints_dir"])).resolve()
    log_parent = (Path(project_path(config["paths"]["reports_dir"])) / "runs").resolve()
    checkpoint_dir = checkpoint_parent / run["name"]
    log_dir = log_parent / run["name"]
    for path, parent in ((checkpoint_dir, checkpoint_parent), (log_dir, log_parent)):
        if path.is_symlink() or path.resolve().parent != parent:
            raise ValueError("Đường dẫn run vượt thư mục dự kiến")
    training = read_report(config).get("training", {"runs": {}})
    if run["name"] in training["runs"] or any(p.exists() and any(p.iterdir()) for p in (checkpoint_dir, log_dir)):
        raise FileExistsError("Run đã có kết quả/log. Giữ nguyên và dùng thư mục reports/checkpoints mới trong config.")
    gpu = torch.cuda.is_available()
    device = "cuda" if gpu else "cpu"
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    else:
        print("Không có GPU: chạy CPU; KPI GPU sẽ ghi Không đo được.")
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    train, val, _ = load_splits(config)
    train_loader = make_loader(train, run["batch_size"], config, shuffle=True)
    val_loader = make_loader(val, settings["eval_batch_size"], config)
    model = build_model(run["model"], config["model"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=run["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=settings["scheduler_patience"])
    stopper = EarlyStopping(settings["early_stopping_patience"], settings["early_stopping_threshold"])
    epochs, step, best, best_loss, stop_epoch = [], 0, None, math.inf, None
    best_epoch = best_loss_epoch = None
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_path("."), capture_output=True, text=True)
    with SummaryWriter(str(log_dir)) as writer:
        for epoch in range(1, settings["max_epochs"] + 1):
            if gpu:
                torch.cuda.synchronize()
            started = time.perf_counter()
            train_loss, step = train_one_epoch(model, train_loader, nn.CrossEntropyLoss(), optimizer,
                                              device, writer, settings["log_every_batches"], step)
            if gpu:
                torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            result = evaluate_model(model, val_loader, device)
            if not math.isfinite(result["loss"]) or not math.isfinite(result["accuracy"]):
                raise FloatingPointError("Validation metrics không hữu hạn")
            lr = optimizer.param_groups[0]["lr"]
            marks = ["final_epoch"]
            if result["loss"] < best_loss:
                best_loss, best_loss_epoch = result["loss"], epoch
                marks.append("best_val_loss")
            if is_better(result, best):
                best, best_epoch = {"accuracy": result["accuracy"], "loss": result["loss"]}, epoch
                marks.append("best_val_acc")
            for mark in marks:
                save_checkpoint(model, checkpoint_dir / f"{mark}.pth", name=run["model"],
                                dropout=config["model"]["dropout"], unfreeze=[], epoch=epoch,
                                val_accuracy=result["accuracy"], val_loss=result["loss"])
            writer.add_scalar("Loss/val", result["loss"], epoch)
            writer.add_scalar("Accuracy/val", result["accuracy"], epoch)
            writer.add_scalar("LearningRate", lr, epoch)
            scheduler.step(result["loss"])
            epochs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": result["loss"],
                           "val_accuracy": result["accuracy"], "learning_rate": lr,
                           "train_seconds": train_seconds, "marks": marks})
            print(f"{run['name']} epoch {epoch}: train={train_loss:.4f} val={result['loss']:.4f} acc={result['accuracy']:.4f}")
            if stopper.update(result["accuracy"]):
                stop_epoch = epoch
                break
    summary = {
        **run, "device": torch.cuda.get_device_name(0) if gpu else "cpu",
        "library_versions": {name: version(name) for name in LIBRARIES},
        "epochs": epochs, "epochs_run": len(epochs), "stop_epoch": stop_epoch,
        "best_epoch": best_epoch, "best_val_accuracy": best["accuracy"], "best_val_accuracy_loss": best["loss"],
        "best_val_loss_epoch": best_loss_epoch, "best_val_loss": best_loss,
        "mean_train_seconds_per_epoch": statistics.mean(e["train_seconds"] for e in epochs),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3 if gpu else None,
        "peak_ram_gb": peak_ram_gb(), "config": config,
        "started_at": started_at, "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operator": os.environ.get("LAB2_OPERATOR", "Chưa khai báo"),
        "code_commit": code.stdout.strip() if code.returncode == 0 else None,
    }
    training["runs"][run["name"]] = summary
    update_report(config, "training", training)
    return summary


def main():
    parser = argument_parser(__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = train_run(config, find_run(config, args.run))
    print(f"Best epoch {result['best_epoch']}, val acc {result['best_val_accuracy']:.4f}, {result['mean_train_seconds_per_epoch']:.2f} giây/epoch")


if __name__ == "__main__":
    main()
