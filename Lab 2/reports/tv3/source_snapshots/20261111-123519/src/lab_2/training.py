"""TV3: huấn luyện, checkpoint và TensorBoard cho Lab 2.

Spec: docs/training.md mục 8
Spec: docs/lab_2_description.md mục 7, 10
Chạy từ Lab 2 với PYTHONPATH=src:
    python -m lab_2.training --run resnet18_baseline
"""
from importlib import import_module, metadata
from pathlib import Path
import math
import shutil
import time

import torch
from torch import nn

LIBRARIES = ["torch", "torchvision", "tensorboard", "scikit-learn"]


def _deferred(module, name):
    """Giữ API nhóm; chỉ yêu cầu module thành viên khác khi thực sự gọi hàm."""
    def call(*args, **kwargs):
        try:
            dependency = import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name == module:
                raise ModuleNotFoundError(
                    f"Thiếu {module}: cần ghép code của thành viên phụ trách trước khi train.",
                    name=module,
                ) from exc
            raise
        return getattr(dependency, name)(*args, **kwargs)
    return call


# Những tên này cũng là điểm thay thế trong kiểm thử của nhóm.
try:
    from lab_2.common import argument_parser, load_config, peak_ram_gb, project_path, read_report, update_report
except ModuleNotFoundError as exc:
    if exc.name != "lab_2.common":
        raise
    argument_parser, load_config, peak_ram_gb, project_path, read_report, update_report = (
        _deferred("lab_2.common", name) for name in
        ("argument_parser", "load_config", "peak_ram_gb", "project_path", "read_report", "update_report"))
try:
    from lab_2.data import load_splits, make_loader
except ModuleNotFoundError as exc:
    if exc.name != "lab_2.data":
        raise
    load_splits, make_loader = (_deferred("lab_2.data", name) for name in ("load_splits", "make_loader"))
try:
    from lab_2.model import build_model, save_checkpoint, set_train_mode, trainable_parameters
except ModuleNotFoundError as exc:
    if exc.name != "lab_2.model":
        raise
    build_model, save_checkpoint, set_train_mode, trainable_parameters = (
        _deferred("lab_2.model", name) for name in
        ("build_model", "save_checkpoint", "set_train_mode", "trainable_parameters"))


def is_better(candidate, current):
    return (current is None or candidate["accuracy"] > current["accuracy"]
            or (candidate["accuracy"] == current["accuracy"] and candidate["loss"] < current["loss"]))


def select_best_run(runs):
    if not runs:
        raise ValueError("Chưa có lần chạy để chọn.")
    best_name, best = None, None
    for name, run in runs.items():
        candidate = {"accuracy": run["best_val_accuracy"], "loss": run["best_val_accuracy_loss"]}
        if is_better(candidate, best):
            best_name, best = name, candidate
    return best_name


class EarlyStopping:
    def __init__(self, patience, threshold):
        if patience < 1 or threshold < 0:
            raise ValueError("patience phải >= 1 và threshold phải >= 0")
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
    raise SystemExit(f"Tên lần chạy không hợp lệ: {name}. Chọn: "
                     + ", ".join(run["name"] for run in config["runs"]))


def train_one_epoch(model, loader, criterion, optimizer, device, writer, log_every, step):
    if log_every < 1:
        raise ValueError("log_every phải >= 1")
    set_train_mode(model)
    total, batches = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Train loss không hữu hạn")
        loss.backward()
        optimizer.step()
        step += 1
        batches += 1
        total += loss.item()
        if step % log_every == 0:
            writer.add_scalar("Loss/train", loss.item(), step)
    if not batches:
        raise ValueError("Train loader không có batch")
    return total / batches, step


def _log_directory(config, run_name):
    """Xác nhận đường dẫn trước khi xóa log cũ, kể cả trên Windows."""
    if not run_name or run_name in (".", "..") or any(c in run_name for c in '/\\:'):
        raise ValueError("Tên run phải là một tên thư mục đơn")
    parent = (Path(project_path(config["paths"]["reports_dir"])) / "runs").resolve()
    destination = parent / run_name
    if destination.is_symlink() or destination.resolve().parent != parent:
        raise ValueError("Thư mục log vượt ngoài reports/runs")
    return destination


def train_run(config, run):
    # Import ở đây để tránh evaluation -> training -> evaluation.
    from lab_2.evaluation import evaluate_model
    from torch.utils.tensorboard import SummaryWriter

    settings = config["training"]
    if settings["max_epochs"] < 1:
        raise ValueError("max_epochs phải >= 1")
    log_dir = _log_directory(config, run["name"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    else:
        print("Cảnh báo: không có GPU, chạy CPU; không đo được KPI GPU.")
    torch.manual_seed(config["seed"])
    train, val, _ = load_splits(config)  # Không duyệt test trong huấn luyện.
    train_loader = make_loader(train, run["batch_size"], config, shuffle=True)
    val_loader = make_loader(val, settings["eval_batch_size"], config)
    model = build_model(run["model"], config["model"]["dropout"]).to(device)
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=run["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=settings["scheduler_patience"])
    stopper = EarlyStopping(settings["early_stopping_patience"], settings["early_stopping_threshold"])
    criterion = nn.CrossEntropyLoss()
    checkpoint_dir = Path(project_path(config["paths"]["checkpoints_dir"])) / run["name"]
    epochs, step, best_acc, best_loss, stop_epoch = [], 0, None, None, None
    best_epoch = best_loss_epoch = None
    # Chỉ xóa thư mục run đã xác nhận, không ảnh hưởng các run khác.
    if log_dir.exists():
        shutil.rmtree(log_dir)
    with SummaryWriter(log_dir=str(log_dir)) as writer:
        for epoch in range(1, settings["max_epochs"] + 1):
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            train_loss, step = train_one_epoch(model, train_loader, criterion, optimizer,
                                               device, writer, settings["log_every_batches"], step)
            if device == "cuda":
                torch.cuda.synchronize()
            train_seconds = time.perf_counter() - started
            result = evaluate_model(model, val_loader, device)
            if not math.isfinite(result["loss"]) or not math.isfinite(result["accuracy"]):
                raise FloatingPointError("Validation metrics không hữu hạn")
            learning_rate = optimizer.param_groups[0]["lr"]
            marks = ["final_epoch"]
            if best_loss is None or result["loss"] < best_loss:
                best_loss, best_loss_epoch = result["loss"], epoch
                marks.append("best_val_loss")
            if is_better(result, best_acc):
                best_acc = {"accuracy": result["accuracy"], "loss": result["loss"]}
                best_epoch = epoch
                marks.append("best_val_acc")
            for mark in marks:
                save_checkpoint(model, checkpoint_dir / f"{mark}.pth", name=run["model"],
                                dropout=config["model"]["dropout"], unfreeze=[], epoch=epoch,
                                val_accuracy=result["accuracy"], val_loss=result["loss"])
            writer.add_scalar("Loss/val", result["loss"], epoch)
            writer.add_scalar("Accuracy/val", result["accuracy"], epoch)
            writer.add_scalar("LearningRate", learning_rate, epoch)
            scheduler.step(result["loss"])
            epochs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": result["loss"],
                           "val_accuracy": result["accuracy"], "learning_rate": learning_rate,
                           "train_seconds": train_seconds, "marks": marks})
            print(f"{run['name']} epoch {epoch}: train_loss={train_loss:.4f}, "
                  f"val_loss={result['loss']:.4f}, val_accuracy={result['accuracy']:.4f}, "
                  f"train_seconds={train_seconds:.2f}")
            if stopper.update(result["accuracy"]):
                stop_epoch = epoch
                break
    summary = {key: run[key] for key in ("purpose", "model", "learning_rate", "batch_size")}
    summary.update(device=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                   library_versions={name: metadata.version(name) for name in LIBRARIES},
                   epochs=epochs, epochs_run=len(epochs), stop_epoch=stop_epoch,
                   best_epoch=best_epoch, best_val_accuracy=best_acc["accuracy"],
                   best_val_accuracy_loss=best_acc["loss"], best_val_loss_epoch=best_loss_epoch,
                   best_val_loss=best_loss,
                   mean_train_seconds_per_epoch=sum(e["train_seconds"] for e in epochs) / len(epochs),
                   peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else None,
                   peak_ram_gb=peak_ram_gb())
    training = read_report(config).get("training", {"runs": {}})
    training.setdefault("runs", {})[run["name"]] = summary
    update_report(config, "training", training)
    return summary


def main():
    parser = argument_parser(__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    summary = train_run(config, find_run(config, args.run))
    print(f"Best epoch: {summary['best_epoch']}; val accuracy: {summary['best_val_accuracy']:.4f}; "
          f"giây/epoch: {summary['mean_train_seconds_per_epoch']:.2f}")


if __name__ == "__main__":
    main()
