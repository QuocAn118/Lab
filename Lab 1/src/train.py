"""Huấn luyện MLP FashionMNIST: train + validation mỗi epoch, early stopping, 5 checkpoint.

Spec: docs/requirements/3_4_train.md
Spec: docs/lab_1_description.md §6.1

Chạy từ thư mục Lab 1:

    python src/train.py --run-id smoke_001 --purpose "Smoke run"
    python src/train.py --run-id run_001 --purpose "Cấu hình cơ sở"
    python src/train.py --run-id run_002 --purpose "Lớp ẩn [256, 128]" --hidden-dims 256 128
    python src/train.py --run-id run_003 --purpose "Learning rate 0.01" --lr 0.01
    python src/train.py --run-id run_004 --purpose "Adam, lr 0.001" --optimizer Adam
"""
import argparse
import csv
import math
import shutil
import time

import torch
from torch import nn

from model import build_model
from model_io import save_model, verify_reload
from run_tracking import Run, now_iso, write_json

SPECS = ["docs/requirements/3_4_train.md", "docs/lab_1_description.md"]
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
SMOKE_LIMITS = {"train": 512, "val": 256, "test": 256}
HISTORY_COLUMNS = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "epoch_seconds"]
RELOAD_SAMPLES = 16
PATIENCE = 2  # Mục 6: dừng khi 2 epoch liên tiếp val accuracy tăng dưới MIN_DELTA.
MIN_DELTA = 0.01


def build_config(run_id, hidden_dims=(512, 512), optimizer="SGD", learning_rate=0.001,
                 epochs=None, device="cpu"):
    """Cấu hình ghi vào config.json. Run tự thêm run_id, created_by, started_at."""
    smoke = run_id.startswith("smoke_")
    return {
        "schema_version": 1,
        "dataset": "FashionMNIST",
        "classes": CLASSES,
        "model": {"name": "FashionMLP", "input_shape": [1, 28, 28], "hidden_dims": list(hidden_dims),
                  "activation": "ReLU", "num_classes": 10},
        "data": {"transforms": ["ToTensor"], "batch_size": 64, "num_workers": 0,
                 "train_shuffle": True, "test_shuffle": False, "drop_last": False,
                 "limits": SMOKE_LIMITS if smoke else None},  # Smoke run: số ảnh mục 5.1.
        "training": {"epochs": epochs or (1 if smoke else 10), "seed": 42, "loss": "CrossEntropyLoss",
                     "optimizer": optimizer, "learning_rate": learning_rate,
                     "momentum": 0.9 if optimizer == "SGD" else None, "device": device},
    }


def next_stale_count(best_acc, val_acc, stale, min_delta=MIN_DELTA):
    """Số epoch liên tiếp val accuracy tăng dưới min_delta so với mức tốt nhất trước đó."""
    if best_acc is None:
        return 0
    # round: tránh 0.51 - 0.50 = 0.00999... bị tính là chưa đủ 0.01.
    return 0 if round(val_acc - best_acc, 10) >= min_delta else stale + 1


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Một lượt qua loader. Có optimizer thì huấn luyện, không có thì chỉ đánh giá."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    mean_loss = total_loss / count
    if not math.isfinite(mean_loss):
        raise FloatingPointError(f"Loss không hữu hạn: {mean_loss}")
    return mean_loss, correct / count


def check_smoke_sizes(config, loaders):
    limits = config["data"]["limits"]
    if limits is None:
        return
    sizes = {name: len(loader.dataset) for name, loader in zip(("train", "val", "test"), loaders)}
    if sizes != limits:
        raise ValueError(f"create_dataloaders chưa áp dụng data.limits: cần {limits}, nhận {sizes}")


def train(config, run, loaders):
    """Huấn luyện, ghi history.csv, 5 checkpoint và checkpoint_index.json."""
    train_loader, val_loader, _ = loaders  # Không dùng test để huấn luyện hay chọn checkpoint.
    check_smoke_sizes(config, loaders)
    cfg = config["training"]
    device = cfg["device"]
    cuda = device.startswith("cuda")

    torch.manual_seed(cfg["seed"])
    model = build_model(config["model"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    if cfg["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["learning_rate"], momentum=cfg["momentum"])
    elif cfg["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    else:
        raise ValueError(f"Optimizer không hỗ trợ: {cfg['optimizer']}")

    index, index_path = {}, run.ckpt_dir / "checkpoint_index.json"
    train_seconds = 0.0

    def record(name, epoch, val_loss, val_acc, **extra):
        index[name] = {"epoch": epoch, "val_loss": val_loss, "val_accuracy": val_acc,
                       "saved_at": now_iso(), "elapsed_train_seconds": round(train_seconds, 3), **extra}
        write_json(index_path, index)

    def save(name, *info, **extra):
        torch.save(model.state_dict(), run.ckpt_dir / name)
        record(name, *info, **extra)

    best_loss = best_acc = None
    best_loss_epoch = best_acc_epoch = None
    stale, early_stopped = 0, False
    with (run.out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HISTORY_COLUMNS)
        f.flush()
        for epoch in range(1, cfg["epochs"] + 1):
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
            if cuda:
                torch.cuda.synchronize()
            epoch_seconds = time.perf_counter() - t0  # Chỉ tính train, không tính validation.
            train_seconds += epoch_seconds
            val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)

            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, round(epoch_seconds, 3)])
            f.flush()
            print(f"epoch {epoch}: train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
                  f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} ({epoch_seconds:.1f}s)")

            if best_loss is None or val_loss < best_loss:
                best_loss, best_loss_epoch = val_loss, epoch
                save("best_val_loss.pth", epoch, val_loss, val_acc)
            stale = next_stale_count(best_acc, val_acc, stale)
            if best_acc is None or val_acc > best_acc:  # Hòa thì giữ checkpoint sớm hơn.
                best_acc, best_acc_epoch = val_acc, epoch
                save("best_val_acc.pth", epoch, val_loss, val_acc)
            if stale >= PATIENCE:
                early_stopped = True
                run.note(f"early stopping tại epoch {epoch}: {stale} epoch liên tiếp val_acc tăng dưới "
                         f"{MIN_DELTA}")
                break

    last = (epoch, val_loss, val_acc)
    save("best_val_pause.pth", *last, early_stopped=early_stopped)
    shutil.copyfile(run.ckpt_dir / "best_val_acc.pth", run.ckpt_dir / "best.pth")
    index["best.pth"] = {**index["best_val_acc.pth"], "saved_at": now_iso()}
    save_model(model, run.run_id)  # Ghi final_epoch.pth (TV2, model_io.py).
    record("final_epoch.pth", *last)

    summary = {"epochs_run": epoch, "early_stopped": early_stopped,
               "best_val_accuracy": best_acc, "best_val_accuracy_epoch": best_acc_epoch,
               "best_val_loss": best_loss, "best_val_loss_epoch": best_loss_epoch}

    sample = torch.stack([val_loader.dataset[i][0] for i in range(RELOAD_SAMPLES)])
    if not verify_reload(model, sample, run.run_id):
        raise RuntimeError("Tải lại final_epoch.pth không khớp, xem reload_check.json")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Huấn luyện MLP FashionMNIST")
    parser.add_argument("--run-id", required=True, help="run_001 hoặc smoke_001")
    parser.add_argument("--purpose", required=True, help="Mục đích lần chạy, ghi vào manifest.json")
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--optimizer", choices=["SGD", "Adam"], default="SGD")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, help="Mặc định 10, smoke run 1")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = build_config(args.run_id, args.hidden_dims, args.optimizer, args.lr, args.epochs, device)
    from data_pipeline import create_dataloaders  # TV1; import ở đây để test không cần file này.

    with Run(args.run_id, purpose=args.purpose, config=config, specs=SPECS) as run:
        summary = train(config, run, create_dataloaders(config))
        run.note(f"best val_acc {summary['best_val_accuracy']:.4f} tại epoch {summary['best_val_accuracy_epoch']}")


if __name__ == "__main__":
    main()
