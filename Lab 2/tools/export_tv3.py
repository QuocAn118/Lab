"""Xuất so sánh validation và biểu đồ từ bốn lần chạy thật của TV3."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lab_2.common import load_config, read_report
from lab_2.evaluation import comparison_table, plot_val_accuracy
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()
    runs = read_report(config)["training"]["runs"]
    expected = {run["name"] for run in config["runs"]}
    if set(runs) != expected:
        raise ValueError(f"Expected {expected}, got {set(runs)}")
    folder = ROOT / "reports/tv3"
    folder.mkdir(parents=True, exist_ok=True)
    rows = comparison_table(runs)
    with (folder / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = ROOT / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_val_accuracy(runs, figures / "tv3_val_accuracy.png")
    tags = {"Loss/train", "Loss/val", "Accuracy/val", "LearningRate"}
    events_report = {}
    checkpoints = {}
    for name in runs:
        events = EventAccumulator(str(ROOT / "reports/runs" / name)).Reload()
        if not tags <= set(events.Tags()["scalars"]):
            raise ValueError(f"Missing TensorBoard tags: {name}")
        for tag in tags - {"Loss/train"}:
            if len(events.Scalars(tag)) != runs[name]["epochs_run"]:
                raise ValueError(f"Wrong event count: {name} {tag}")
        events_report[name] = {tag: len(events.Scalars(tag)) for tag in sorted(tags)}
        run = runs[name]
        checkpoints[name] = {}
        for mark, epoch in (("final_epoch", run["epochs_run"]), ("best_val_acc", run["best_epoch"]),
                            ("best_val_loss", run["best_val_loss_epoch"])):
            path = ROOT / "checkpoints" / name / (mark + ".pth")
            info = torch.load(path, map_location="cpu", weights_only=True)
            if info["epoch"] != epoch or info["name"] != run["model"]:
                raise ValueError(f"Checkpoint metadata mismatch: {path}")
            expected_epoch = run["epochs"][epoch - 1]
            if info["val_accuracy"] != expected_epoch["val_accuracy"] or info["val_loss"] != expected_epoch["val_loss"]:
                raise ValueError(f"Checkpoint metrics mismatch: {path}")
            checkpoints[name][mark] = {"epoch": epoch, "bytes": path.stat().st_size,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder / "tensorboard_check.json").write_text(json.dumps(events_report, indent=2), encoding="utf-8")
    (folder / "checkpoint_check.json").write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")
    lines = ["# Báo cáo TV3 — Huấn luyện và TensorBoard, Lab 2", "",
             f"Ngày xuất: {datetime.now().astimezone().isoformat(timespec='seconds')}. "
             "Khai báo công cụ: `docs/AI_Usage.md`.", "",
             "## Kết quả bốn cấu hình", "",
             "Nguồn: `reports/metrics.json`, mục `training.runs`; so sánh bằng validation, không dùng test.", "",
             "| Run | Epoch | Best epoch | Val accuracy | Val loss | Giây/epoch | Stop epoch | VRAM GB |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        run = runs[row["run"]]
        lines.append(f"| {row['run']} | {run['epochs_run']} | {run['best_epoch']} | "
                     f"{run['best_val_accuracy']:.2%} | {run['best_val_accuracy_loss']:.4f} | "
                     f"{run['mean_train_seconds_per_epoch']:.2f} | {run['stop_epoch'] or '—'} | "
                     f"{run['peak_vram_gb']:.3f} |")
    base = runs["resnet18_baseline"]["best_val_accuracy"]
    lines += ["", "## Nhận xét Exercise 1 và 2", ""]
    for name in ("densenet121", "resnet18_lr1e-2", "resnet18_bs128"):
        change = 100 * (runs[name]["best_val_accuracy"] - base)
        lines.append(f"- `{name}` chênh {change:+.2f} điểm phần trăm validation accuracy so với baseline.")
    lines += ["", "Mỗi cấu hình chạy một lần với seed 42 trên "
              + runs["resnet18_baseline"]["device"]
              + ". Kết quả không đại diện cho benchmark nhiều seed hoặc GPU T4 của Colab.", "",
              "## Exercise 3 — TensorBoard", "",
              "Cả bốn run có đủ `Loss/train`, `Loss/val`, `Accuracy/val`, `LearningRate`; "
              "số điểm theo epoch được đối chiếu với summary.", "",
              "- Log: `reports/runs/<run>/`.",
              "- Kiểm tra event: [tensorboard_check.json](../reports/tv3/tensorboard_check.json).",
              "- Kiểm tra checkpoint: [checkpoint_check.json](../reports/tv3/checkpoint_check.json).",
              "- Biểu đồ validation: [tv3_val_accuracy.png](../reports/figures/tv3_val_accuracy.png).",
              "- [Ảnh chụp TensorBoard](../reports/figures/tensorboard.png), tạo bằng `tools/capture_tensorboard.py`.", "",
              "## Kiểm chứng", "",
              "- 41 test logic/integration bổ sung và ba smoke test pretrained đạt trên CPU; log: `reports/tv3/20261111-122708/`.",
              "- 41 test đạt trong 8.98s và ba smoke test đạt trong 39.28s trên GPU; log: `reports/tv3/20261111-123359/`.",
              "- Bốn run có trạng thái completed; log, môi trường và hash source: `reports/tv3/20261111-123519/manifest.json`.",
              "- Ba mốc checkpoint mỗi run khớp epoch/validation metrics; đủ event TensorBoard theo số epoch thực tế.", "",
              "## Checkpoint và bàn giao", "",
              "Mỗi run có `final_epoch.pth`, `best_val_loss.pth`, `best_val_acc.pth` trong "
              "`checkpoints/<run>/`. Early stopping dùng patience 3 và mức tăng phải hơn 0.01. "
              "Checkpoint accuracy ưu tiên accuracy cao, hòa thì loss thấp.", "",
              "Không chạy đánh giá test hoặc tạo `best.pth` trong phần TV3; TV4 tiếp nhận các run "
              "và dùng validation để chọn mô hình trước khi đánh giá test. Đủ 12 checkpoint được version "
              "bằng Git LFS. Nguồn code lịch sử khớp manifest và giới hạn schema cũ được ghi tại "
              "[biên bản review](TV3_REVIEW_RESOLUTION.md); không gán kết quả cũ cho code tích hợp mới.", "",
              "## Cách kiểm tra lại", "", "```powershell",
              "$env:PYTHONPATH = 'src'",
              "& .venv/Scripts/python.exe tools/run_tv3.py --stage verify",
              "& .venv/Scripts/python.exe tools/export_tv3.py", "```", "",
              "Log kiểm chứng và nguồn code được lưu trong `reports/tv3/<timestamp>/`. "
              "Công cụ train từ chối ghi đè run đã có. Chi tiết ghép phụ thuộc xem [TV3_TRAINING_GUIDE.md](TV3_TRAINING_GUIDE.md)."]
    (ROOT / "docs/training.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
