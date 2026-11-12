"""Kiểm số liệu trước khi chốt run; CLI chỉ đọc và kiểm thêm manifest/hash.

Spec: docs/verification.md
Spec: docs/lab_1_description.md
"""
import argparse
import csv
import json
import math
from datetime import datetime

import torch
import run_tracking as tracking

CHECKPOINTS = ("final_epoch.pth", "best_val_loss.pth", "best_val_acc.pth", "best_val_pause.pth", "best.pth")
HISTORY = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "epoch_seconds"]


def verify_run(run_id, finalized=True):
    if not tracking.RUN_ID_RE.fullmatch(run_id):
        raise ValueError("Invalid run_id")
    out, checkpoints = tracking.OUTPUTS / run_id, tracking.CHECKPOINTS / run_id
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    def read(name):
        return json.loads((out / name).read_text(encoding="utf-8"))

    required = [out / name for name in ("config.json", "history.csv", "metrics.json", "predictions.csv",
                "performance.json", "reload_check.json", "loss_curve.png", "prediction_samples.png")]
    required += [checkpoints / name for name in (*CHECKPOINTS, "checkpoint_index.json")]
    if finalized:
        required += [out / "manifest.json", out / "verify_report.json"]
    for path in required:
        check(path.is_file(), f"Missing {path.name}")
    report = {"run_id": run_id, "verified_at": tracking.now_iso(),
              "phase": "finalized" if finalized else "before_manifest", "num_samples": 0}
    if errors:
        return {**report, "passed": False, "errors": errors}
    config, metrics, performance, reload = (read(name) for name in
                                          ("config.json", "metrics.json", "performance.json", "reload_check.json"))
    check(config["run_id"] == run_id and metrics["run_id"] == run_id, "run_id mismatch")
    classes = config["classes"]
    check(len(classes) == 10 and len(set(classes)) == 10, "Expected ten unique classes")
    with (out / "history.csv").open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames == HISTORY, "History columns mismatch")
        history = list(reader)
    check(bool(history) and len(history) <= config["training"]["epochs"], "Invalid epoch count")
    if not history:
        return {**report, "passed": False, "errors": errors}
    for epoch, row in enumerate(history, 1):
        check(int(row["epoch"]) == epoch, "Epoch sequence mismatch")
        for key in HISTORY[1:]:
            value = float(row[key])
            check(math.isfinite(value) and value >= 0, f"Invalid history {key}")
            if key.endswith("accuracy"):
                check(value <= 1, f"Accuracy outside [0,1]: {key}")
    with (out / "predictions.csv").open(encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        check(reader.fieldnames == ["sample_index", "true_label", "predicted_label", "true_name", "predicted_name"],
              "Prediction columns mismatch")
        predictions = list(reader)
    expected = config["data"]["limits"]["test"] if config["data"]["limits"] else 10000
    check(len(predictions) == expected, "Test sample count mismatch")
    report["num_samples"] = len(predictions)
    counts, correct_per_class = [0] * 10, [0] * 10
    for i, row in enumerate(predictions):
        label, prediction = int(row["true_label"]), int(row["predicted_label"])
        check(int(row["sample_index"]) == i, "Test sample order mismatch")
        valid = 0 <= label < 10 and 0 <= prediction < 10
        check(valid, "Prediction label outside [0,9]")
        if valid:
            check(row["true_name"] == classes[label] and row["predicted_name"] == classes[prediction],
                  "Class name mismatch")
            counts[label] += 1
            correct_per_class[label] += label == prediction
    correct = sum(correct_per_class)
    accuracy = correct / len(predictions) if predictions else 0
    check(metrics["num_samples"] == len(predictions) and metrics["num_correct"] == correct, "Metric counts mismatch")
    check(math.isclose(metrics["test_accuracy"], accuracy, abs_tol=1e-12), "CSV accuracy differs from metrics")
    check(math.isfinite(metrics["test_loss"]) and metrics["test_loss"] >= 0, "Invalid test loss")
    check(metrics["split"] == "test" and metrics["checkpoint"] == f"checkpoints/{run_id}/best.pth",
          "Test must use best checkpoint")
    check(reload["passed"] is True and reload["labels_match"] is True and reload["checked_samples"] == 16,
          "Reload check failed")
    check(reload["checkpoint"] == f"checkpoints/{run_id}/best.pth" and
          0 <= reload["max_abs_logit_difference"] <= reload["tolerance"] <= 1e-6, "Best reload tolerance mismatch")
    check(set(performance) == {"accuracy", "timing", "resources"}, "Performance groups mismatch")
    quality, timing, resources = (performance[key] for key in ("accuracy", "timing", "resources"))
    check(quality["test_accuracy"] == metrics["test_accuracy"] and quality["test_loss"] == metrics["test_loss"],
          "Performance/metrics mismatch")
    for i, name in enumerate(classes):
        expected_accuracy = correct_per_class[i] / counts[i] if counts[i] else None
        check(quality["per_class_accuracy"].get(name) == expected_accuracy, f"Per-class accuracy mismatch: {name}")
    best_acc = max(history, key=lambda row: float(row["val_accuracy"]))
    best_loss = min(history, key=lambda row: float(row["val_loss"]))
    check(quality["best_val_accuracy"] == float(best_acc["val_accuracy"]) and
          quality["best_val_accuracy_epoch"] == int(best_acc["epoch"]), "Best validation accuracy mismatch")
    check(quality["best_val_loss"] == float(best_loss["val_loss"]) and
          quality["best_val_loss_epoch"] == int(best_loss["epoch"]), "Best validation loss mismatch")
    for key in ("train_seconds", "test_seconds", "inference_ms_per_image"):
        check(math.isfinite(timing[key]) and timing[key] > 0, f"Invalid timing: {key}")
    check(abs(sum(float(row["epoch_seconds"]) for row in history) - timing["train_seconds"]) <= 0.01,
          "Training timing/history mismatch")
    check(timing["epoch_seconds"] == [float(row["epoch_seconds"]) for row in history], "Epoch timing mismatch")
    check(math.isclose(timing["inference_ms_per_image"], timing["test_seconds"] * 1000 / expected), "Latency mismatch")
    check(datetime.fromisoformat(timing["training_started_at"]) <= datetime.fromisoformat(timing["training_finished_at"]),
          "Training timestamps reversed")
    dims = [784, *config["model"]["hidden_dims"], 10]
    check(resources["num_parameters"] == sum((a + 1) * b for a, b in zip(dims, dims[1:])), "Parameter count mismatch")
    check(resources["checkpoint_bytes"] == (checkpoints / "best.pth").stat().st_size and
          resources["checkpoint_mb"] == resources["checkpoint_bytes"] / 1_000_000, "Checkpoint size mismatch")
    check(resources["device"] == metrics["device"], "Device mismatch")
    peak = resources["peak_gpu_memory_mb"]
    check((peak is None) if metrics["device"] == "cpu" else (peak is not None and math.isfinite(peak) and peak > 0),
          "GPU memory mismatch")
    index = json.loads((checkpoints / "checkpoint_index.json").read_text(encoding="utf-8"))
    check(set(index) == set(CHECKPOINTS), "Checkpoint index mismatch")
    for name, row in [("best_val_acc.pth", best_acc), ("best.pth", best_acc), ("best_val_loss.pth", best_loss),
                      ("best_val_pause.pth", history[-1]), ("final_epoch.pth", history[-1])]:
        item = index[name]
        check(item["epoch"] == int(row["epoch"]) and item["val_loss"] == float(row["val_loss"]) and
              item["val_accuracy"] == float(row["val_accuracy"]), f"Checkpoint metadata mismatch: {name}")
        check(0 <= item["elapsed_train_seconds"] <= timing["train_seconds"] + 0.002, "Invalid checkpoint timing")
        check(datetime.fromisoformat(item["saved_at"]).tzinfo is not None, "Checkpoint timestamp needs timezone")
    check(tracking.sha256_file(checkpoints / "best.pth") == tracking.sha256_file(checkpoints / "best_val_acc.pth"),
          "Best is not a byte copy of best_val_acc")
    final = torch.load(checkpoints / "final_epoch.pth", map_location="cpu", weights_only=True)
    pause = torch.load(checkpoints / "best_val_pause.pth", map_location="cpu", weights_only=True)
    check(final.keys() == pause.keys() and all(torch.equal(final[key], pause[key]) for key in final),
          "Final/pause weights differ")
    for name in ("loss_curve.png", "prediction_samples.png"):
        with (out / name).open("rb") as stream:
            check(stream.read(8) == b"\x89PNG\r\n\x1a\n", f"Invalid PNG: {name}")
    if finalized:
        manifest = read("manifest.json")
        check(manifest["status"] == "completed", "Manifest is not completed")
        check(read("verify_report.json")["passed"] is True, "Original verification did not pass")
        listed = {item["path"] for item in manifest["files"]}
        present = {tracking.rel(p) for folder in (out, checkpoints) for p in folder.rglob("*")
                   if p.is_file() and p != out / "manifest.json"}
        check(listed == present, "Manifest file inventory mismatch")
        for item in manifest["files"]:
            path = (tracking.LAB_ROOT / item["path"]).resolve()
            inside = path.is_relative_to(out.resolve()) or path.is_relative_to(checkpoints.resolve())
            check(inside and path.is_file(), "Invalid/missing manifest file")
            if inside and path.is_file():
                check(tracking.sha256_file(path) == item["sha256"] and path.stat().st_size == item["bytes"],
                      f"Manifest hash/size mismatch: {path.name}")
    return {**report, "passed": not errors, "errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kiểm tra finalized run, không ghi thêm file")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    report = verify_run(args.run_id)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
