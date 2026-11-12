"""Tổng hợp normal run đã kiểm chứng, không sửa artifact trong từng run.

Spec: docs/lab_1_description.md
Spec: docs/evaluation.md
"""
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_tracking as tracking
from verify import verify_run

SPECS = ["docs/lab_1_description.md", "docs/evaluation.md"]


def compare_runs(run_ids):
    if len(set(run_ids)) != len(run_ids) or "run_001" not in run_ids or len(run_ids) < 3:
        raise ValueError("Cần run_001 và ít nhất hai normal run khác, không trùng mã")
    rows, inputs, machines = [], [], set()
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        for run_id in run_ids:
            if not run_id.startswith("run_") or not tracking.RUN_ID_RE.fullmatch(run_id):
                raise ValueError("Chỉ so sánh normal run")
            report = verify_run(run_id)
            if not report["passed"]:
                raise ValueError(f"{run_id} không đạt: {report['errors']}")
            out = tracking.OUTPUTS / run_id
            data = {name: json.loads((out / f"{name}.json").read_text(encoding="utf-8"))
                    for name in ("config", "metrics", "performance", "manifest")}
            config, metrics, performance = (data[key] for key in ("config", "metrics", "performance"))
            resources, timing = performance["resources"], performance["timing"]
            machines.add((data["manifest"]["environment"]["machine"], resources["device_name"]))
            rows.append({"run_id": run_id, "hidden_dims": str(config["model"]["hidden_dims"]),
                         "optimizer": config["training"]["optimizer"], "learning_rate": config["training"]["learning_rate"],
                         "test_accuracy": metrics["test_accuracy"], "test_loss": metrics["test_loss"],
                         "best_val_accuracy": performance["accuracy"]["best_val_accuracy"],
                         "train_seconds": timing["train_seconds"], "inference_ms_per_image": timing["inference_ms_per_image"],
                         "num_parameters": resources["num_parameters"], "checkpoint_mb": resources["checkpoint_mb"],
                         "peak_gpu_memory_mb": resources["peak_gpu_memory_mb"], "device": resources["device_name"]})
            with (out / "history.csv").open(encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            ax.plot([int(row["epoch"]) for row in history], [float(row["val_loss"]) for row in history],
                    marker="o", label=run_id)
            inputs.extend(out / f"{name}.json" for name in data)
            inputs.append(out / "history.csv")
        if len(machines) != 1:
            raise ValueError("Normal run phải chạy trên cùng một máy/thiết bị")
        path = tracking.OUTPUTS / "compare_runs.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        ax.set(xlabel="Epoch", ylabel="Validation cross-entropy", title="FashionMNIST: compare normal runs")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        figure = tracking.OUTPUTS / "compare_loss.png"
        fig.savefig(figure, dpi=150)
    finally:
        plt.close(fig)
    for path in (path, figure):
        tracking.record_artifact(path, specs=SPECS, inputs=inputs)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="So sánh normal run hoàn tất")
    parser.add_argument("--run-ids", nargs="+", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(compare_runs(args.run_ids)))


if __name__ == "__main__":
    main()
