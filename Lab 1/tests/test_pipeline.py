"""Behavior checks for evaluation, best reload, CLI pipeline and tamper detection.

Spec: docs/verification.md
"""
import csv
import json
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model_io
import run_tracking
import train
from evaluate import evaluate_model
from model import build_model
from verify import verify_run


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tracking, "OUTPUTS", tmp_path / "outputs")
    monkeypatch.setattr(run_tracking, "CHECKPOINTS", tmp_path / "checkpoints")
    monkeypatch.setattr(model_io, "project_root", tmp_path)
    torch.set_num_threads(2)
    return tmp_path


def test_evaluation_weights_loss_and_reports_known_predictions():
    logits = torch.zeros(4, 10)
    logits[:, 0] = 2
    labels = torch.tensor([0, 1, 0, 2])
    result = evaluate_model(torch.nn.Identity(), DataLoader(TensorDataset(logits, labels), batch_size=3), "cpu")
    assert result["num_samples"] == 4 and result["num_correct"] == 2
    assert result["test_accuracy"] == 0.5
    assert result["test_loss"] == pytest.approx(torch.nn.functional.cross_entropy(logits, labels).item())
    assert result["predictions"] == [0, 0, 0, 0]


def test_evaluation_rejects_empty_and_nonfinite_logits():
    empty = DataLoader(TensorDataset(torch.zeros(0, 10), torch.zeros(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="rỗng"):
        evaluate_model(torch.nn.Identity(), empty, "cpu")
    bad = DataLoader(TensorDataset(torch.full((2, 10), float("nan")), torch.zeros(2, dtype=torch.long)))
    with pytest.raises(FloatingPointError):
        evaluate_model(torch.nn.Identity(), bad, "cpu")


@pytest.mark.parametrize('status', ['completed', 'failed', 'interrupted'])
def test_reload_check_cannot_modify_a_finalized_run(lab, status):
    out = lab / 'outputs/run_010'
    out.mkdir(parents=True)
    config = train.build_config('run_010', hidden_dims=(8, 4))
    (out / 'config.json').write_text(json.dumps(config))
    model = build_model(config['model'])
    model_io.save_model(model, 'run_010')
    (out / 'manifest.json').write_text(json.dumps({'status': status}))
    saved = out / 'reload_check.json'
    saved.write_bytes(b'preserve original verification')
    with pytest.raises(FileExistsError, match='chốt'):
        model_io.verify_reload(model, torch.zeros(16, 1, 28, 28), 'run_010')
    assert saved.read_bytes() == b'preserve original verification'


def test_load_selects_best_independently_and_preserves_exclusive_save(lab):
    config = train.build_config("smoke_101", hidden_dims=(8, 4))
    out = lab / "outputs/smoke_101"
    out.mkdir(parents=True)
    run_tracking.write_json(out / "config.json", config)
    model = build_model(config["model"])
    model_io.save_model(model, "smoke_101")
    final = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        next(model.parameters()).add_(1)
    torch.save(model.state_dict(), lab / "checkpoints/smoke_101/best.pth")
    best = model_io.load_model("smoke_101", "cpu", checkpoint="best")
    last = model_io.load_model("smoke_101", "cpu")
    assert all(torch.equal(value, final[name]) for name, value in last.state_dict().items())
    assert any(not torch.equal(value, final[name]) for name, value in best.state_dict().items())
    with pytest.raises(FileExistsError):
        model_io.save_model(model, "smoke_101")
    with pytest.raises(ValueError):
        model_io.load_model("../outside", "cpu")
    with pytest.raises(ValueError):
        model_io.load_model("smoke_101", "cpu", checkpoint="../outside")


def test_cli_pipeline_completes_and_readonly_verifier_detects_tampering(lab, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data import data_pipeline
    generator = torch.Generator().manual_seed(7)
    loaders = tuple(DataLoader(TensorDataset(torch.rand(n, 1, 28, 28, generator=generator),
                                             torch.randint(0, 10, (n,), generator=generator)), batch_size=64)
                    for n in (512, 256, 256))
    monkeypatch.setattr(data_pipeline, "create_dataloaders", lambda config: loaders)
    # Verify the CPU/null-memory path even when the test machine has a GPU.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    train.main(["--run-id", "smoke_102", "--purpose", "integration test", "--hidden-dims", "8", "4"])
    out = lab / "outputs/smoke_102"
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert verify_run("smoke_102")["passed"] is True
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}
    reload = json.loads((out / "reload_check.json").read_text())
    assert reload["checkpoint"].endswith("/best.pth")
    performance = json.loads((out / "performance.json").read_text())
    assert performance["resources"]["peak_gpu_memory_mb"] is None
    with (out / "predictions.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 256
    rows[0]["predicted_label"] = str((int(rows[0]["predicted_label"]) + 1) % 10)
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert verify_run("smoke_102")["passed"] is False


@pytest.mark.parametrize("epochs", [0, -1, 11])
def test_config_rejects_invalid_epoch_count(epochs):
    with pytest.raises(ValueError):
        train.build_config("run_001", epochs=epochs)


def test_model_check_records_matching_sidecar(lab, monkeypatch):
    import model
    monkeypatch.setattr(model, "__file__", str(lab / "src/model.py"))
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    model.check_model()
    path = lab / "outputs/model_summary.txt"
    sidecar = json.loads(path.with_name(path.name + ".prov.json").read_text(encoding="utf-8"))
    assert sidecar["sha256"] == run_tracking.sha256_file(path)
    assert "669706" in path.read_text(encoding="utf-8")


def test_comparison_preserves_each_runs_metrics_and_provenance(lab, monkeypatch):
    import compare_runs
    monkeypatch.setattr(compare_runs, "verify_run", lambda run_id: {"passed": True})
    ids = ["run_001", "run_002", "run_003"]
    for i, run_id in enumerate(ids):
        out = lab / "outputs" / run_id
        out.mkdir(parents=True)
        records = {
            "config": {"model": {"hidden_dims": [8, 4]}, "training": {"optimizer": "SGD", "learning_rate": 0.001 * (i + 1)}},
            "metrics": {"test_accuracy": 0.7 + 0.05 * i, "test_loss": 0.9 - 0.1 * i},
            "performance": {"accuracy": {"best_val_accuracy": 0.72 + 0.05 * i},
                            "timing": {"train_seconds": 2 + i, "inference_ms_per_image": 0.3 + i},
                            "resources": {"device_name": "CPU test", "num_parameters": 100 + i,
                                          "checkpoint_mb": 0.1 + i, "peak_gpu_memory_mb": None}},
            "manifest": {"environment": {"machine": "same-test-machine"}},
        }
        for name, data in records.items():
            run_tracking.write_json(out / f"{name}.json", data)
        (out / "history.csv").write_text(f"epoch,val_loss\n1,{0.9 - 0.1 * i}\n", encoding="utf-8")
    compare_runs.compare_runs(ids)
    path = lab / "outputs/compare_runs.csv"
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["run_id"] for row in rows] == ids
    assert [float(row["test_accuracy"]) for row in rows] == [0.7 + 0.05 * i for i in range(3)]
    assert [float(row["train_seconds"]) for row in rows] == [2, 3, 4]
    assert all(row["peak_gpu_memory_mb"] == "" for row in rows)
    sidecar = json.loads(path.with_name(path.name + ".prov.json").read_text(encoding="utf-8"))
    assert sidecar["sha256"] == run_tracking.sha256_file(path)
