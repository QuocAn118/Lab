"""Kiểm chứng bàn giao offline; mọi kết quả giả chỉ nằm trong tmp_path."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from lab_2 import data, demo, evaluation, model, training
from lab_2.common import load_config, read_report, update_report


def isolated_config(tmp_path):
    config = load_config()
    config["paths"] = {"reports_dir": str(tmp_path / "reports"), "checkpoints_dir": str(tmp_path / "ckpt")}
    config["data"]["num_workers"] = 0
    return config


@pytest.mark.parametrize('module', [data, model, training, evaluation, demo])
def test_module_cli_preserves_finalized_session(tmp_path, monkeypatch, module):
    config = isolated_config(tmp_path)
    report_dir = tmp_path / 'session/reports'
    report_dir.mkdir(parents=True)
    config['paths']['reports_dir'] = str(report_dir)
    config['paths']['checkpoints_dir'] = str(tmp_path / 'session/checkpoints')
    (tmp_path / 'session/SESSION.json').write_text(json.dumps({'status': 'completed'}))
    metrics = report_dir / 'metrics.json'
    metrics.write_bytes(b'{"demo": "preserve evidence"}')
    image = report_dir / 'figures/demo_probabilities.png'
    image.parent.mkdir()
    image.write_bytes(b'preserve original plot')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    args = ['module', '--config', str(path)]
    if module is training:
        args += ['--run', config['runs'][0]['name']]
    monkeypatch.setattr(sys, 'argv', args)
    loading = {data: 'load_splits', model: 'build_model', training: 'load_splits',
               evaluation: 'completed_runs', demo: 'load_checkpoint'}
    monkeypatch.setattr(module, loading[module],
                        lambda *args, **kwargs: pytest.fail('loading before finalized-session check'))
    with pytest.raises(FileExistsError, match='chốt'):
        module.main()
    assert metrics.read_bytes() == b'{"demo": "preserve evidence"}'
    assert image.read_bytes() == b'preserve original plot'


def test_copy_best_preserves_finalized_checkpoint_directory(tmp_path):
    session = tmp_path / 'session'
    checkpoints = session / 'checkpoints'
    checkpoints.mkdir(parents=True)
    (session / 'FILES.json').write_text('{}')
    best = checkpoints / 'best.pth'
    best.write_bytes(b'preserve selected model')
    config = {'paths': {'checkpoints_dir': str(checkpoints)}}
    with pytest.raises(FileExistsError, match='chốt'):
        evaluation.copy_best(config, 'resnet18_baseline')
    assert best.read_bytes() == b'preserve selected model'


def test_existing_checkpoint_is_not_overwritten(tmp_path):
    config = isolated_config(tmp_path)
    path = tmp_path / "ckpt" / config["runs"][0]["name"] / "best_val_acc.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"preserve existing experiment")
    with pytest.raises(FileExistsError):
        training.train_run(config, config["runs"][0])
    assert path.read_bytes() == b"preserve existing experiment"


def test_invalid_report_update_preserves_previous_bytes(tmp_path):
    config = isolated_config(tmp_path)
    update_report(config, "data", {"samples": 10})
    path = tmp_path / "reports/metrics.json"
    before = path.read_bytes()
    with pytest.raises(ValueError):
        update_report(config, "evaluation", {"accuracy": float("nan")})
    assert path.read_bytes() == before


def test_evaluation_rejects_empty_or_nonfinite_data():
    with pytest.raises(ValueError, match="rỗng"):
        evaluation.evaluate_model(torch.nn.Identity(), [], "cpu")
    loader = DataLoader(TensorDataset(torch.tensor([[float("nan"), 0.0]]), torch.tensor([0])))
    with pytest.raises(FloatingPointError):
        evaluation.evaluate_model(torch.nn.Identity(), loader, "cpu")


def test_model_main_reads_yaml_and_preserves_other_sections(tmp_path, monkeypatch):
    config = isolated_config(tmp_path)
    config["model"]["dropout"] = 0.35
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    update_report(config, "data", {"samples": 10})
    build = model.build_model
    seen = []

    def offline_build(name, dropout, unfreeze=()):
        seen.append(dropout)
        return build(name, dropout, unfreeze, pretrained=False)

    monkeypatch.setattr(model, "build_model", offline_build)
    monkeypatch.setattr(sys, "argv", ["model", "--config", str(path)])
    model.main()
    report = read_report(config)
    assert seen == [0.35] * 3 and report["data"] == {"samples": 10}
    assert report["model"]["resnet18"] == {"total": 11181642, "trainable": 5130}


def test_checkpoint_with_string_unfreeze_can_be_reloaded(tmp_path):
    network = model.build_model("resnet18", unfreeze="layer4", pretrained=False)
    path = tmp_path / "best.pth"
    model.save_checkpoint(network, path, name="resnet18", dropout=0.2, unfreeze="layer4",
                          epoch=1, val_accuracy=0.1, val_loss=2.0)
    _, info = model.load_checkpoint(path)
    assert info["unfreeze"] == ["layer4"]


def test_offline_train_evaluate_and_user_image_demo(tmp_path, monkeypatch):
    config = isolated_config(tmp_path)
    config["runs"] = config["runs"][:1]
    config["training"]["max_epochs"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.arange(8))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(training, "load_splits", lambda config: (dataset, dataset, dataset))
    monkeypatch.setattr(training, "build_model", lambda name, dropout: model.build_model(name, dropout, pretrained=False))
    monkeypatch.setenv("LAB2_OPERATOR", "offline-test")
    training.train_run(config, config["runs"][0])
    monkeypatch.setattr(evaluation, "load_splits", lambda config: (dataset, dataset, dataset))
    monkeypatch.setattr(evaluation, "raw_samples", lambda subset, count: [(Image.new("RGB", (32, 32)), 0)] * 2)
    monkeypatch.setattr(sys, "argv", ["evaluation", "--config", str(path)])
    evaluation.main()
    report = read_report(config)
    result = report["evaluation"]
    assert result["test_samples"] == 8
    assert sum(map(sum, result["confusion_matrix"]["rows_true_columns_pred"])) == 8
    assert report["training"]["runs"][config["runs"][0]["name"]]["operator"] == "offline-test"
    image = tmp_path / "gray.png"
    Image.new("L", (20, 20)).save(image)
    monkeypatch.setattr(sys, "argv", ["demo", "--config", str(path), "--image", str(image)])
    demo.main()
    predicted, = read_report(config)["demo"]
    assert predicted["truth"] is None and predicted["correct"] is None
    assert sum(predicted["probabilities"].values()) == pytest.approx(1.0)
    assert (tmp_path / "reports/figures/demo_probabilities.png").stat().st_size > 0
    monkeypatch.setattr(sys, "argv", ["evaluation", "--config", str(path)])
    with pytest.raises(SystemExit, match="Đã có"):
        evaluation.main()


def test_evaluation_refuses_mismatched_config_before_using_test_data(tmp_path, monkeypatch):
    config = isolated_config(tmp_path)
    update_report(config, "training", {"runs": {run["name"]: {"config": {}} for run in config["runs"]}})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["evaluation", "--config", str(path)])
    with pytest.raises(SystemExit, match="Config"):
        evaluation.main()


def test_colab_peak_ram_converts_linux_kib_to_gib(monkeypatch):
    from lab_2.common import peak_ram_gb
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=2 * 1024**2),
    ))
    monkeypatch.setattr(sys, "platform", "linux")
    assert peak_ram_gb() == 2.0


def test_notebook_session_setup_preserves_parameters_and_rejects_config_changes(tmp_path, monkeypatch):
    notebook = Path(__file__).resolve().parents[1] / "notebooks/Lab_2_colab.ipynb"
    cells = [c for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    original = load_config()
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/lab_2.yaml").write_text(yaml.safe_dump(original), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    namespace = {"Path": Path, "PROJECT_DIR": tmp_path, "SESSION_ID": "offline_test",
                 "sys": sys, "subprocess": subprocess}
    setup = compile("".join(cells[2]["source"]), str(notebook), "exec")
    exec(setup, namespace)
    path = namespace["CONFIG_PATH"]
    session = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert session["runs"] == original["runs"] and session["training"] == original["training"]
    assert session["data"] == original["data"]
    assert session["paths"]["reports_dir"] == str(Path("reports/sessions/offline_test/reports"))
    session["training"]["max_epochs"] = 999
    path.write_text(yaml.safe_dump(session), encoding="utf-8")
    with pytest.raises(AssertionError, match="Config"):
        exec(setup, namespace)
