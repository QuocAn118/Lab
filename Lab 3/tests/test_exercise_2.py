# Sources: docs/lab_3_description.md; docs/plan_progress_status.md; docs/08_implementation.md §10
"""Kiểm thử logic Exercise 2, không cần GPU hay tải model."""
import numpy as np
import pytest
from datasets import Dataset

from exercise_1.sentiment_analysis import validate_text
from exercise_2.common import ci_half_width
from exercise_2.data import clean_text, split_dataset
from exercise_2.evaluation import compute_classification_metrics, kpi_status
from exercise_2.training import is_better


@pytest.mark.parametrize('existing', ['checkpoint', 'training_report'])
def test_training_refuses_existing_experiment_before_downloading(tmp_path, monkeypatch, existing):
    from exercise_2 import common, training
    config = {'paths': {'checkpoints_dir': str(tmp_path / 'checkpoints'),
                        'reports_dir': str(tmp_path / 'reports')}}
    if existing == 'checkpoint':
        marker = tmp_path / 'checkpoints/model.safetensors'
        marker.parent.mkdir()
        marker.write_bytes(b'preserve weights')
    else:
        common.update_report(config, 'training', {'completed': True})
        marker = tmp_path / 'reports/metrics.json'
    before = marker.read_bytes()
    monkeypatch.setattr(training, 'config_from_args', lambda _: config)
    monkeypatch.setattr(training, 'load_tokenizer', lambda _: pytest.fail('download before preflight'))
    with pytest.raises(FileExistsError):
        training.main()
    assert marker.read_bytes() == before


def test_nonfinite_report_update_preserves_existing_json(tmp_path):
    from exercise_2.common import update_report
    config = {'paths': {'reports_dir': str(tmp_path)}}
    update_report(config, 'data', {'samples': 10})
    before = (tmp_path / 'metrics.json').read_bytes()
    with pytest.raises(ValueError):
        update_report(config, 'evaluation', {'accuracy': float('nan')})
    assert (tmp_path / 'metrics.json').read_bytes() == before


def test_evaluation_refuses_existing_result_before_loading_model(tmp_path, monkeypatch):
    from exercise_2 import common, evaluation
    config = {'paths': {'checkpoints_dir': str(tmp_path / 'checkpoints'),
                        'reports_dir': str(tmp_path / 'reports')}}
    (tmp_path / 'checkpoints/best').mkdir(parents=True)
    common.update_report(config, 'evaluation', {'accuracy': 0.9})
    before = (tmp_path / 'reports/metrics.json').read_bytes()
    monkeypatch.setattr(evaluation, 'config_from_args', lambda _: config)
    monkeypatch.setattr(evaluation.AutoTokenizer, 'from_pretrained', lambda _: pytest.fail('model loaded'))
    with pytest.raises(FileExistsError):
        evaluation.main()
    assert (tmp_path / 'reports/metrics.json').read_bytes() == before


def test_demo_refuses_existing_result_before_loading_model(tmp_path, monkeypatch):
    from exercise_2 import common, demo
    config = {'paths': {'reports_dir': str(tmp_path)}}
    common.update_report(config, 'demo', [{'prediction': 'POSITIVE'}])
    before = (tmp_path / 'metrics.json').read_bytes()
    monkeypatch.setattr(demo, 'config_from_args', lambda _: config)
    monkeypatch.setattr(demo.AutoTokenizer, 'from_pretrained', lambda _: pytest.fail('model loaded'))
    with pytest.raises(FileExistsError):
        demo.main()
    assert (tmp_path / 'metrics.json').read_bytes() == before


@pytest.mark.parametrize('device, expected', [('Tesla T4', 12.5), ('NVIDIA T4', 12.5),
                                            ('NVIDIA GeForce RTX 5060 Laptop GPU', None), ('cpu', None)])
def test_speed_kpi_uses_t4_measurements(device, expected):
    from exercise_2.evaluation import t4_kpi_value
    assert t4_kpi_value(12.5, device) == expected


def make_dataset(prefix, n):
    return Dataset.from_dict(
        {"text": [f"{prefix} {i}" for i in range(n)], "label": [i % 2 for i in range(n)]}
    )


def test_splits_have_requested_sizes_and_share_no_sample():
    train, val, test = split_dataset(
        make_dataset("train", 50), make_dataset("test", 30),
        train_size=20, val_size=10, test_size=5, seed=42,
    )
    assert (len(train), len(val), len(test)) == (20, 10, 5)
    assert not set(train["text"]) & set(val["text"])
    assert not (set(train["text"]) | set(val["text"])) & set(test["text"])


def test_split_is_reproducible_with_same_seed():
    args = dict(train_size=20, val_size=10, test_size=5, seed=42)
    first = split_dataset(make_dataset("train", 50), make_dataset("test", 30), **args)
    second = split_dataset(make_dataset("train", 50), make_dataset("test", 30), **args)
    assert list(first[0]["text"]) == list(second[0]["text"])


def test_split_rejects_request_larger_than_source():
    with pytest.raises(ValueError):
        split_dataset(
            make_dataset("train", 10), make_dataset("test", 10),
            train_size=8, val_size=5, test_size=5, seed=42,
        )


def test_clean_text_replaces_imdb_line_breaks_with_space():
    assert clean_text("Great film.<br /><br />Loved it") == "Great film.  Loved it"


def test_higher_accuracy_wins_even_with_worse_loss():
    assert is_better({"accuracy": 0.9, "loss": 0.5}, {"accuracy": 0.8, "loss": 0.1})


def test_accuracy_tie_is_broken_by_lower_loss():
    assert is_better({"accuracy": 0.9, "loss": 0.2}, {"accuracy": 0.9, "loss": 0.3})
    assert not is_better({"accuracy": 0.9, "loss": 0.3}, {"accuracy": 0.9, "loss": 0.2})


def test_first_candidate_is_always_best():
    assert is_better({"accuracy": 0.1, "loss": 9.0}, None)


def test_metrics_match_hand_computed_values():
    # TP=2 FN=1 FP=1 TN=2: mỗi lớp có precision = recall = 2/3.
    metrics = compute_classification_metrics(
        preds=np.array([1, 1, 0, 1, 0, 0]), labels=np.array([1, 1, 1, 0, 0, 0])
    )
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["macro_f1"] == pytest.approx(2 / 3)


def test_macro_f1_differs_from_binary_f1_on_skewed_predictions():
    metrics = compute_classification_metrics(
        preds=np.array([1, 1, 1, 1]), labels=np.array([1, 1, 1, 0])
    )
    assert metrics["f1"] == pytest.approx(6 / 7)
    assert metrics["macro_f1"] == pytest.approx(3 / 7)


def test_ci_half_width_for_lab_test_size():
    assert ci_half_width(0.83, 200) == pytest.approx(0.052, abs=0.001)


@pytest.mark.parametrize(
    "value, target, higher_is_better, expected",
    [
        (0.85, 0.85, True, "Đạt"),
        (0.849, 0.85, True, "Không đạt"),
        (25.0, 25.0, False, "Đạt"),
        (25.1, 25.0, False, "Không đạt"),
        (None, 25.0, False, "Không đo được"),
    ],
)
def test_kpi_status_at_boundaries(value, target, higher_is_better, expected):
    assert kpi_status(value, target, higher_is_better) == expected


@pytest.mark.parametrize("bad_text", ["", "   "])
def test_exercise_1_rejects_blank_text(bad_text):
    with pytest.raises(ValueError):
        validate_text(bad_text)
