
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


LAB_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = LAB_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluate


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def create_config():
    return {
        "model": {
            "name": "MLP",
            "num_classes": 10,
        },
        "data": {
            "dataset": "FashionMNIST",
            "limits": 100,
        },
        "classes": CLASS_NAMES,
    }


class FakeDataset:
    def __len__(self):
        return 10000


class FakeLoader:
    def __init__(self):
        self.dataset = FakeDataset()

    def __iter__(self):
        for _ in range(10):
            images = torch.zeros((1000, 1, 28, 28))
            labels = torch.arange(1000) % 10
            yield images, labels


class FakeModel(nn.Module):
    """Always predicts class 0."""

    def __init__(self):
        super().__init__()
        self.eval()

    def forward(self, images):
        logits = torch.zeros((images.shape[0], 10))
        logits[:, 0] = 10.0
        return logits


class EvaluationTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.run_id = "test_run"

        self.outputs_dir = self.root / "outputs" / self.run_id
        self.checkpoint_dir = self.root / "checkpoints" / self.run_id
        self.raw_data_dir = self.root / "data" / "raw" / "FashionMNIST"
        self.reports_dir = self.root / "data" / "reports"

        self.outputs_dir.mkdir(parents=True)
        self.checkpoint_dir.mkdir(parents=True)
        self.raw_data_dir.mkdir(parents=True)
        self.reports_dir.mkdir(parents=True)

        self.config_path = self.outputs_dir / "config.json"
        self.checkpoint_path = self.checkpoint_dir / "final_epoch.pth"

        self.config_path.write_text(
            json.dumps(create_config()),
            encoding="utf-8",
        )

        self.checkpoint_path.write_bytes(b"fake checkpoint")

        split_path = self.reports_dir / "split_indices.json"
        split_path.write_text("{}", encoding="utf-8")

        self.lab_root_patch = patch.object(
            evaluate,
            "LAB_ROOT",
            self.root,
        )
        self.lab_root_patch.start()

    def tearDown(self):
        self.lab_root_patch.stop()
        self.temp_dir.cleanup()

    def test_valid_run_id(self):
        self.assertEqual(
            evaluate.validate_run_id("test_run"),
            "test_run",
        )

    def test_invalid_run_id_rejected(self):
        invalid_ids = [
            "",
            "../outside",
            "test/run",
            "test run",
            ".hidden",
        ]

        for run_id in invalid_ids:
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    evaluate.validate_run_id(run_id)

    def test_missing_required_files_reported(self):
        with self.assertRaises(FileNotFoundError):
            evaluate.check_required_files("missing_run")

    def test_config_requires_ten_classes(self):
        config = create_config()
        config["classes"] = ["class_0", "class_1"]

        with self.assertRaises(ValueError):
            evaluate.read_config(config)

    def test_config_requires_model_and_data(self):
        with self.assertRaises(ValueError):
            evaluate.read_config({"classes": CLASS_NAMES})

    def test_evaluate_all_10000_samples_and_write_reports(self):
        with patch.object(
            evaluate,
            "load_model",
            return_value=FakeModel(),
        ), patch.object(
            evaluate,
            "create_dataloaders",
            return_value=(None, None, FakeLoader()),
        ):
            metrics = evaluate.evaluate(self.run_id)

        self.assertEqual(metrics["run_id"], self.run_id)
        self.assertEqual(metrics["split"], "test")
        self.assertEqual(metrics["num_samples"], 10000)
        self.assertEqual(metrics["num_correct"], 1000)
        self.assertAlmostEqual(metrics["test_accuracy"], 0.1, places=5)
        self.assertGreaterEqual(metrics["test_loss"], 0)

        metrics_path = (
            self.root / "outputs" / self.run_id / "metrics.json"
        )
        predictions_path = (
            self.root / "outputs" / self.run_id / "predictions.csv"
        )

        self.assertTrue(metrics_path.exists())
        self.assertTrue(predictions_path.exists())

        saved_metrics = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

        self.assertEqual(saved_metrics["num_samples"], 10000)
        self.assertEqual(saved_metrics["num_correct"], 1000)

        with predictions_path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        expected_header = [
            "sample_index",
            "true_label",
            "predicted_label",
            "true_name",
            "predicted_name",
        ]

        self.assertEqual(reader.fieldnames, expected_header)
        self.assertEqual(len(rows), 10000)

        self.assertEqual(int(rows[0]["sample_index"]), 0)
        self.assertEqual(int(rows[-1]["sample_index"]), 9999)

        correct_count = sum(
            int(row["true_label"]) == int(row["predicted_label"])
            for row in rows
        )

        self.assertEqual(correct_count, 1000)

    def test_failed_evaluation_does_not_overwrite_old_reports(self):
        metrics_path = self.outputs_dir / "metrics.json"
        predictions_path = self.outputs_dir / "predictions.csv"

        old_metrics = '{"old": true}'
        old_predictions = "old,predictions"

        metrics_path.write_text(old_metrics, encoding="utf-8")
        predictions_path.write_text(old_predictions, encoding="utf-8")

        class InvalidModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.eval()

            def forward(self, images):
                return torch.full(
                    (images.shape[0], 10),
                    float("nan"),
                )

        with patch.object(
            evaluate,
            "load_model",
            return_value=InvalidModel(),
        ), patch.object(
            evaluate,
            "create_dataloaders",
            return_value=(None, None, FakeLoader()),
        ):
            with self.assertRaises(FloatingPointError):
                evaluate.evaluate(self.run_id)

        self.assertEqual(
            metrics_path.read_text(encoding="utf-8"),
            old_metrics,
        )

        self.assertEqual(
            predictions_path.read_text(encoding="utf-8"),
            old_predictions,
        )


if __name__ == "__main__":
    unittest.main()
