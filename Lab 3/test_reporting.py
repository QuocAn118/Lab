"""Behavioral checks for reports; tiny random models are test fixtures only.
Sources: docs/08_implementation.md §9; docs/05_evaluation.md §3;
docs/04_training.md §6; docs/11_AI_Usage.md.
"""
import contextlib
import csv
import io
import json
import logging
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from reporting import RunReport, collect_output, verify_report


class ReportTests(unittest.TestCase):
    def test_exact_streams_events_and_integrity(self):
        with tempfile.TemporaryDirectory(prefix="lab3-report-test-") as folder:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with RunReport("test", root=folder) as report:
                    print("Tiếng Việt\rtiến trình", end="")
                    print("cảnh báo", file=sys.stderr)
                    logging.warning("recorded warning")
                    report.event("sample", step=1)
            self.assertEqual(report.path("stdout.log").read_bytes(), "Tiếng Việt\rtiến trình".encode())
            self.assertEqual(report.path("stderr.log").read_bytes(), "cảnh báo\n".encode())
            self.assertIn("recorded warning", report.path("python.log").read_text(encoding="utf-8"))
            events = [json.loads(line) for line in report.path("events.jsonl").read_text().splitlines()]
            self.assertEqual([e["event"] for e in events], ["run_started", "sample", "run_finished"])
            self.assertTrue(all(e["recorded_at"] for e in events))
            self.assertEqual(verify_report(report.directory)["status"], "PASS")
            report.path("stdout.log").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_report(report.directory)

    def test_interrupt_records_error_and_restores_streams(self):
        with tempfile.TemporaryDirectory(prefix="lab3-report-test-") as folder:
            original = sys.stdout
            with self.assertRaises(KeyboardInterrupt):
                with RunReport("test", root=folder) as report:
                    raise KeyboardInterrupt("test interruption")
            self.assertIs(sys.stdout, original)
            run = json.loads(report.path("run.json").read_text(encoding="utf-8"))
            self.assertEqual(run["status"], "interrupted_or_failed")
            self.assertIn("KeyboardInterrupt", report.path("traceback.txt").read_text())
            self.assertEqual(verify_report(report.directory)["status"], "PASS")

    def test_import_preserves_bytes_and_does_not_invent_logs(self):
        with tempfile.TemporaryDirectory(prefix="lab3-report-test-") as folder:
            source = Path(folder) / "old-output"
            source.mkdir()
            content = b'old console\r\nexact bytes\r'
            (source / "run.log").write_bytes(content)
            report = RunReport("import", root=Path(folder)/"reports", evidence="copied_existing_output")
            report.meta.update(status="imported", started_at=None, ended_at=None)
            collect_output(report, source)
            report.seal()
            self.assertEqual(report.path("original_run.log").read_bytes(), content)
            self.assertFalse(report.path("stdout.log").exists())
            self.assertEqual(verify_report(report.directory)["status"], "PASS")
            report.path("unexpected.txt").write_text("extra")
            with self.assertRaisesRegex(ValueError, "Unrecorded"):
                verify_report(report.directory)

    def test_cli_interrupt_keeps_report_and_output_provenance(self):
        import lab3
        with tempfile.TemporaryDirectory(prefix="lab3-report-test-") as folder:
            reports = Path(folder)/"reports"
            def interrupt(config, output, report=None):
                report.event("before_interruption")
                raise KeyboardInterrupt("fixture")
            with patch.object(lab3, "RunReport", side_effect=lambda kind: RunReport(kind, root=reports)):
                with patch.object(lab3, "train_run", side_effect=interrupt):
                    with self.assertRaises(KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
                        lab3.main(["train", "--mode", "smoke", "--output", str(Path(folder)/"output")])
            directory = next(reports.iterdir())
            self.assertEqual(verify_report(directory)["status"], "PASS")
            self.assertEqual(lab3.verify_artifacts(Path(folder)/"output")["status"], "PASS")
            self.assertEqual(json.loads((directory/"failure.json").read_text())["exception"], "KeyboardInterrupt")

    def test_real_tiny_training_evaluation_and_failed_training(self):
        import matplotlib
        matplotlib.use("Agg")
        import torch
        from datasets import Dataset
        from transformers import (DistilBertConfig, DistilBertForSequenceClassification,
                                  DistilBertTokenizerFast, Trainer, TrainingArguments, TrainerCallback,
                                  DataCollatorWithPadding)
        from notebook_helpers import train_and_save, evaluate_and_report
        from lab3 import compute_metrics
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="lab3-report-test-") as folder:
            root = Path(folder)
            (root/"vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nbad\nmovie\n")
            tokenizer = DistilBertTokenizerFast(vocab_file=str(root/"vocab.txt"), model_max_length=256)
            data = Dataset.from_dict({**tokenizer(["good movie", "bad movie"]*2), "labels": [1, 0]*2})
            def make_trainer(callbacks=None):
                model = DistilBertForSequenceClassification(DistilBertConfig(
                    vocab_size=8, n_layers=1, dim=16, hidden_dim=32, n_heads=2,
                    dropout=0., attention_dropout=0., seq_classif_dropout=0.,
                    id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1}))
                args = TrainingArguments(str(root/"unused"), use_cpu=True, num_train_epochs=3,
                    per_device_train_batch_size=2, per_device_eval_batch_size=2, learning_rate=.001,
                    eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
                    load_best_model_at_end=True, metric_for_best_model="accuracy", logging_steps=1,
                    disable_tqdm=True, report_to="none", dataloader_pin_memory=False)
                return Trainer(model=model, args=args, train_dataset=data, eval_dataset=data,
                    processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                    compute_metrics=compute_metrics, callbacks=callbacks)
            trainer = make_trainer()
            with contextlib.redirect_stdout(io.StringIO()):
                checkpoint = train_and_save(trainer, tokenizer, root/"models", report_root=root/"reports")
                evaluate_and_report(trainer, data, checkpoint, root, True,
                                    sentences=["good movie", "bad movie"], report_root=root/"reports")
            training = json.loads((checkpoint/"training_time.json").read_text())
            self.assertEqual(len(training["epochs"]), 3)
            directories = list((root/"reports").iterdir())
            evaluation = next(p for p in directories if p.name.startswith("evaluate-"))
            with (evaluation/"predictions.csv").open(encoding="utf-8") as stream:
                next(stream)
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual([int(row["actual"]) for row in rows], [1, 0, 1, 0])
            self.assertTrue((evaluation/"confusion_matrix.png").is_file())
            self.assertTrue((evaluation/"loss_curve.png").is_file())
            class Interrupt(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    raise KeyboardInterrupt("tiny training interrupted")
            failed = make_trainer([Interrupt()])
            with self.assertRaises(KeyboardInterrupt), contextlib.redirect_stdout(io.StringIO()):
                train_and_save(failed, tokenizer, root/"models", report_root=root/"reports")
            all_reports = list((root/"reports").iterdir())
            self.assertEqual(len(all_reports), 3)
            for directory in all_reports:
                self.assertEqual(verify_report(directory)["status"], "PASS")
            self.assertFalse(any(type(c).__name__ == "TrainingTime" for c in failed.callback_handler.callbacks))
            failed_report = next(p for p in all_reports if (p/"error.json").exists())
            self.assertEqual(json.loads((failed_report/"training_time.json").read_text())["status"], "interrupted_or_failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
