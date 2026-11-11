"""Offline behavioral tests; the tiny random model is NOT a trained deliverable.
Sources: docs/08_implementation.md R-02..R-09; docs/06_verification.md;
docs/04_training.md; docs/05_evaluation.md; docs/11_AI_Usage.md.
Run from repository root: python "Lab 3/test.py".
"""
import json
import contextlib
import io
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset
from transformers import (DistilBertConfig, DistilBertForSequenceClassification,
                          DistilBertTokenizerFast, Trainer, TrainingArguments,
                          TrainerControl, TrainerState, DataCollatorWithPadding, set_seed)

from lab3 import (CheckpointCallback, StopPolicy, compute_metrics,
                  fresh_directory, load_demo, record_provenance, sha256,
                  stratified_indices, validate_texts, verify_artifacts)


class Lab3Tests(unittest.TestCase):
    def test_metrics_distinguish_macro_and_binary(self):
        labels = np.array([0, 0, 0, 1])
        logits = np.array([[2., 0.]] * 4)
        metrics = compute_metrics((logits, labels))
        self.assertEqual(metrics["accuracy"], .75)
        self.assertEqual(metrics["f1_binary"], 0.)
        self.assertAlmostEqual(metrics["f1_macro"], 3 / 7)

    def test_stratified_selection_reproducible_and_balanced(self):
        labels = np.array([0] * 50 + [1] * 50)
        selected = stratified_indices(labels, 20, 42)
        np.testing.assert_array_equal(selected, stratified_indices(labels, 20, 42))
        self.assertEqual(len(set(selected)), 20)
        self.assertEqual(int(labels[selected].sum()), 10)

    def test_early_stop_small_improvements_update_best_but_stop(self):
        policy = StopPolicy()
        self.assertEqual(policy.observe(.80, .50), (True, True, False))
        self.assertEqual(policy.observe(.805, .49), (True, True, False))
        self.assertEqual(policy.observe(.809, .48), (True, True, True))
        self.assertAlmostEqual(policy.best_accuracy, .809)

    def test_early_stop_absolute_threshold_and_reset(self):
        policy = StopPolicy()
        policy.observe(.80, .5)
        policy.observe(.81, .51)  # Exactly .01 is not greater than threshold.
        self.assertEqual(policy.stale, 1)
        self.assertEqual(policy.observe(.83, .49), (True, True, False))
        self.assertEqual(policy.stale, 0)
        self.assertEqual(policy.observe(.83, .49), (False, False, False))
        self.assertEqual(policy.observe(.82, .50), (False, False, True))

    def test_nonfinite_metrics_rejected(self):
        for accuracy, loss in [(float("nan"), .5), (.8, float("inf"))]:
            with self.assertRaises(ValueError):
                StopPolicy().observe(accuracy, loss)

    def test_stop_condition_at_final_step_does_not_claim_shorter_training(self):
        for max_steps, shortened in [(3750, False), (5000, True)]:
            with tempfile.TemporaryDirectory() as tmp:
                callback = CheckpointCallback(tmp, None)
                callback.epoch_start = time.perf_counter()
                control = TrainerControl()
                with patch.object(callback, "save"), contextlib.redirect_stdout(io.StringIO()):
                    for epoch, accuracy in enumerate([.8, .805, .809], 1):
                        state = SimpleNamespace(epoch=epoch, global_step=1250*epoch,
                                                max_steps=max_steps)
                        metrics = {"eval_accuracy": accuracy, "eval_loss": .5,
                                   "eval_f1_macro": .8, "eval_f1_binary": .8,
                                   "eval_precision": .8, "eval_recall": .8}
                        callback.on_evaluate(None, state, control, metrics=metrics)
                self.assertTrue(callback.history[-1]["early_stopping_triggered"])
                self.assertEqual(callback.early_stopped, shortened)
                self.assertEqual(control.should_training_stop, shortened)

    def test_empty_text_and_missing_checkpoint(self):
        for texts in [[], [" "], [None]]:
            with self.assertRaises(ValueError):
                validate_texts(texts)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_demo(tmp, ["hello"])

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            fresh_directory(path)
            (path / "existing.txt").write_text("keep me", encoding="utf-8")
            with self.assertRaises(ValueError):
                fresh_directory(path)
            self.assertEqual((path / "existing.txt").read_text(), "keep me")

    def test_manifest_tracks_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.txt"
            path.write_text("first", encoding="utf-8")
            record_provenance(Path(tmp), "offline provenance test")
            manifest = json.loads((Path(tmp) / "provenance.json").read_text(encoding="utf-8"))
            entry = next(x for x in manifest["artifacts"] if x["path"] == "evidence.txt")
            self.assertEqual(entry["sha256"], sha256(path))
            self.assertEqual(verify_artifacts(tmp)["status"], "PASS")
            path.write_text("changed", encoding="utf-8")
            self.assertNotEqual(entry["sha256"], sha256(path))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_artifacts(tmp)

    def test_tiny_trainer_gradient_checkpoint_and_reload(self):
        """True Trainer integration offline; checks loss, weights, callback, reload."""
        torch.set_num_threads(2)
        set_seed(42)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vocab = root / "vocab.txt"
            vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                        "good", "bad", "movie", "great", "boring"]), encoding="utf-8")
            tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab))
            model = DistilBertForSequenceClassification(DistilBertConfig(
                vocab_size=10, n_layers=1, n_heads=2, dim=16, hidden_dim=32,
                max_position_embeddings=32, dropout=0., attention_dropout=0.,
                seq_classif_dropout=0., num_labels=2,
                id2label={0: "NEGATIVE", 1: "POSITIVE"},
                label2id={"NEGATIVE": 0, "POSITIVE": 1}))
            before = model.classifier.weight.detach().clone()
            encoded = tokenizer(["good movie", "bad movie", "great movie", "boring movie"])
            data = Dataset.from_dict({**encoded, "label": [1, 0, 1, 0]})
            callback = CheckpointCallback(root, tokenizer)
            args = TrainingArguments(output_dir=str(root / "trainer"), use_cpu=True,
                per_device_train_batch_size=2, per_device_eval_batch_size=2,
                num_train_epochs=1, eval_strategy="epoch", save_strategy="no",
                logging_steps=1, learning_rate=.01, report_to="none", disable_tqdm=True,
                dataloader_pin_memory=False)
            trainer = Trainer(model=model, args=args, train_dataset=data, eval_dataset=data,
                processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=compute_metrics, callbacks=[callback])
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train()
            self.assertFalse(torch.equal(before, model.classifier.weight))
            self.assertTrue(all(np.isfinite(x) for x in callback.gradient_norms))
            self.assertTrue(any(x > 0 for x in callback.gradient_norms))
            self.assertEqual(len(callback.history), 1)
            for name in ["best_val_acc", "best_val_loss", "best_val_pause"]:
                self.assertTrue((root / "checkpoints" / name / "model.safetensors").is_file())
                timing = json.loads((root / "checkpoints" / name / "training_time.json").read_text(encoding="utf-8"))
                self.assertEqual(timing["checkpoint"], name)
                self.assertEqual(timing["epoch"], 1.)
                self.assertEqual(timing["step"], 2)
                self.assertGreater(timing["elapsed_train_seconds"], 0)
                self.assertGreater(timing["epoch_seconds"], 0)
                self.assertLessEqual(timing["started_at"], timing["saved_at"])
            callback.save(model, "final_epoch")
            final_time = json.loads((root / "checkpoints/final_epoch/training_time.json").read_text(encoding="utf-8"))
            self.assertEqual(final_time["elapsed_train_seconds"], callback.training_duration)
            self.assertEqual(final_time["epoch_seconds"], callback.history[-1]["epoch_wall_seconds"])
            reloaded, _, predictions = load_demo(root / "checkpoints/final_epoch", ["good movie"])
            self.assertEqual(len(predictions), 1)
            self.assertIn(predictions[0]["label"], {"NEGATIVE", "POSITIVE"})
            inputs = tokenizer("good movie", return_tensors="pt")
            model.eval()
            with torch.inference_mode():
                expected = model(**inputs).logits
                actual = reloaded(**inputs.to(reloaded.device)).logits.cpu()
            torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
