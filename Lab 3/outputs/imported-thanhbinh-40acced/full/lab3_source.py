"""Lab 3: pretrained inference and reproducible IMDB fine-tuning.

Sources: docs/implementation.md (R-01..R-09); docs/AI_Usage.md;
docs/data.md; docs/model.md; docs/training.md; docs/evaluation.md; docs/demo.md.
Technical references are registered in docs/sources.md.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback,
                          TrainingArguments, pipeline, set_seed)

ROOT = Path(__file__).resolve().parent
SOURCES = ["docs/implementation.md", "docs/AI_Usage.md", "docs/data.md",
           "docs/model.md", "docs/training.md", "docs/evaluation.md", "docs/demo.md"]
BASE_MODEL = "distilbert/distilbert-base-uncased"
INFERENCE_MODEL = "tabularisai/multilingual-sentiment-analysis"
DATASET = "stanfordnlp/imdb"
DEMO_TEXTS = [
    "This movie was an absolute masterpiece! The acting and plot were brilliant.",
    "Waste of time. The acting was terrible and the storyline made no sense.",
    "It was an average film, neither great nor awful.",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """R-09: machine-readable results include their Markdown basis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"_provenance": {"sources_md": SOURCES,
                               "generated_at_utc": utc_now()}, **value}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def fresh_directory(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class RunConfig:
    """Sources: docs/implementation.md §1–2; docs/training.md."""
    mode: str = "full"
    seed: int = 42
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 3
    weight_decay: float = 0.01

    def __post_init__(self):
        if self.mode not in {"smoke", "quick", "full"}:
            raise ValueError("mode must be smoke, quick or full")


def stratified_indices(labels, size, seed):
    """R-02: reproducible selection; never mix official train/test indexes."""
    indices = np.arange(len(labels))
    if size == len(labels):
        return indices
    selected, _ = train_test_split(indices, train_size=size,
                                   stratify=labels, random_state=seed)
    return selected


def prepare_data(config, tokenizer, dataset_revision):
    """R-02/R-03; docs/data.md and docs/implementation.md §2."""
    train, test = load_dataset(DATASET, revision=dataset_revision,
                               split=["train", "test"])
    train_idx, val_idx = train_test_split(np.arange(len(train)), test_size=0.2,
                                          stratify=train["label"],
                                          random_state=config.seed)
    sizes = {"smoke": (32, 16, 16), "quick": (1000, 200, 200),
             "full": (len(train_idx), len(val_idx), len(test))}[config.mode]
    train_idx = train_idx[stratified_indices(np.asarray(train["label"])[train_idx],
                                             sizes[0], config.seed)]
    val_idx = val_idx[stratified_indices(np.asarray(train["label"])[val_idx],
                                         sizes[1], config.seed)]
    test_idx = stratified_indices(test["label"], sizes[2], config.seed)
    if set(train_idx) & set(val_idx):
        raise AssertionError("Train and validation overlap")
    raw = DatasetDict(train=train.select(train_idx), validation=train.select(val_idx),
                      test=test.select(test_idx))
    split_info = {}
    for name, indices, official in [("train", train_idx, "train"),
                                     ("validation", val_idx, "train"),
                                     ("test", test_idx, "test")]:
        split_info[name] = {
            "official_split": official, "size": len(raw[name]),
            "class_counts": np.bincount(raw[name]["label"], minlength=2).tolist(),
            "indices_sha256": hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
            "fingerprint": raw[name]._fingerprint,
        }

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=config.max_length)

    encoded = raw.map(tokenize, batched=True, remove_columns=["text"])
    return raw, encoded, split_info


def compute_metrics(prediction):
    """R-07; docs/evaluation.md: binary and macro are distinct metrics."""
    logits, labels = prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    predicted = np.argmax(logits, axis=-1)
    precision, recall, binary_f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        labels, predicted, labels=[0, 1], average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "precision": float(precision), "recall": float(recall),
            "f1_binary": float(binary_f1), "f1_macro": float(macro_f1)}


@dataclass
class StopPolicy:
    """R-06; docs/implementation.md §3: absolute accuracy delta > .01."""
    patience: int = 2
    threshold: float = 0.01
    best_accuracy: float = -math.inf
    best_loss: float = math.inf
    stale: int = 0

    def observe(self, accuracy, loss):
        if not math.isfinite(accuracy) or not math.isfinite(loss):
            raise ValueError("Non-finite validation metric")
        accuracy_improved = accuracy > self.best_accuracy
        loss_improved = loss < self.best_loss
        meaningful = accuracy - self.best_accuracy > self.threshold + 1e-12
        self.stale = 0 if meaningful else self.stale + 1
        self.best_accuracy = max(self.best_accuracy, accuracy)
        self.best_loss = min(self.best_loss, loss)
        return accuracy_improved, loss_improved, self.stale >= self.patience


class CheckpointCallback(TrainerCallback):
    """R-05/R-06/R-07. Save inference checkpoints; track sampled resources."""
    def __init__(self, output, tokenizer):
        self.output = Path(output)
        self.tokenizer = tokenizer
        self.policy = StopPolicy()
        self.history = []
        self.gradient_norms = []
        self.epoch_start = None
        self.peak_rss = 0
        self.early_stopped = False

    def sample_resources(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)

    def save(self, model, name):
        destination = self.output / "checkpoints" / name
        model.save_pretrained(destination, safe_serialization=True)
        self.tokenizer.save_pretrained(destination)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.perf_counter()
        self.sample_resources()

    def on_step_end(self, args, state, control, **kwargs):
        self.sample_resources()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "grad_norm" in logs:
            norm = float(logs["grad_norm"])
            if not math.isfinite(norm):
                raise RuntimeError("Non-finite gradient norm")
            self.gradient_norms.append(norm)
        if logs and "loss" in logs and not math.isfinite(float(logs["loss"])):
            raise RuntimeError("Non-finite training loss")

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        # Ignore later standalone validation/test passes: this callback is train-only.
        if metrics is None or "eval_accuracy" not in metrics:
            return control
        acc, loss = float(metrics["eval_accuracy"]), float(metrics["eval_loss"])
        improved_acc, improved_loss, stop = self.policy.observe(acc, loss)
        if improved_acc:
            self.save(model, "best_val_acc")
        if improved_loss:
            self.save(model, "best_val_loss")
        self.save(model, "best_val_pause")
        self.early_stopped = stop
        if stop:
            control.should_training_stop = True
        self.sample_resources()
        row = {"epoch": float(state.epoch), "step": state.global_step,
               "val_loss": loss, "val_accuracy": acc,
               "val_f1_macro": float(metrics["eval_f1_macro"]),
               "val_f1_binary": float(metrics["eval_f1_binary"]),
               "val_precision": float(metrics["eval_precision"]),
               "val_recall": float(metrics["eval_recall"]),
               "epoch_wall_seconds": time.perf_counter() - self.epoch_start,
               "best_val_acc": improved_acc, "best_val_loss": improved_loss,
               "stale_evaluations": self.policy.stale, "early_stopped": stop}
        self.history.append(row)
        write_json(self.output / "epoch_history.json", {"epochs": self.history})
        print("EPOCH " + json.dumps(row), flush=True)
        return control


def validate_texts(texts):
    """R-01/R-08; docs/demo.md: actionable errors instead of empty input."""
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Provide one or more non-empty text strings")
    return texts


def infer_pretrained(texts):
    """R-01; docs/model.md §4: multilingual five-class Module A."""
    validate_texts(texts)
    revision = HfApi().model_info(INFERENCE_MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(INFERENCE_MODEL, revision=revision)
    classifier = pipeline("text-classification", model=INFERENCE_MODEL,
                          revision=revision, tokenizer=tokenizer,
                          device=0 if torch.cuda.is_available() else -1)
    results = classifier(texts, truncation=True, max_length=512)
    return {"model": INFERENCE_MODEL, "model_revision": revision,
            "predictions": [{"text": text, "tokens": tokenizer.tokenize(text),
                              **result} for text, result in zip(texts, results)]}


def load_demo(checkpoint, texts):
    """R-08; always reload serialized model and tokenizer from disk."""
    validate_texts(texts)
    checkpoint = Path(checkpoint)
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True).to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    results = []
    with torch.inference_mode():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=256).to(model.device)
            probs = model(**inputs).logits.softmax(-1)[0]
            label_id = int(probs.argmax())
            results.append({"text": text, "label": model.config.id2label[label_id],
                            "score": float(probs[label_id])})
    return model, tokenizer, results


def measure_latency(model, tokenizer, text, repetitions=20):
    """R-07; docs/implementation.md §5: end-to-end tokenization + forward."""
    def sync():
        if model.device.type == "cuda":
            torch.cuda.synchronize()

    timings = []
    model.eval()
    with torch.inference_mode():
        for i in range(repetitions + 3):
            sync()
            start = time.perf_counter()
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=256).to(model.device)
            model(**inputs).logits.softmax(-1)
            sync()
            if i >= 3:
                timings.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(timings)), "p50_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)), "repetitions": repetitions,
            "warmup": 3, "batch_size": 1, "token_count": inputs["input_ids"].shape[1],
            "includes_tokenization_and_transfer": True}


def environment():
    """R-09; docs/AI_Usage.md §2. Record actual installed versions."""
    return {"python": platform.python_version(), "platform": platform.platform(),
            "versions": {name: importlib.metadata.version(name) for name in
                         ["torch", "transformers", "datasets", "accelerate", "scikit-learn",
                          "numpy", "matplotlib", "psutil"]},
            "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}


def plot_results(output, trainer_history, confusion):
    """R-07; docs/evaluation.md. PNG embeds Markdown source metadata."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [x for x in trainer_history if "loss" in x]
    val = [x for x in trainer_history if "eval_loss" in x]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([x["step"] for x in train], [x["loss"] for x in train], label="Train")
    ax.plot([x["step"] for x in val], [x["eval_loss"] for x in val], "o-", label="Validation")
    ax.set(xlabel="Optimizer step", ylabel="Cross-entropy loss", title="IMDB / DistilBERT")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "loss_curve.png", dpi=160, metadata={"Sources": "; ".join(SOURCES)})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion, cmap="Blues")
    for (i, j), value in np.ndenumerate(confusion):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["NEG", "POS"],
           yticklabels=["NEG", "POS"], xlabel="Predicted", ylabel="Actual", title="Held-out test")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=160, metadata={"Sources": "; ".join(SOURCES)})
    plt.close(fig)


def render_report(output, result):
    """R-07/R-09. Derived only from this run's measured result."""
    metrics = result["test_metrics"]
    kpis = result["kpis"]
    text = ["# Lab 3 — báo cáo thực nghiệm", "",
            f"Mode: **{result['config']['mode']}**; thiết bị: {result['environment']['device']}.",
            "Smoke/quick không thay thế nghiệm thu full. Test không dùng chọn checkpoint.", "",
            "| Chỉ số test | Giá trị |", "| --- | ---: |"]
    text += [f"| {k} | {v:.6f} |" for k, v in metrics.items() if isinstance(v, (float, int))]
    text += ["", "| Tiêu chí | Kết quả |", "| --- | --- |"]
    text += [f"| {k} | {v} |" for k, v in kpis.items()]
    text += ["", "## Epoch", "", "| Epoch | Val loss | Val accuracy | Macro F1 | Wall seconds |",
             "| ---: | ---: | ---: | ---: | ---: |"]
    text += [f"| {x['epoch']:.2f} | {x['val_loss']:.6f} | {x['val_accuracy']:.4f} | "
             f"{x['val_f1_macro']:.4f} | {x['epoch_wall_seconds']:.2f} |" for x in result["epochs"]]
    text += ["", "![Loss](loss_curve.png)", "", "![Confusion matrix](confusion_matrix.png)", "",
             "Chi tiết số đo, phân lớp, latency, RAM/GPU và cấu hình: [metrics.json](metrics.json).",
             "Nguồn/hash từng artifact: [artifacts.md](artifacts.md).", "", "## Nguồn và phạm vi", "",
             "- Đặc tả: `docs/implementation.md`, `docs/evaluation.md`, `docs/AI_Usage.md` (tính từ Lab 3/).",
             "- Dữ liệu trực tiếp: `metrics.json`, `epoch_history.json`, `trainer_history.json` cùng run.",
             "- Sinh bằng `lab3.py::render_report`; không sao chép số từ run khác.", ""]
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")


def record_provenance(output, command):
    """R-09: hash every artifact, including native library/binary formats."""
    output = Path(output)
    source_hashes = {s: sha256(ROOT / s) for s in SOURCES}
    # Preserve the exact basis even when Markdown or code changes after this run.
    source_texts = {s: (ROOT / s).read_bytes().decode("utf-8") for s in SOURCES}
    shutil.copyfile(Path(__file__), output / "lab3_source.py")
    entries = []
    for file in sorted(output.rglob("*")):
        if file.is_file() and file.name not in {"provenance.json", "artifacts.md"}:
            entries.append({"path": file.relative_to(output).as_posix(),
                            "sha256": sha256(file), "bytes": file.stat().st_size,
                            "sources_md": SOURCES})
    write_json(output / "provenance.json", {"command": command,
               "source_root": str(ROOT), "source_sha256": source_hashes,
               "source_snapshots": source_texts,
               "code_sha256": sha256(Path(__file__)), "artifacts": entries,
               "manifest_note": "artifacts.md indexes this manifest; manifests exclude themselves to avoid recursive hashes"})
    lines = ["# Nguồn artifact của run", "", f"Lệnh: `{command}`", "",
             "SHA-256, kích thước và nguồn từng file: [provenance.json](provenance.json).",
             "Manifest không tự hash chính nó; artifacts.md là chỉ mục, tránh chuỗi hash đệ quy.", "",
             "## Nguồn và phạm vi", "",
             "Nguồn gốc tính từ thư mục Lab 3/: " + ", ".join(f"`{s}`" for s in SOURCES) + ".",
             "Trọng số/cấu hình/tokenizer do thư viện lưu có định dạng nguyên bản; nguồn của chúng ghi ở manifest.", ""]
    (output / "artifacts.md").write_text("\n".join(lines), encoding="utf-8")


def verify_artifacts(output):
    """R-09; verify immutable run evidence before reuse/reporting."""
    output = Path(output).resolve()
    manifest = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        file = (output / entry["path"]).resolve()
        if not file.is_relative_to(output) or not file.is_file():
            raise ValueError(f"Missing or invalid artifact: {entry['path']}")
        if sha256(file) != entry["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {entry['path']}")
    for source, snapshot in manifest.get("source_snapshots", {}).items():
        if hashlib.sha256(snapshot.encode("utf-8")).hexdigest() != manifest["source_sha256"][source]:
            raise ValueError(f"Source snapshot hash mismatch: {source}")
    return {"status": "PASS", "artifacts_checked": len(manifest["artifacts"])}


def train_run(config, output):
    """Sources: docs/implementation.md R-02..R-09; docs/training.md."""
    set_seed(config.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    env = environment()
    model_revision = HfApi().model_info(BASE_MODEL).sha
    dataset_revision = HfApi().dataset_info(DATASET).sha
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=model_revision)
    raw, encoded, splits = prepare_data(config, tokenizer, dataset_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=model_revision, num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1})
    callback = CheckpointCallback(output, tokenizer)
    write_json(output / "run_config.json", {"config": asdict(config), "splits": splits,
               "environment": env, "model": BASE_MODEL, "model_revision": model_revision,
               "dataset": DATASET, "dataset_revision": dataset_revision})
    args = TrainingArguments(
        output_dir=str(output / "trainer"), eval_strategy="epoch", save_strategy="no",
        learning_rate=config.learning_rate, per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size, num_train_epochs=config.epochs,
        max_steps=2 if config.mode == "smoke" else -1, weight_decay=config.weight_decay,
        max_grad_norm=1.0, optim="adamw_torch", fp16=torch.cuda.is_available(),
        logging_steps=1 if config.mode == "smoke" else 50, logging_first_step=True,
        report_to="none", seed=config.seed, data_seed=config.seed,
        dataloader_num_workers=0, dataloader_pin_memory=torch.cuda.is_available(),
        disable_tqdm=True, log_level="error", save_safetensors=True)
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"],
                       eval_dataset=encoded["validation"], processing_class=tokenizer,
                       data_collator=DataCollatorWithPadding(tokenizer,
                           pad_to_multiple_of=8 if torch.cuda.is_available() else None),
                       compute_metrics=compute_metrics, callbacks=[callback])
    print("RUN " + json.dumps({"config": asdict(config), "splits": splits,
                                 "environment": env}), flush=True)
    start = time.perf_counter()
    training = trainer.train()
    train_seconds = time.perf_counter() - start
    if not callback.history or not callback.gradient_norms or not any(n > 0 for n in callback.gradient_norms):
        raise RuntimeError("No validation or finite nonzero gradient evidence")
    callback.save(trainer.model, "final_epoch")
    shutil.copytree(output / "checkpoints/best_val_acc", output / "checkpoints/best")
    write_json(output / "checkpoint_policy.json", {"policy": asdict(callback.policy),
               "early_stopped": callback.early_stopped, "best_source": "best_val_acc",
               "best_val_pause_means": "last validation model; see early_stopped flag"})
    write_json(output / "trainer_history.json", {"history": trainer.state.log_history})
    # Free optimizer + old model before reloading best; final_epoch is already saved.
    trainer.remove_callback(callback)
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.model.to("cpu")
    model, tokenizer, demo_results = load_demo(output / "checkpoints/best", DEMO_TEXTS)
    evaluator = Trainer(model=model, args=args, processing_class=tokenizer,
                        eval_dataset=encoded["validation"],
                        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    validation = evaluator.evaluate(encoded["validation"], metric_key_prefix="validation")
    prediction = evaluator.predict(encoded["test"], metric_key_prefix="test")
    predicted = prediction.predictions.argmax(-1)
    matrix = confusion_matrix(prediction.label_ids, predicted, labels=[0, 1])
    latency = measure_latency(model, tokenizer, raw["test"][0]["text"])
    peak_gpu = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
    callback.sample_resources()
    ram_gib = callback.peak_rss / 1024**3
    with (output / "test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("# Sources: docs/data.md; docs/evaluation.md; docs/implementation.md R-07\n")
        writer = csv.writer(f)
        writer.writerow(["test_subset_index", "actual", "predicted"])
        writer.writerows(zip(range(len(predicted)), prediction.label_ids.tolist(), predicted.tolist()))
    result = {"config": asdict(config), "environment": env, "splits": splits,
              "model_revision": model_revision, "dataset_revision": dataset_revision,
              "training": {**training.metrics, "wall_seconds": train_seconds,
                            "total_parameters": sum(p.numel() for p in model.parameters()),
                            "gradient_norms": callback.gradient_norms},
              "epochs": callback.history, "validation_metrics": validation,
              "test_metrics": prediction.metrics, "confusion_matrix": matrix.tolist(),
              "classification_report": classification_report(prediction.label_ids, predicted,
                  labels=[0, 1], target_names=["NEGATIVE", "POSITIVE"], output_dict=True, zero_division=0),
              "latency": latency, "gpu_peak_allocated_gib": peak_gpu,
              "host_peak_observed_rss_gib": ram_gib, "demo": demo_results,
              "early_stopped": callback.early_stopped}
    quality = config.mode == "full"
    result["kpis"] = {
        "test_accuracy_ge_0.85": bool(prediction.metrics["test_accuracy"] >= .85) if quality else "N/A: non-full run",
        "test_macro_f1_ge_0.85": bool(prediction.metrics["test_f1_macro"] >= .85) if quality else "N/A: non-full run",
        "validation_accuracy_ge_0.85": bool(validation["validation_accuracy"] >= .85) if quality else "N/A: non-full run",
        "latency_mean_le_25ms": latency["mean_ms"] <= 25,
        "gpu_peak_le_4GiB": peak_gpu <= 4 if peak_gpu is not None else "N/A: CPU",
        "host_observed_rss_le_6GiB": ram_gib <= 6,
        "T4_epoch_le_180s": (max(x["epoch_wall_seconds"] for x in callback.history) <= 180)
            if "T4" in env["device"] else "N/A: device is not Colab T4"}
    write_json(output / "metrics.json", result)
    plot_results(output, trainer.state.log_history, matrix)
    render_report(output, result)
    print("RESULT " + json.dumps({"test": prediction.metrics, "kpis": result["kpis"]}), flush=True)
    return result


def audit_sources():
    """R-09; docs/AI_Usage.md: enforce file-level Markdown references."""
    import re
    errors, count = [], 0
    for file in sorted(ROOT.rglob("*")):
        relative = file.relative_to(ROOT)
        if any(part in {".venv", "outputs", "__pycache__", ".ipynb_checkpoints"} for part in relative.parts):
            continue
        if not file.is_file() or file.suffix not in {".md", ".py", ".txt"} and file.name != ".gitignore":
            continue
        count += 1
        text = file.read_text(encoding="utf-8-sig")
        if file.suffix == ".md":
            if "Nguồn và phạm vi" not in text:
                errors.append(f"{relative}: missing Nguồn và phạm vi")
        elif "Sources:" not in text[:2000]:
            errors.append(f"{relative}: missing Sources header")
        # Validate explicit docs/*.md references in headers/text.
        for ref in re.findall(r"docs/[A-Za-z0-9_/-]+\.md", text):
            if not (ROOT / ref).is_file():
                errors.append(f"{relative}: missing source {ref}")
        if file.suffix == ".md":
            for ref in re.findall(r"\]\(([^)]+\.md)(?:#[^)]*)?\)", text):
                if "://" not in ref and not (file.parent / ref).is_file():
                    errors.append(f"{relative}: broken Markdown link {ref}")
    notebook = json.loads((ROOT.parent / "Lab_3 (1).ipynb").read_text(encoding="utf-8"))
    if not notebook.get("metadata", {}).get("provenance", {}).get("sources_md"):
        errors.append("Notebook: missing provenance metadata")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "Sources:" not in "".join(cell["source"])[:600]:
            errors.append(f"Notebook cell {index}: missing Sources")
    if errors:
        raise ValueError("\n".join(errors))
    return {"files_checked": count, "notebook_code_cells_checked":
            sum(c["cell_type"] == "code" for c in notebook["cells"]), "status": "PASS"}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="smoke/quick/full; see docs/implementation.md")
    train.add_argument("--mode", choices=["smoke", "quick", "full"], default="full")
    train.add_argument("--output", type=Path)
    infer = sub.add_parser("infer", help="Module A: pretrained multilingual sentiment")
    infer.add_argument("--text", action="append", required=True)
    infer.add_argument("--output", type=Path)
    demo = sub.add_parser("demo", help="Module B: reload local best checkpoint")
    demo.add_argument("--checkpoint", type=Path, required=True)
    demo.add_argument("--text", action="append")
    audit = sub.add_parser("audit", help="Check Markdown source traceability")
    audit.add_argument("--run", type=Path, help="Also verify an existing run's artifact hashes")
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit_sources(), indent=2))
        if args.run:
            print(json.dumps(verify_artifacts(args.run), indent=2))
    elif args.command == "infer":
        result = infer_pretrained(args.text)
        if args.output:
            output = fresh_directory(args.output)
            write_json(output / "inference.json", result)
            record_provenance(output, "lab3.py " + " ".join(sys.argv[1:]))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "demo":
        _, _, result = load_demo(args.checkpoint, args.text or DEMO_TEXTS)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        output = fresh_directory(args.output or ROOT / "outputs" /
                                  f"{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
        command = "lab3.py " + " ".join(sys.argv[1:])
        try:
            with (output / "run.log").open("w", encoding="utf-8") as log:
                with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                    print("Sources: " + "; ".join(SOURCES), flush=True)
                    train_run(RunConfig(mode=args.mode), output)
        except Exception as exc:
            write_json(output / "failure.json", {"exception": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            record_provenance(output, command)
        print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
