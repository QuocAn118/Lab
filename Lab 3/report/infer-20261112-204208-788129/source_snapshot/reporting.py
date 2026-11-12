"""Per-run reports, exact console capture and file integrity records.
Sources: docs/08_implementation.md §9; docs/05_evaluation.md §3;
docs/04_training.md §6; docs/11_AI_Usage.md.
"""
from contextlib import ExitStack, redirect_stdout, redirect_stderr
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import hashlib
import io
import json
import logging
import platform
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parent
SOURCES = ["docs/08_implementation.md", "docs/05_evaluation.md",
           "docs/04_training.md", "docs/11_AI_Usage.md"]


def now():
    return datetime.now().astimezone().isoformat()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Cannot record {type(value).__name__} as JSON")


class _Tee(io.TextIOBase):
    def __init__(self, original, log):
        self.original, self.log = original, log

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        if not self.log.closed:
            self.log.flush()

    def isatty(self):
        return False


class _LogFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created).astimezone().isoformat()


class RunReport:
    def __init__(self, kind, root=None, run_id=None, evidence="live"):
        self.root = Path(root) if root is not None else ROOT / "report"
        self.root.mkdir(parents=True, exist_ok=True)
        name = run_id or f"{kind}-{datetime.now():%Y%m%d-%H%M%S-%f}"
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Report ID must be one directory name")
        self.directory = self.root / name
        self.directory.mkdir(exist_ok=False)
        self.origins = {}
        self.meta = {"kind": kind, "evidence": evidence, "report_created_at": now(),
                     "status": "created", "log_scope": kind}
        self.write_text("SOURCES.md", "# Nguồn và phạm vi\n\n"
                        "Áp dụng cho toàn bộ file của report này. Nguồn tính từ Lab 3/:\n\n"
                        + "\n".join(f"- `{source}`" for source in SOURCES)
                        + "\n\nLog cũ được sao chép có trường origin trong manifest. "
                        "Không đồng nhất report được nhập lại với một lượt train mới.\n")

    def path(self, name):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Report files must stay inside the run directory")
        target = self.directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_text(self, name, text):
        self.path(name).write_text(text, encoding="utf-8", newline="")

    def write_json(self, name, values):
        payload = {"_provenance": {"sources_md": SOURCES}, **values}
        self.write_text(name, json.dumps(payload, ensure_ascii=False, indent=2,
                                        default=json_value, allow_nan=False) + "\n")

    def copy(self, source, name):
        source = Path(source)
        target = self.path(name)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite report evidence: {target}")
        shutil.copyfile(source, target)
        self.origins[target.relative_to(self.directory).as_posix()] = str(source.resolve())
        return target

    def event(self, event, **values):
        record = {"recorded_at": now(), "event": event, **values}
        with self.path("events.jsonl").open("a", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=json_value,
                                    allow_nan=False) + "\n")
            stream.flush()

    def snapshot_sources(self):
        files = [ROOT / source for source in SOURCES]
        files += [ROOT / "docs/01_lab_3_description.md", ROOT / "docs/02_data.md",
                  ROOT / "docs/03_model.md", ROOT / "lab3.py", ROOT / "reporting.py",
                  ROOT / "notebook_helpers.py", ROOT.parent / "Lab_3 (1).ipynb"]
        for source in files:
            if source.is_file():
                relative = source.relative_to(ROOT) if source.is_relative_to(ROOT) else Path(source.name)
                self.copy(source, Path("source_snapshot") / relative)

    def environment(self):
        packages = {}
        for name in ["torch", "transformers", "datasets", "accelerate", "numpy", "scikit-learn", "matplotlib"]:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        self.write_json("environment.json", {"python": platform.python_version(),
                        "executable": sys.executable, "platform": platform.platform(), "packages": packages})

    def __enter__(self):
        self.meta.update(started_at=now(), status="running")
        self.snapshot_sources()
        self.environment()
        self.stack = ExitStack()
        stdout = self.stack.enter_context(self.path("stdout.log").open("w", encoding="utf-8", newline=""))
        stderr = self.stack.enter_context(self.path("stderr.log").open("w", encoding="utf-8", newline=""))
        self.stack.enter_context(redirect_stdout(_Tee(sys.stdout, stdout)))
        self.stack.enter_context(redirect_stderr(_Tee(sys.stderr, stderr)))
        handler = logging.FileHandler(self.path("python.log"), encoding="utf-8")
        handler.setFormatter(_LogFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        self.loggers = [logging.getLogger()]
        self.loggers += [logging.getLogger(name) for name in ["transformers", "datasets", "accelerate", "torch"]
                         if not logging.getLogger(name).propagate]
        for logger in self.loggers:
            logger.addHandler(handler)
        self.handler = handler
        self.write_json("run.json", self.meta)
        self.event("run_started", kind=self.meta["kind"])
        return self

    def __exit__(self, error_type, error, error_traceback):
        try:
            if error is not None:
                self.write_json("error.json", {"type": error_type.__name__, "message": str(error)})
                self.write_text("traceback.txt", "Sources: " + "; ".join(SOURCES) + "\n"
                                + "".join(traceback.format_exception(error_type, error, error_traceback)))
            self.meta.update(ended_at=now(), status="completed" if error is None else "interrupted_or_failed")
            self.event("run_finished", status=self.meta["status"])
            self.write_json("run.json", self.meta)
        finally:
            for logger in self.loggers:
                logger.removeHandler(self.handler)
            self.handler.close()
            self.stack.close()
            self.seal()
        return False

    def seal(self):
        self.write_json("run.json", self.meta)
        lines = ["# Báo cáo lượt chạy", "", f"- Loại: `{self.meta['kind']}`",
                 f"- Trạng thái: `{self.meta['status']}`", f"- Nguồn dữ liệu: `{self.meta['evidence']}`",
                 f"- Phạm vi log: `{self.meta.get('log_scope', 'không có log console gốc')}`", "",
                 "Thông tin đầy đủ: `run.json`. Danh mục file/hash: `manifest.json`.", "",
                 "## File hiện có", ""]
        lines += [f"- `{p.relative_to(self.directory).as_posix()}`" for p in sorted(self.directory.rglob("*"))
                  if p.is_file() and "source_snapshot" not in p.parts and p.name not in {"manifest.json", "summary.md"}]
        lines += ["", "## Nguồn và phạm vi", "", "Nguồn tính từ thư mục Lab 3/: "
                  + ", ".join(f"`{source}`" for source in SOURCES) + ".", ""]
        self.write_text("summary.md", "\n".join(lines))
        entries = []
        for path in sorted(self.directory.rglob("*")):
            if path.is_file() and path.name != "manifest.json":
                name = path.relative_to(self.directory).as_posix()
                entries.append({"path": name, "sha256": file_hash(path), "bytes": path.stat().st_size,
                                "origin": self.origins.get(name, "generated"), "sources_md": SOURCES})
        self.write_json("manifest.json", {"files": entries, "note": "Manifest excludes itself."})


def verify_report(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    recorded = set()
    for entry in manifest["files"]:
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing/invalid report file: {entry['path']}")
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Report hash mismatch: {entry['path']}")
        recorded.add(entry["path"])
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != recorded:
        raise ValueError("Unrecorded files in report directory")
    return {"status": "PASS", "files_checked": len(recorded)}


def collect_output(report, output):
    """Copy existing evidence byte-for-byte; keep weight files in outputs/."""
    output = Path(output)
    names = {"run_config.json": "config.json", "metrics.json": "metrics.json",
             "epoch_history.json": "epoch_history.json", "trainer_history.json": "trainer_history.json",
             "training_time.json": "training_time.json", "checkpoint_policy.json": "checkpoint_policy.json",
             "test_predictions.csv": "predictions.csv", "confusion_matrix.png": "confusion_matrix.png",
             "loss_curve.png": "loss_curve.png", "report.md": "original_report.md",
             "lab3_source.py": "executed_cli.py", "run.log": "original_run.log",
             "failure.json": "failure.json", "provenance.json": "original_provenance.json",
             "timing_sources.md": "timing_sources.md"}
    missing = []
    for source, destination in names.items():
        path = output / source
        if path.is_file():
            report.copy(path, destination)
        else:
            missing.append(source)
    checkpoints = []
    directory = output / "checkpoints"
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if not path.is_dir():
                continue
            timing = path / "training_time.json"
            if timing.is_file():
                report.copy(timing, f"checkpoint_times/{path.name}.json")
            checkpoints.append({"path": str(path.resolve()), "config_present": (path / "config.json").is_file(),
                                "weights_present": (path / "model.safetensors").is_file(),
                                "timing_present": timing.is_file()})
    report.write_json("checkpoints.json", {"checkpoints": checkpoints, "weights_copied": False})
    report.write_json("availability.json", {"original_output": str(output.resolve()),
                      "not_present_in_original_output": missing,
                      "note": "Only files present at collection time are copied; missing logs are not reconstructed."})
