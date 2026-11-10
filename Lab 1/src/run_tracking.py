"""Lưu vết mọi lần chạy và mọi file đầu ra của Lab 1.

Spec: docs/team_plan_integrity.md §9.3
Spec: docs/lab_1_description.md §10

Cách dùng trong script huấn luyện:

    from run_tracking import Run

    with Run("run_001", purpose="Cấu hình cơ sở", config=config,
             specs=["docs/requirements/3_4_train.md"]) as run:
        ...  # ghi file vào run.out_dir và run.ckpt_dir
        run.note("epoch 3: val_acc 0.87")

Khi khối `with` kết thúc (kể cả lỗi hoặc Ctrl+C), manifest.json được chốt với
trạng thái completed / failed / interrupted và hash của mọi file đã tạo.

File đầu ra không thuộc một run (ví dụ outputs/data_summary.json):

    record_artifact(path, specs=["docs/requirements/3_1_prepare_data.md"])
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

LAB_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = LAB_ROOT / "outputs"
CHECKPOINTS = LAB_ROOT / "checkpoints"
REGISTRY = LAB_ROOT / "docs" / "provenance.json"

SCHEMA_VERSION = 1
RUN_ID_RE = re.compile(r"^(run|smoke)_\d{3}$")
# Chuẩn hóa xuống dòng trước khi hash để Windows (CRLF) và Linux/macOS (LF) cho cùng kết quả.
TEXT_SUFFIXES = {".md", ".py", ".json", ".csv", ".txt", ".yaml", ".yml", ".log", ""}
MANIFEST = "manifest.json"
SIDECAR_SUFFIX = ".prov.json"


# ---------------------------------------------------------------- tiện ích chung
def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def rel(path: Path | str) -> str:
    """Đường dẫn tương đối từ thư mục Lab 1, dạng dấu '/'."""
    p = Path(path).resolve()
    try:
        return p.relative_to(LAB_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def sha256_file(path: Path | str) -> str:
    p = Path(path)
    data = p.read_bytes()
    if p.suffix.lower() in TEXT_SUFFIXES:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=LAB_ROOT, capture_output=True,
                             text=True, encoding="utf-8", check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info() -> dict:
    status = _git("status", "--porcelain", "--", ".")
    dirty = [line for line in (status or "").splitlines() if line.strip()]
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": dirty,
    }


def current_user() -> str:
    return os.environ.get("LAB_USER") or _git("config", "user.name") or getpass.getuser()


def environment() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.node(),
        "processor": platform.processor(),
    }
    torch = sys.modules.get("torch")
    if torch is None:
        try:
            import torch  # noqa: F401  (chỉ đọc phiên bản nếu đã cài)
        except ImportError:
            torch = None
    if torch is not None:
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    try:
        import torchvision
        env["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    return env


def spec_hashes(specs: Iterable[str]) -> list[dict]:
    result = []
    for spec in specs:
        path = LAB_ROOT / spec
        if not path.is_file():
            raise FileNotFoundError(f"Không thấy file đặc tả: {spec}")
        result.append({"path": rel(path), "sha256": sha256_file(path)})
    return result


def code_snapshot() -> list[dict]:
    """Hash mọi file code trong src/ kèm đặc tả nguồn ghi trong sổ đăng ký."""
    sources: dict[str, list[str]] = {}
    if REGISTRY.is_file():
        for entry in json.loads(REGISTRY.read_text(encoding="utf-8")).get("entries", []):
            sources[entry["path"]] = [s["path"] for s in entry.get("sources", []) if "path" in s]
    snapshot = []
    for path in sorted([*(LAB_ROOT / "src").glob("*.py"), *(LAB_ROOT / "data").glob("*.py")]):
        key = rel(path)
        snapshot.append({"path": key, "sha256": sha256_file(path),
                         "specs": sources.get(key, []), "registered": key in sources})
    return snapshot


def _ignored_by_git(paths: list[str]) -> set[str]:
    if not paths:
        return set()
    try:
        out = subprocess.run(["git", "check-ignore", "--stdin"], cwd=LAB_ROOT,
                             input="\n".join(paths), capture_output=True,
                             text=True, encoding="utf-8")
        return {line.strip() for line in out.stdout.splitlines() if line.strip()}
    except OSError:
        return set()


def list_files(*folders: Path, exclude: set[str] = frozenset()) -> list[dict]:
    files = [p for folder in folders if folder.is_dir()
             for p in sorted(folder.rglob("*")) if p.is_file()]
    keys = [rel(p) for p in files]
    keys_files = [(k, p) for k, p in zip(keys, files)
                  if k not in exclude and not k.endswith(".tmp")]
    ignored = _ignored_by_git([k for k, _ in keys_files])
    return [{"path": k, "bytes": p.stat().st_size, "sha256": sha256_file(p),
             "ignored_by_git": k in ignored} for k, p in keys_files]


def _safe_print(text: str, file=None) -> None:
    """In ra console không bị lỗi khi console Windows không hỗ trợ tiếng Việt."""
    stream = file or sys.stdout
    try:
        print(text, file=stream)
    except UnicodeEncodeError:
        enc = getattr(stream, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc), file=stream)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- lưu vết một run
class Run:
    """Tạo outputs/<run_id>/ và checkpoints/<run_id>/, ghi manifest từ đầu đến cuối run."""

    def __init__(self, run_id: str, purpose: str, config: dict | None = None,
                 specs: Iterable[str] = (), allow_dirty: bool = False):
        if not RUN_ID_RE.match(run_id):
            raise ValueError(f"run_id '{run_id}' sai mẫu; dùng run_001 hoặc smoke_001")
        if not purpose.strip():
            raise ValueError("Cần ghi mục đích (purpose) của lần chạy")
        self.run_id = run_id
        self.purpose = purpose
        self.config = config
        self.specs = list(specs)
        self.allow_dirty = allow_dirty
        self.out_dir = OUTPUTS / run_id
        self.ckpt_dir = CHECKPOINTS / run_id
        self.manifest_path = self.out_dir / MANIFEST
        self.manifest: dict = {}
        self._t0 = 0.0

    def __enter__(self) -> "Run":
        if self.out_dir.exists() or self.ckpt_dir.exists():
            raise FileExistsError(f"{self.run_id} đã tồn tại; chọn run_id mới, không ghi đè")
        git = git_info()
        is_normal = self.run_id.startswith("run_")
        if is_normal and git["dirty"] and not self.allow_dirty:
            raise RuntimeError(
                "Normal run phải chạy từ code đã commit. Hãy commit các thay đổi sau "
                "rồi chạy lại:\n  " + "\n  ".join(git["dirty_files"]))
        if git["commit"] is None:
            _safe_print("[run_tracking] Cảnh báo: không đọc được git commit", file=sys.stderr)

        self.out_dir.mkdir(parents=True)
        self.ckpt_dir.mkdir(parents=True)
        started = now_iso()
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "kind": "normal" if is_normal else "smoke",
            "purpose": self.purpose,
            "status": "running",
            "created_by": current_user(),
            "started_at": started,
            "finished_at": None,
            "duration_seconds": None,
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]] if sys.argv and sys.argv[0] else [],
            "git": git,
            "allow_dirty": self.allow_dirty,
            "specs": spec_hashes(self.specs),
            "code": code_snapshot(),
            "environment": environment(),
            "config_sha256": None,
            "notes": [],
            "error": None,
            "files": [],
        }
        if self.config is not None:
            config = {"run_id": self.run_id, "created_by": self.manifest["created_by"],
                      "started_at": started, **self.config}
            config_path = self.out_dir / "config.json"
            write_json(config_path, config)
            self.manifest["config_sha256"] = sha256_file(config_path)
        write_json(self.manifest_path, self.manifest)
        self._t0 = time.perf_counter()
        return self

    def note(self, text: str) -> None:
        """Ghi một sự kiện vào manifest ngay lập tức (giữ lại kể cả khi run bị kill)."""
        self.manifest["notes"].append({"at": now_iso(), "text": text})
        write_json(self.manifest_path, self.manifest)

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            status = "completed"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "interrupted"
        else:
            status = "failed"
        self.manifest.update({
            "status": status,
            "finished_at": now_iso(),
            "duration_seconds": round(time.perf_counter() - self._t0, 3),
        })
        if exc_type is not None:
            self.manifest["error"] = {
                "type": exc_type.__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exception(exc_type, exc, tb)[-3:],
            }
        self.manifest["files"] = list_files(self.out_dir, self.ckpt_dir,
                                            exclude={rel(self.manifest_path)})
        write_json(self.manifest_path, self.manifest)
        _safe_print(f"[run_tracking] {self.run_id}: {status} - {rel(self.manifest_path)}")
        return False  # không nuốt lỗi


# ---------------------------------------------------------------- file đầu ra lẻ
def record_artifact(path: Path | str, specs: Iterable[str],
                    inputs: Iterable[Path | str] = (), note: str = "") -> Path:
    """Ghi <file>.prov.json bên cạnh một file đầu ra không thuộc run nào."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_name(path.name + SIDECAR_SUFFIX)
    history = []
    if sidecar.is_file():  # file được tạo lại: giữ lịch sử các lần trước
        old = json.loads(sidecar.read_text(encoding="utf-8"))
        history = old.pop("history", [])
        history.append(old)
    data = {
        "schema_version": SCHEMA_VERSION,
        "artifact": rel(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "produced_by": rel(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "created_by": current_user(),
        "created_at": now_iso(),
        "git": git_info(),
        "specs": spec_hashes(specs),
        "inputs": [{"path": rel(p), "sha256": sha256_file(p)} for p in map(Path, inputs) if p.is_file()],
        "note": note,
        "history": history,
    }
    write_json(sidecar, data)
    return sidecar
