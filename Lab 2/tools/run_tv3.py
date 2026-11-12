"""Chạy và lưu bằng chứng TV3; không sửa test hoặc kết quả lịch sử.

python tools/run_tv3.py --stage verify
python tools/run_tv3.py --stage train
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["verify", "train"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lab_2.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    destination = args.output_dir or ROOT / "reports" / "tv3" / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    destination.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]), "PYTHONUTF8": "1",
           "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    import torch
    import torchvision
    manifest = {"started_at": datetime.now().astimezone().isoformat(), "stage": args.stage,
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "cuda": torch.cuda.is_available(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "operator": os.environ.get("LAB2_OPERATOR", "Chưa khai báo"),
                "config_path": str(args.config.resolve()), "commands": [], "sources": {}}
    for path in [*sorted((ROOT / "src/lab_2").glob("*.py")), ROOT / "model.py", args.config.resolve()]:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else Path("config.yaml")
        content = path.read_bytes()
        manifest["sources"][relative.as_posix()] = hashlib.sha256(content).hexdigest()
        snapshot = destination / "sources" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(content)
    manifest["source_snapshot"] = "sources/"
    if args.stage == "verify":
        commands = [
            ("logic", ["-m", "pytest", "tests", "test.py", "--ignore=tests/test_smoke.py", "-q"]),
            ("smoke", ["-m", "pytest", "tests/test_smoke.py", "-q", "-s"])]
    else:
        from lab_2.common import load_config, read_report
        config = load_config(args.config)
        existing = read_report(config).get("training", {}).get("runs", {})
        commands = [(run["name"], ["-m", "lab_2.training", "--config", str(args.config.resolve()), "--run", run["name"]]) for run in config["runs"]]
        for name, _ in commands:
            paths = [ROOT / config["paths"]["checkpoints_dir"] / name,
                     ROOT / config["paths"]["reports_dir"] / "runs" / name]
            if name in existing or any(path.exists() for path in paths):
                raise SystemExit(f"Refusing to overwrite existing run: {name}")
        commands.insert(0, ("data", ["-m", "lab_2.data", "--config", str(args.config.resolve())]))
    status = 0
    manifest["status"] = "running"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for label, command in commands:
        full = [sys.executable, *command]
        started = time.perf_counter()
        print(f"Running {label}; log: {destination / (label + '.log')}", flush=True)
        with (destination / (label + ".log")).open("w", encoding="utf-8") as output:
            result = subprocess.run(full, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        manifest["commands"].append({"command": full, "exit_code": result.returncode,
                                     "seconds": time.perf_counter() - started, "log": label + ".log"})
        manifest["status"] = "failed" if result.returncode else "running"
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if result.returncode:
            status = result.returncode
            break
    manifest.update(status="failed" if status else "completed",
                    finished_at=datetime.now().astimezone().isoformat())
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest["status"], destination, flush=True)
    return status


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
