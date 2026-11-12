"""Cấu hình, báo cáo và đo tài nguyên dùng chung; không import module Lab 2 khác."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "lab_2.yaml"


def load_config(path=DEFAULT_CONFIG):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Cấu hình YAML phải là mapping")
    names = [run["name"] for run in config["runs"]]
    if not names or len(names) != len(set(names)) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names
    ):
        raise ValueError("Tên run phải duy nhất, chỉ gồm chữ, số, _ hoặc -")
    return config


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def project_path(relative_or_absolute):
    return PROJECT_ROOT / relative_or_absolute


def read_report(config):
    path = project_path(config["paths"]["reports_dir"]) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_report(config, section, values):
    report = read_report(config)
    report[section] = values
    path = project_path(config["paths"]["reports_dir"]) / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ci_half_width(accuracy, sample_count):
    if not 0 <= accuracy <= 1 or sample_count <= 0:
        raise ValueError("Accuracy phải từ 0–1 và số mẫu phải dương")
    return 1.96 * math.sqrt(accuracy * (1 - accuracy) / sample_count)


def peak_ram_gb():
    try:
        import resource
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024**3
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024**3 if sys.platform == "darwin" else 1024**2)
