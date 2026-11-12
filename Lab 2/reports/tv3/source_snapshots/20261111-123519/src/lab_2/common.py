"""Tiện ích ghép TV3 theo docs/lab_2_description.md §9; Codex, 2026-11-11."""
import argparse
import json
import math
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/lab_2.yaml"


def load_config(path=DEFAULT_CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser


def project_path(relative_or_absolute):
    return PROJECT_ROOT / relative_or_absolute


def read_report(config):
    path = project_path(config["paths"]["reports_dir"]) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_report(config, section, values):
    path = project_path(config["paths"]["reports_dir"]) / "metrics.json"
    report = read_report(config)
    report[section] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def ci_half_width(accuracy, sample_count):
    return 1.96 * math.sqrt(accuracy * (1 - accuracy) / sample_count)


def peak_ram_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024**3
