# Sources: docs/docs demo/lab_3_description.md; docs/08_implementation.md §10
"""Tiện ích dùng chung của Exercise 2: đọc cấu hình, ghi báo cáo, đo RAM."""
import argparse
import json
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "exercise_2.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def config_from_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Đường dẫn file cấu hình YAML")
    return load_config(parser.parse_args().config)


def project_path(relative_or_absolute):
    return PROJECT_ROOT / relative_or_absolute  # pathlib giữ nguyên nếu vế phải là đường dẫn tuyệt đối


def _report_path(config):
    return project_path(config["paths"]["reports_dir"]) / "metrics.json"


def read_report(config):
    return json.loads(_report_path(config).read_text(encoding="utf-8"))


def update_report(config, section, values):
    """Ghi một mục vào metrics.json, giữ nguyên các mục do bước khác đã ghi."""
    path = _report_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = read_report(config) if path.exists() else {}
    report[section] = values
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def ci_half_width(accuracy, sample_count):
    """Nửa khoảng tin cậy 95% của accuracy theo xấp xỉ chuẩn."""
    return 1.96 * math.sqrt(accuracy * (1 - accuracy) / sample_count)


def peak_ram_gb():
    """RAM cao nhất của tiến trình hiện tại, tính từ lúc tiến trình khởi động."""
    try:
        import resource  # Linux/Colab
    except ImportError:
        import psutil  # Windows không có module resource

        return psutil.Process().memory_info().peak_wset / 1024**3
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
