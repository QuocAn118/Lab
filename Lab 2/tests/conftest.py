"""Fixture dùng chung. Đọc YAML trực tiếp để test của mỗi thành viên không phụ thuộc common.py."""
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs" / "lab_2.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
