"""TV5: kiểm thử src/lab_2/common.py. Chạy: python -m pytest tests/test_common.py -q"""
import pytest

from lab_2.common import ci_half_width, load_config, read_report, update_report


def test_load_config_reads_the_lab_config():
    names = [run["name"] for run in load_config()["runs"]]
    assert len(names) == 4 and len(set(names)) == 4


def test_report_keeps_sections_written_by_other_steps(tmp_path):
    config = {"paths": {"reports_dir": str(tmp_path)}}
    assert read_report(config) == {}
    update_report(config, "data", {"a": 1})
    update_report(config, "training", {"b": 2})
    assert read_report(config) == {"data": {"a": 1}, "training": {"b": 2}}


def test_ci_half_width_for_five_thousand_samples():
    assert ci_half_width(0.8, 5000) == pytest.approx(0.01109, abs=1e-4)
