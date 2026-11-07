"""TV5: kiểm thử src/lab_2/demo.py. Chạy: python -m pytest tests/test_demo.py -q"""
import pytest
from PIL import Image

from lab_2.demo import load_inputs


def test_user_images_are_converted_to_rgb_and_have_no_label(tmp_path):
    path = tmp_path / "anh.png"
    Image.new("L", (10, 10)).save(path)
    (name, image, truth), = load_inputs({}, [str(path)])
    assert (name, image.mode, truth) == (str(path), "RGB", None)


def test_unreadable_image_stops_with_clear_message(tmp_path):
    with pytest.raises(SystemExit, match="Không đọc được"):
        load_inputs({}, [str(tmp_path / "khong_co.jpg")])
