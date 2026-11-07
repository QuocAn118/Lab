# Thử Nghiệm Thực Tế (Demo Specification) — Lab 2

Mục đích: Kiểm chứng model đã huấn luyện bằng cách phân loại từng ảnh và xem xác suất của các lớp. Các lệnh chạy trong thư mục `Lab 2` trên Colab, với `PYTHONPATH=src` đã được notebook thiết lập.

Mã nguồn: `src/lab_2/demo.py`.

> **Phụ trách**: TV5. Đặc tả từng hàm cần viết nằm ở mục 4.

---

## 4. Đặc Tả Mã Nguồn (TV5)

**File cần tạo**: `src/lab_2/demo.py`. **Điều kiện trước**: Đã có đủ `common.py`, `data.py`, `model.py`, `evaluation.py`.
**Lấy từ người khác**: `build_transform`, `load_splits`, `raw_samples`, `CLASS_NAMES` (TV1); `load_checkpoint` (TV2); `SERIES`, `SURFACE`, `style_axis` (TV4).

**Đầu file**: Gọi `matplotlib.use("Agg")` trước khi import `matplotlib.pyplot`. Hằng số: `DEFAULT_SAMPLES = 3`.

Viết theo thứ tự dưới đây. Các test nằm trong `tests/test_demo.py` và `tests/test_integration.py`.

#### `load_inputs(config, paths)`
- **Trả về**: List `(tên, ảnh PIL RGB, nhãn thật)`.
- **Khi `paths` rỗng**: Lấy `DEFAULT_SAMPLES` ảnh đầu của tập test bằng `raw_samples`. Tên là `Ảnh test số 1`, nhãn là tên lớp lấy từ `CLASS_NAMES`.
- **Khi có `paths`**: Mở bằng `Image.open(path).convert("RGB")`, tên là đường dẫn, nhãn là `None`. Không đọc `config` trong nhánh này.
- **Lỗi**: Bắt `OSError` (gồm cả file không phải ảnh) và `raise SystemExit("Không đọc được ảnh ...")`.
- **Test**: `test_user_images_are_converted_to_rgb_and_have_no_label`, `test_unreadable_image_stops_with_clear_message`.

#### `plot_probabilities(results, path)`
- **Đầu vào**: List dict, chỉ dùng các khóa `image`, `prediction`, `probabilities`.
- **Cách vẽ**: Mỗi ảnh một biểu đồ cột ngang trong cùng một hàng, dùng `plt.subplots(1, n, sharey=True, squeeze=False)`. Tiêu đề gồm tên ảnh và lớp dự đoán. Trục x từ 0 đến 1, màu `SERIES[0][0]`, định dạng bằng `style_axis`. Lớp đầu tiên nằm trên cùng.
- **Test**: `test_demo_chart_uses_shared_style`.

#### `main()`
1. Tạo parser bằng `argument_parser(__doc__)`, thêm `--image` với `nargs="+"`, `default=[]`.
2. Thiếu `<checkpoints_dir>/best.pth` thì `raise SystemExit` báo cần chạy `lab_2.evaluation` trước.
3. `load_checkpoint`, `model.eval()`, `build_transform(config, train=False)`.
4. Với từng ảnh: suy luận trong `no_grad`, `torch.softmax(..., dim=1)[0]`, in dự đoán, xác suất và nhãn thật.
5. Lưu `<reports_dir>/figures/demo_probabilities.png`.
6. Ghi `update_report(config, "demo", results)`. Mỗi phần tử có `image`, `prediction`, `confidence`, `truth`, `correct` (`None` nếu không có nhãn), `probabilities` (`{tên lớp: xác suất}`).

**Tiêu chí đạt**: `python -m pytest tests/test_demo.py -q` cho `2 passed`.
