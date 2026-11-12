# Thử Nghiệm Thực Tế (Demo Specification) — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Thử Nghiệm Thực Tế (Demo Specification) — Lab 2 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T11:13:30+07:00 |
| Updated by — nguồn đặc tả/test nền | Nguyễn Quốc An (Quoc An), `3a86e93`, `00b8d04`, `d0cd42f` |
| Người tích hợp/chuẩn hóa | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Mục đích: Kiểm chứng model đã huấn luyện bằng cách phân loại từng ảnh và xem xác suất của các lớp. Các lệnh chạy trong thư mục `Lab 2` trên Colab, với `PYTHONPATH=src` đã được notebook thiết lập.

Mã nguồn: `src/lab_2/demo.py`.

Nguồn đặc tả demo và bộ test nền: Nguyễn Quốc An (Quoc An). Phần triển khai package/demo do HoangDang chuẩn bị tại `4176139` được ghi riêng, không quy cho người viết đặc tả. Ghi nhận bổ sung do HoangDang đối chiếu Git ngày 2026-11-12.

> **Phụ trách**: TV5. Đặc tả từng hàm đã triển khai nằm ở mục 4.

> Đã demo thật ba ảnh test bằng DenseNet121 đã chọn trong phiên `rerun-thanhbinh-20261112-175308`; [ảnh xác suất](../reports/sessions/rerun-thanhbinh-20261112-175308/reports/figures/demo_probabilities.png), [metrics/demo](../reports/sessions/rerun-thanhbinh-20261112-175308/reports/metrics.json). Người chạy HoangDang, RTX 5060; code/đặc tả nguồn giữ tên riêng. Dùng `--config` trỏ tới config phiên khi đọc model đã train, tránh nhầm checkpoint nguồn của Bình.

## 1. Cách Chạy

```bash
# Mặc định: 3 ảnh đầu của tập test, có nhãn thật để đối chiếu
python -m lab_2.demo

# Dùng ảnh của bạn, có thể truyền nhiều ảnh
python -m lab_2.demo --image "duong_dan/anh_1.jpg" "duong_dan/anh_2.png"
```

Lệnh này nạp `checkpoints/best.pth`, tiền xử lý ảnh theo cột `val`, `test`, demo trong `data.md` mục 3, rồi dùng softmax để đổi logits thành xác suất. Kết quả được ghi vào mục `demo` của `reports/metrics.json`, biểu đồ xác suất lưu tại `reports/figures/demo_probabilities.png`.

Nếu chưa có `checkpoints/best.pth` hoặc không đọc được ảnh, lệnh dừng và báo lỗi rõ ràng.

## 2. Kết Quả Trên Ảnh Tập Test

| Ảnh | Dự đoán | Xác suất | Nhãn thật | Đúng/Sai |
| :--- | :---: | :---: | :---: | :---: |
| Ảnh test số 1 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |
| Ảnh test số 2 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |
| Ảnh test số 3 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |

## 3. Giới Hạn

- Model chỉ biết 10 lớp của CIFAR-10. Ảnh ngoài 10 lớp này vẫn bị xếp vào một trong 10 lớp, nên xác suất cao không có nghĩa là kết quả đáng tin.
- Model học từ ảnh $32 \times 32$ đã phóng to, nên có thể kém chính xác hơn với ảnh chụp thật có độ phân giải cao.

---

## 4. Đặc Tả Mã Nguồn (TV5)

**File đã triển khai**: `src/lab_2/demo.py`. **Điều kiện trước**: Đã có đủ `common.py`, `data.py`, `model.py`, `evaluation.py`.
**Lấy từ người khác**: `build_transform`, `load_splits`, `raw_samples`, `CLASS_NAMES` (TV1); `load_checkpoint` (TV2); `SERIES`, `SURFACE`, `style_axis` (TV4).

**Đầu file**: Gọi `matplotlib.use("Agg")` trước khi import `matplotlib.pyplot`. Hằng số: `DEFAULT_SAMPLES = 3`.

Giao diện triển khai và test tương ứng như dưới đây. Các test nằm trong `tests/test_demo.py` và `tests/test_integration.py`.

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
