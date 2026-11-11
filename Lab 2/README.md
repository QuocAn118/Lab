# Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-11T11:13:30+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

**Trạng thái: đã chuẩn bị code để bàn giao train; chưa nghiệm thu bài thực nghiệm.** Bám đặc tả trên `main` tại `a7b7ef8`: ResNet18/DenseNet121 pre-trained, 4 run, tối đa 6 epoch, chia CIFAR-10 22.500/2.500/5.000 ảnh.

Đã có đủ 6 module package và notebook Colab. Kiểm tra offline: **63 passed**, gồm 49 test package và 14 test tương thích model cũ. Mọi dữ liệu/model dùng trong kiểm tra tích hợp đều là dữ liệu giả/trọng số khởi tạo, không phải kết quả CIFAR-10. Xem [bằng chứng chuẩn bị](docs/verification.md).

## Bàn giao cho bạn train

Đọc [kế hoạch train và bàn giao](docs/training_handoff.md), mở [Lab_2_colab.ipynb](notebooks/Lab_2_colab.ipynb). Chọn T4 GPU, điền đường dẫn project, họ tên người chạy và mã phiên trong cell đầu. Notebook chạy lần lượt kiểm tra offline → tải dữ liệu/model → smoke thật → 4 run → đánh giá → demo → TensorBoard. Lệnh lỗi sẽ dừng cell bằng `check=True`.

Code đã chuẩn bị trên nhánh `hoangdang`, nền là `main` tại `a7b7ef8`; dùng bản bàn giao mới nhất trên nhánh này cho đến khi PR được merge. Không lấy riêng các file từ nhánh cá nhân cũ.

## Code, input và output

| Thành phần | Vị trí | Vai trò |
| --- | --- | --- |
| Cấu hình chuẩn | [configs/lab_2.yaml](configs/lab_2.yaml) | Dataset, 4 run, early stopping, KPI; giữ nguyên khi so sánh |
| Tiện ích | [common.py](src/lab_2/common.py) | Đọc YAML, cập nhật JSON nguyên tử, đo RAM |
| Input | [data.py](src/lab_2/data.py) | Tải CIFAR-10, chia tập seed 42, augmentation/chuẩn hóa |
| Model | [model.py](src/lab_2/model.py) | Model 10 lớp, đóng băng/mở khóa, checkpoint |
| Training | [training.py](src/lab_2/training.py) | AdamW, scheduler, checkpoint, early stopping, TensorBoard |
| Đánh giá | [evaluation.py](src/lab_2/evaluation.py) | Chọn bằng validation, test một lần, KPI và biểu đồ |
| Demo | [demo.py](src/lab_2/demo.py) | Ảnh test hoặc ảnh của bạn → 10 xác suất và nhãn dự đoán |

Input model: tensor float32 `[B,3,224,224]`, chuẩn hóa theo ImageNet; nhãn int64 0–9. Output: logits `[B,10]`; demo đổi thành xác suất bằng softmax.

Notebook tạo config phiên trong `configs/sessions/<SESSION_ID>.yaml`; output thực tế nằm dưới `reports/sessions/<SESSION_ID>/`: `checkpoints/`, `reports/metrics.json`, `reports/runs/`, `reports/figures/`. Phiên mới không ghi đè báo cáo lịch sử `reports/metrics.json`. Các đường dẫn `checkpoints/`, `reports/` trong đặc tả là mặc định của config gốc; config phiên chỉ đổi nơi lưu, giữ nguyên tham số thí nghiệm.

Hiện **chưa có** dataset CIFAR-10, checkpoint đã train, accuracy chính thức hay log TensorBoard của 4 run. [metrics.json hiện có](reports/metrics.json) là báo cáo tham số lịch sử của Bảo; giữ nguyên, chưa phải kết quả training.

## Kiểm tra offline trên máy cá nhân

Từ `Lab 2/`, dùng Python 3.12 và môi trường riêng:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
$env:PYTHONUTF8 = '1'
& .venv/Scripts/python.exe -m pytest tests test.py --ignore=tests/test_smoke.py -q --basetemp .venv/pytest-temp
```

Lệnh trên không tải CIFAR-10 hoặc trọng số ImageNet. Trên Linux/Colab dùng `export PYTHONPATH=src` và `python` của môi trường tương ứng. Bộ smoke thật nằm riêng trong `tests/test_smoke.py`, dành cho người train thực hiện trước 4 run.

## Tài liệu và nguồn

- [Đặc tả tổng quan](docs/lab_2_description.md), [dữ liệu](docs/data.md), [model](docs/model.md), [huấn luyện](docs/training.md).
- [Báo cáo đánh giá](docs/evaluation.md), [demo](docs/demo.md): giữ ô chờ số đo thật.
- [Tiến độ, phân công và khai báo AI](docs/plan_progress_status.md).
- [Triển khai model gốc](docs/model_implementation.md), [bản tiếng Việt](docs/model_vi.md): nguồn Xuân Bảo; đã chuyển code sang package, giữ `model.py` làm lệnh tương thích cho `test.py` cũ.

Không ghi đè run đã có log/checkpoint/kết quả. Nếu phiên dở dang cần chạy lại, dùng mã phiên mới; giữ phiên cũ làm bằng chứng. Không bảo đảm đạt KPI trước khi đo; GPU khác T4 phải ghi rõ, không trình bày như số đo trên T4.
