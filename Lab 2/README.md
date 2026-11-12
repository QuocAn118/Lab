# Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-12T21:38:06+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

**Trạng thái: đã nhận và kiểm chứng bản bàn giao TV3 của Thanh Bình; phiên chạy lại được lưu riêng. Chưa nghiệm thu KPI Colab T4.** ResNet18/DenseNet121 pre-trained, 4 cấu hình, tối đa 6 epoch, chia CIFAR-10 22.500/2.500/5.000 ảnh.

Đủ 6 module package và notebook Colab. Sau ghép TV3: **93 test offline PASS** (kiểm lại sau `25cc678`), gồm 79 test package và 14 test tương thích; **3 smoke pretrained thật PASS trên RTX 5060 Laptop**. Bộ test offline dùng dữ liệu giả, được phân biệt với smoke thật và training. Xem [kiểm chứng](docs/verification.md).

## Bàn giao cho bạn train

Đọc [kế hoạch train và bàn giao](docs/training_handoff.md), mở [Lab_2_colab.ipynb](notebooks/Lab_2_colab.ipynb). Chọn T4 GPU, điền đường dẫn project, họ tên người chạy và mã phiên trong cell đầu. Notebook chạy lần lượt kiểm tra offline → tải dữ liệu/model → smoke thật → 4 run → đánh giá → demo → TensorBoard. Lệnh lỗi sẽ dừng cell bằng `check=True`.

Code và kết quả chạy lại đã được tích hợp vào `main` tại `c6f4d54` qua PR #4. Dùng toàn bộ bản hiện hành của Lab 2 để giữ đồng bộ config, module và test.

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

Phần dữ liệu của Vân Anh đã kiểm chứng với CIFAR-10 thật trên Windows ngày 2026-11-12: xem [bằng chứng dữ liệu](reports/data_validation_2026-11-12.json). Dataset trong `data/` bị Git bỏ qua; máy mới cần tải lại.

Đã nhận **một phiên lịch sử của Thanh Bình, gồm bốn cấu hình**, tại commit `92930e6`: [biên bản nhập](reports/imported-thanhbinh-92930e6/IMPORT.json), [metrics gốc](reports/imported-thanhbinh-92930e6/metrics.json), [báo cáo gốc](reports/imported-thanhbinh-92930e6/training_report.md). Đủ 12 checkpoint LFS trong `checkpoints/<run>/`, log TensorBoard trong `reports/runs/`, source/manifest gốc trong `reports/tv3/`. Đã kiểm hash 12 checkpoint, 21 source snapshot, ba manifest và nạp bốn model. Kết quả gốc dùng RTX 4060, không phải lần chạy mới trên máy này. [metrics.json ở gốc reports](reports/metrics.json) vẫn là báo cáo tham số lịch sử của Bảo, giữ nguyên.

Phiên chạy lại mới đã hoàn tất: `rerun-thanhbinh-20261112-175308`, dùng [config riêng](configs/sessions/rerun-thanhbinh-20261112-175308.yaml); trạng thái và kết quả tại [SESSION.json](reports/sessions/rerun-thanhbinh-20261112-175308/SESSION.json). Không chấm KPI tốc độ T4 từ số đo RTX; số đo thực tế vẫn được lưu.

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
- [Tiến độ, phân công](docs/plan_progress_status.md); [khai báo công cụ](docs/AI_Usage.md).
- [Triển khai model gốc](docs/model_implementation.md), [bản tiếng Việt](docs/model_vi.md): nguồn Xuân Bảo; đã chuyển code sang package, giữ `model.py` làm lệnh tương thích cho `test.py` cũ.

Không ghi đè run đã có log/checkpoint/kết quả. Nếu phiên dở dang cần chạy lại, dùng mã phiên mới; giữ phiên cũ làm bằng chứng. Không bảo đảm đạt KPI trước khi đo; GPU khác T4 phải ghi rõ, không trình bày như số đo trên T4.

## Kết quả chạy lại trên máy hiện tại

DenseNet121 được chọn bằng validation 88,24%, epoch 5; test 87,68%, macro F1 0,8755 trên 5.000 ảnh. Bốn cấu hình có validation 84,76% / 88,24% / 83,52% / 85,08%; chênh bản nguồn từ −0,04 đến +0,04 điểm phần trăm. Kiểm lại trọng số DenseNet gốc của Bình cũng cho validation 88,24% và test 87,68%.

[Báo cáo và biểu đồ so sánh](reports/sessions/rerun-thanhbinh-20261112-175308/comparison.md), [báo cáo đánh giá](docs/evaluation.md), [best checkpoint mới](reports/sessions/rerun-thanhbinh-20261112-175308/checkpoints/best.pth). Phiên mới có 13 file trọng số (12 theo run và bản best), checkpoint/log/metrics gốc của Bình giữ nguyên. Lấy trọng số mới sau pull bằng Git LFS; dùng phiên/mã mới nếu tái lập, không chạy lại đè config phiên đã hoàn tất.

Đã chụp [giao diện TensorBoard](reports/sessions/tensorboard-review-20261112-191218/tensorboard.png) của bốn run với accuracy, train loss và validation loss. [Biên bản kiểm tra bổ sung](reports/sessions/tensorboard-review-20261112-191218/REVIEW.json) ghi nguồn event, hash và tùy chọn hiển thị.
