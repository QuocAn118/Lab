# Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-14 |
| Updated by | Nguyễn Quốc An (Quoc An), AI hỗ trợ: dọn repo, notebook chạy bằng VS Code |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

**Trạng thái 2026-11-14: code và test offline đã xong; chờ chạy lại notebook trên Colab T4.** ResNet18/DenseNet121 pre-trained, 4 cấu hình, tối đa 6 epoch, chia CIFAR-10 22.500/2.500/5.000 ảnh. Kết quả các lần chạy trước (RTX 4060, RTX 5060, Colab T4 ngày 13/11) đã gỡ khỏi repo và vẫn có trong lịch sử Git.

## Chạy trên Colab bằng VS Code

1. Push code lên nhánh `QuocAn` trên GitHub (máy Colab chỉ thấy code đã push).
2. Mở [notebooks/lab_2_colab.ipynb](notebooks/lab_2_colab.ipynb) trong VS Code, chọn kernel Google Colab có GPU T4.
3. Chạy lần lượt từ trên xuống. Ô đầu tự `git clone` nhánh trong biến `BRANCH`; lệnh lỗi thì notebook dừng.
4. Ô cuối kiểm tra đủ kết quả rồi chép `checkpoints/` và `reports/` sang `MyDrive/Lab2_ket_qua/<ngày-giờ>/`.
5. Tải thư mục đó về, chép vào `Lab 2/`, lưu notebook có output (Ctrl+S), rồi commit.

Thứ tự bước, điều kiện qua bước và danh sách kết quả cần có: [training.md](docs/training.md) mục 9.

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

Kết quả sau khi chạy (tạo khi chạy notebook, không có sẵn trong repo):

| Vị trí | Nội dung |
| --- | --- |
| `checkpoints/<run>/` | `final_epoch.pth`, `best_val_loss.pth`, `best_val_acc.pth` của từng run; `checkpoints/best.pth` là model được chọn |
| `reports/metrics.json` | Mục `data`, `model`, `training`, `evaluation`, `demo` |
| `reports/runs/<run>/` | Log TensorBoard |
| `reports/figures/` | Biểu đồ validation, loss, confusion matrix, demo và ảnh TensorBoard |

Dataset tải về `data/`, thư mục này bị Git bỏ qua.

## Kiểm tra offline trên máy cá nhân

Từ `Lab 2/`, dùng Python 3.12 và môi trường riêng:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'
$env:PYTHONUTF8 = '1'
& .venv/Scripts/python.exe -m pytest tests --ignore=tests/test_smoke.py -q --basetemp .venv/pytest-temp
```

Lệnh trên không tải CIFAR-10 hoặc trọng số ImageNet. Trên Linux/Colab dùng `export PYTHONPATH=src`. Bộ smoke thật nằm riêng trong `tests/test_smoke.py`, chạy trong notebook trước 4 run.

## Tài liệu

- [Đặc tả tổng quan](docs/lab_2_description.md), [dữ liệu](docs/data.md), [model](docs/model.md), [huấn luyện](docs/training.md).
- [Báo cáo đánh giá](docs/evaluation.md), [demo](docs/demo.md): điền số thật sau khi chạy lại.
- [Tiến độ, phân công](docs/plan_progress_status.md); [khai báo công cụ](docs/ai_usage.md).
- Notebook cũ (chỉ tham khảo, khác đặc tả hiện tại): [notebooks/lab_2_old.ipynb](notebooks/lab_2_old.ipynb).

Không ghi đè run đã có log/checkpoint/kết quả. Không bảo đảm đạt KPI trước khi đo; GPU khác T4 phải ghi rõ, không trình bày như số đo trên T4.
