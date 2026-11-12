# Lab 1 — FashionMNIST MLP bằng PyTorch

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Lab 1 — FashionMNIST MLP bằng PyTorch |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-12T21:38:06+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## Sản phẩm đã hoàn thành

Pipeline Python CLI phân loại FashionMNIST bằng MLP đã chạy end to end: data → train/checkpoint → reload best → test/performance → đồ thị/demo → verification/manifest. **20 test đạt; smoke_001 và cả bốn normal run PASS**, kiểm lại hash sau khi chốt.

Model được chọn bằng validation: **run_004**, validation **88.10%**, test **87.70%** trên 10.000 ảnh. [Báo cáo so sánh](docs/evaluation.md), [kiểm chứng](docs/verification.md), [truy vết](docs/traceability.md).

## Input và output ở đâu

| Thành phần | Vị trí | Nội dung |
| --- | --- | --- |
| Dataset input | [data/FashionMNIST/raw/](data/FashionMNIST/raw/) | IDX/gz gốc FashionMNIST, được lưu trong repo qua Git LFS |
| Data reports | [data/reports/](data/README.md) | Summary, split seed 42, batch check và ảnh khảo sát; không phải kết quả huấn luyện |
| Một ảnh input thật | [demo_input.png](outputs/demo_input.png) | Ảnh xám 28×28, test index 0 |
| Output ảnh đó | [demo_example.json](outputs/demo_example.json) | Input shape, 10 logits, nhãn thật và dự đoán; `Ankle boot` |
| Model đã train | [checkpoints/run_004/best.pth](checkpoints/run_004/best.pth) | State_dict được chọn bằng validation, lưu qua Git LFS; cần config cùng run để tải |
| Toàn bộ kết quả | [outputs/run_004/](outputs/run_004/metrics.json) | Config, history, metrics, predictions CSV, performance, reload, đồ thị, ảnh mẫu, verify_report và manifest |
| So sánh bốn run | [compare_runs.csv](outputs/compare_runs.csv), [compare_loss.png](outputs/compare_loss.png) | Accuracy/thời gian/tài nguyên và val loss |

Input model là tensor float32 `[B,1,28,28]`, giá trị [0,1]; nhãn int64 0–9. Output là `[B,10]` logits; argmax chọn nhãn. Dataset chia 54.000/6.000/10.000, seed 42. Ảnh dự đoán có nhãn thật/dự đoán, tô đỏ khi sai; xem [16 ảnh mẫu](outputs/run_004/prediction_samples.png).

## Chạy trên máy hiện tại

Từ `Lab 1` (PowerShell):

```powershell
$env:PYTHONUTF8='1'
& .venv/Scripts/python.exe src/demo.py --run-id run_004 --index 0
& .venv/Scripts/python.exe src/verify.py --run-id run_004
& .venv/Scripts/python.exe tools/check_provenance.py --strict
& .venv/Scripts/python.exe -m pytest tests data/tests -q --basetemp .venv/pytest-temp
```

CLI demo/verify chỉ đọc. Đường dẫn Python bên trên dùng môi trường Lab 1 đã chuẩn bị; Python toàn hệ thống có thể chưa có torch. Thư mục tạm pytest đặt trong .venv để tránh thư mục temp Windows bị từ chối quyền.

## Chuẩn bị trên máy mới và chạy lại

Sản phẩm chính nằm trực tiếp trong cây repo: `src/`, `data/`, `checkpoints/`, `outputs/`, `docs/`. Sau khi clone/pull, lấy nội dung nhị phân của dataset và cả 25 checkpoint bằng Git LFS (từ gốc repo):

```powershell
git lfs install
git lfs pull origin hoangdang
```

Sau đó vào `Lab 1` và chuẩn bị môi trường. Không cần giải nén ZIP hoặc chạy lại huấn luyện để dùng model đã có. `data/prepare_data.py` là bước tải/khảo sát lại khi cần; nếu tạo lại báo cáo, sidecar giữ lịch sử nguồn.

Môi trường đã đo: Windows, Python 3.12.10 và GPU RTX 5060 Laptop. Cặp wheel CUDA 12.8 đã kiểm tra hoạt động trên máy này:

```powershell
$env:PYTHONUTF8='1'
python -m venv .venv
& .venv/Scripts/python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
& .venv/Scripts/python.exe -m pip install -r requirements.txt
& .venv/Scripts/python.exe data/prepare_data.py
& .venv/Scripts/python.exe data/data_pipeline.py --check
```

Máy CPU/thiết bị khác cần chọn wheel phù hợp theo [PyTorch](https://pytorch.org/get-started/previous-versions/); `requirements-lock.txt` ghi môi trường đã đo, không cam kết cùng tốc độ trên máy khác.

Trước run mới, commit các thay đổi Lab 1 để manifest truy về đúng code. Không ghi đè run cũ; các mã 001–004 đã có kết quả. Ví dụ tạo smoke mới rồi baseline mới:

```powershell
& .venv/Scripts/python.exe src/train.py --run-id smoke_002 --purpose "Smoke on this machine"
& .venv/Scripts/python.exe src/verify.py --run-id smoke_002
# Chỉ chạy normal sau khi smoke PASS và cây Lab 1 đã commit.
& .venv/Scripts/python.exe src/train.py --run-id run_005 --purpose "Repeat baseline"
```

Xem [training.md](docs/training.md) và [đặc tả](docs/lab_1_description.md) cho bốn cấu hình, checkpoint và early stopping. Git lưu code/báo cáo và pointer LFS; Git LFS lưu nội dung dataset/trọng số, được khôi phục đúng đường dẫn sau pull. ZIP cũ chỉ là snapshot dự phòng cục bộ, không phải sản phẩm chính hoặc dependency khi chạy repo.

## Nguồn và phạm vi

- Các file trong `docs/requirements/`, source và artifact Lab 1. Không dùng kết quả Lab 3 làm bằng chứng cho Lab 1.
- HoangDang hoàn thiện; tác giả code/tài liệu kế thừa giữ trong Git và registry. Tên thành viên TV1–TV5/ngày tạo model chưa được xác nhận chính thức không được tự điền.
