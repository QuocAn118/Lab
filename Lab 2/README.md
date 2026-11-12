# Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Dùng model pre-trained từ `torchvision.models` để phân loại ảnh CIFAR-10 và làm 3 Exercise:
- **Exercise 1**: So sánh ResNet18 với DenseNet121.
- **Exercise 2**: Điều chỉnh learning rate và batch size.
- **Exercise 3**: Theo dõi quá trình huấn luyện bằng TensorBoard.

> **Trạng thái (2026-11-11)**: TV3 đã chạy 41 test, ba smoke test pretrained và đủ bốn run trên RTX 4060 Laptop. Validation accuracy cao nhất 88,24% (DenseNet121); đã có checkpoint, bảng so sánh và ảnh TensorBoard. Xem [báo cáo TV3](docs/training.md), [hướng dẫn](docs/TV3_TRAINING_GUIDE.md). Chưa chạy đánh giá test, demo hoặc nghiệm thu notebook Colab; không dùng số đo cục bộ thay benchmark T4.

**Review 2026-11-12:** Đã ghép module Lab 2 với main `1224a67`, giữ data Vân Anh và các phần hoàn thiện của nhóm; có notebook/demo để chạy phiên mới. Đủ 12 checkpoint được version bằng Git LFS, source lịch sử khớp manifest được lưu riêng. Xem [biên bản xử lý](docs/TV3_REVIEW_RESOLUTION.md). Kết quả train cũ không được gán cho code mới.

## Phân Công

| Thành viên | Tạo file | Đặc tả từng hàm | Kiểm tra |
| :---: | :--- | :--- | :--- |
| **TV1** | `src/lab_2/data.py` | `docs/data.md` mục 5 | `tests/test_data.py` |
| **TV2** | `src/lab_2/model.py` | `docs/model.md` mục 7 | `tests/test_model.py` |
| **TV3** | `src/lab_2/training.py` | `docs/training_specification.md` mục 8 | `tests/test_training.py`, `tests/test_smoke.py` |
| **TV4** | `src/lab_2/evaluation.py` | `docs/evaluation.md` mục 8 | `tests/test_evaluation.py` |
| **TV5** | `src/lab_2/common.py`, `src/lab_2/demo.py`, `notebooks/Lab_2_colab.ipynb` | `docs/lab_2_description.md` mục 9, `docs/demo.md` mục 4 | `tests/test_common.py`, `tests/test_demo.py`, `tests/test_integration.py` |

Thứ tự làm và cách đẩy code: xem [`plan_progress_status.md`](./docs/plan_progress_status.md) mục 2. TV5 làm `common.py` đầu tiên vì mọi file đều cần file này.

## Bắt Đầu Viết Phần Của Bạn

Lấy code về:

```bash
git clone -b main https://github.com/QuocAn118/Lab.git
```

Đứng trong thư mục `Lab 2`:

```bash
pip install -r requirements.txt

# Windows PowerShell: $env:PYTHONPATH = "src"
# Linux, macOS, Colab:
export PYTHONPATH=src

python -m pytest tests/test_data.py -q   # đổi thành file test của bạn
```

Ban đầu test báo `ModuleNotFoundError` vì chưa có file. Tạo file của bạn theo đặc tả, viết từng hàm và chạy test cho đến khi đạt. Máy cá nhân chỉ cần chạy test; việc huấn luyện đầy đủ chạy trên Colab.

## Cấu Trúc Đích Và File Đã Có

Cây dưới là cấu trúc đích; ngoài `evaluation.py`, các module trong `src/lab_2/` và `Lab_2_colab.ipynb` chưa có. Bản model hiện có nằm ở `model.py` tại gốc Lab 2, dùng `test.py` cùng thư mục. Xem [tài liệu triển khai](docs/model_implementation.md); chưa coi bản độc lập đã ghép với package.

```text
Lab 2/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/lab_2.yaml           # Tham số và 4 lần chạy
├── docs/                        # 7 file đặc tả
├── notebooks/
│   ├── Lab_2_colab.ipynb        # TV5 tạo: Notebook chạy trên Colab, chỉ gọi code trong src/
│   └── Lab 2.ipynb              # Notebook cũ, giữ làm tham khảo
├── src/lab_2/                   # Các thành viên tự tạo
│   ├── common.py                # TV5: Đọc cấu hình, ghi báo cáo, đo RAM
│   ├── data.py                  # TV1: Bước 5
│   ├── model.py                 # TV2: Bước 2, 3, 4
│   ├── training.py              # TV3: Bước 6, Exercise 1, 2, 3
│   ├── evaluation.py            # TV4: Bước 7
│   └── demo.py                  # TV5
├── tests/
│   ├── conftest.py              # Fixture dùng chung
│   ├── test_common.py           # TV5
│   ├── test_data.py             # TV1
│   ├── test_model.py            # TV2
│   ├── test_training.py         # TV3
│   ├── test_evaluation.py       # TV4
│   ├── test_demo.py             # TV5
│   ├── test_integration.py      # Cả nhóm: ghép các file
│   └── test_smoke.py            # TV3: Smoke test 2 batch
├── reports/                     # Tạo khi chạy: metrics.json, figures/, runs/
├── checkpoints/                 # Tạo khi chạy
└── data/                        # Tạo khi chạy, không đưa lên git
```

## Cách Chạy Trên Google Colab

Làm sau khi các phần đã được ghép xong.

1. Tải cả thư mục `Lab 2` lên `MyDrive` trên Google Drive. Drive cần trống khoảng 1 GB.
2. Mở `notebooks/Lab_2_colab.ipynb` bằng Google Colab.
3. Chọn **Runtime → Change runtime type → T4 GPU**.
4. Chọn **Run all** và cho phép Colab truy cập Drive khi được hỏi.
5. Chụp biểu đồ TensorBoard ở cell cuối, lưu tại `reports/figures/tensorboard.png`.
6. Lấy kết quả trong `reports/metrics.json` và `reports/figures/` để điền vào `docs/evaluation.md`, `docs/demo.md`.

Nếu đặt thư mục ở nơi khác trên Drive, hãy sửa `PROJECT_DIR` trong cell đầu của notebook.

Muốn chạy từng phần, hãy đứng trong thư mục `Lab 2` và đặt `PYTHONPATH=src`:

```bash
python -m pytest tests -q -s
python -m lab_2.data
python -m lab_2.model
python -m lab_2.training --run resnet18_baseline
python -m lab_2.training --run densenet121
python -m lab_2.training --run resnet18_lr1e-2
python -m lab_2.training --run resnet18_bs128
python -m lab_2.evaluation
python -m lab_2.demo
tensorboard --logdir reports/runs
```

## Tài Liệu Đặc Tả

1. **[`lab_2_description.md`](./docs/lab_2_description.md)**: Bài toán, đối chiếu đề bài, thiết kế, KPI, checkpoint, bảng nghiệm thu.
2. **[`data.md`](./docs/data.md)**: Nguồn dữ liệu, cách chia tập, tiền xử lý, `DataLoader`.
3. **[`model.md`](./docs/model.md)**: Các model, lớp phân loại, đóng băng và fine-tuning, số tham số.
4. **[`training.md`](./docs/training.md)**: Báo cáo huấn luyện và TensorBoard; [đặc tả kỹ thuật](./docs/training_specification.md) chứa siêu tham số và giao diện hàm.
5. **[`evaluation.md`](./docs/evaluation.md)**: So sánh các lần chạy, đối chiếu KPI, kết quả test.
6. **[`demo.md`](./docs/demo.md)**: Lệnh chạy và kết quả suy luận.
7. **[`plan_progress_status.md`](./docs/plan_progress_status.md)**: Thành viên, cách làm việc nhóm, tiến độ, nhật ký, việc dùng AI.

## Giới Hạn

- Bài dùng một nửa dữ liệu CIFAR-10 và tối đa 6 epoch để giảm thời gian dùng GPU.
- Fine-tuning `layer4` có trong model độc lập; kế hoạch nghiệm thu package dùng smoke test. Chưa có bằng chứng chạy đầy đủ theo đặc tả chung.
- Nếu Colab ngắt phiên giữa lúc huấn luyện thì phải chạy lại lần đang chạy dở.
