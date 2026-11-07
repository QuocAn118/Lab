# Practice 2 — Hands-on Practice with Pre-trained Neural Network Architectures

Dùng model pre-trained từ `torchvision.models` để phân loại ảnh CIFAR-10 và làm 3 Exercise:
- **Exercise 1**: So sánh ResNet18 với DenseNet121.
- **Exercise 2**: Điều chỉnh learning rate và batch size.
- **Exercise 3**: Theo dõi quá trình huấn luyện bằng TensorBoard.

## Cấu Trúc Thư Mục

```text
Lab 2/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/lab_2.yaml           # Tham số và 4 lần chạy
├── docs/                        # 7 file đặc tả
├── notebooks/
│   ├── Lab_2_colab.ipynb        # Notebook chạy trên Colab, chỉ gọi code trong src/
│   └── Lab 2.ipynb              # Notebook cũ, giữ làm tham khảo
├── src/lab_2/
│   ├── common.py                # Đọc cấu hình, ghi báo cáo, đo RAM
│   ├── data.py                  # Bước 5
│   ├── model.py                 # Bước 2, 3, 4
│   ├── training.py              # Bước 6, Exercise 1, 2, 3
│   ├── evaluation.py            # Bước 7
│   └── demo.py
├── tests/
│   ├── test_lab_2.py            # Test logic
│   └── test_smoke.py            # Smoke test 2 batch
├── reports/                     # Tạo khi chạy: metrics.json, figures/, runs/
├── checkpoints/                 # Tạo khi chạy
└── data/                        # Tạo khi chạy, không đưa lên git
```

## Cách Chạy Trên Google Colab

1. Tải cả thư mục `Lab 2` lên `MyDrive` trên Google Drive. Drive cần trống khoảng 1 GB.
2. Mở `notebooks/Lab_2_colab.ipynb` bằng Google Colab.
3. Chọn **Runtime → Change runtime type → T4 GPU**.
4. Chọn **Run all** và cho phép Colab truy cập Drive khi được hỏi.
5. Chụp biểu đồ TensorBoard ở cell cuối, lưu tại `reports/figures/tensorboard.png`.
6. Lấy kết quả trong `reports/metrics.json` và `reports/figures/` để điền vào `docs/evaluation.md`, `docs/demo.md`.

Nếu đặt thư mục ở nơi khác trên Drive, hãy sửa `PROJECT_DIR` trong cell đầu của notebook.

Muốn chạy từng phần, hãy đứng trong thư mục `Lab 2` và đặt `PYTHONPATH=src`:

```bash
pip install -r requirements.txt
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
4. **[`training.md`](./docs/training.md)**: Siêu tham số, các lần chạy, smoke test, TensorBoard, early stopping.
5. **[`evaluation.md`](./docs/evaluation.md)**: So sánh các lần chạy, đối chiếu KPI, kết quả test.
6. **[`demo.md`](./docs/demo.md)**: Lệnh chạy và kết quả suy luận.
7. **[`plan_progress_status.md`](./docs/plan_progress_status.md)**: Tiến độ, nhật ký, việc dùng AI.

## Giới Hạn

- Bài dùng một nửa dữ liệu CIFAR-10 và tối đa 6 epoch để giảm thời gian dùng GPU.
- Fine-tuning `layer4` chỉ được kiểm tra bằng smoke test, không có lần chạy đầy đủ.
- Nếu Colab ngắt phiên giữa lúc huấn luyện thì phải chạy lại lần đang chạy dở.
