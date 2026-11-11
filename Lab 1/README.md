# Lab 1 — FashionMNIST MLP bằng PyTorch

Phân loại ảnh xám 28 × 28 thành 10 lớp trang phục bằng MLP 784 → 512 → 512 → 10. Cấu hình cơ sở: `ToTensor`, batch 64, SGD (lr 0.001, momentum 0.9), tối đa 10 epoch; split train/validation/test là 54.000/6.000/10.000, seed 42.

Last updated: 2026-11-10 — Codex (AI hỗ trợ, chờ nhóm review).

Đã có code data, model, train và lưu/tải checkpoint. Chưa có bằng chứng smoke/normal run đầy đủ; evaluation, đồ thị và demo chưa triển khai. `load_model` hiện chỉ nạp checkpoint cuối, còn yêu cầu đánh giá bằng `best.pth` chưa hoàn tất.

Phần **3.1–3.2** được gom trong **[data/](data/README.md)**: code chuẩn bị dữ liệu, DataLoader, dữ liệu gốc, báo cáo và kiểm thử. Xem README trong đó để chạy.

Các lệnh sau chạy từ thư mục `Lab 1`; cần Python có `torch`, `torchvision`, `matplotlib` và `pytest` (cho kiểm thử). Dataset gốc không lưu trên Git, bước chuẩn bị cần mạng khi chưa có dữ liệu.

```powershell
python data/prepare_data.py
python data/data_pipeline.py --check
python -m pytest data/tests tests -q
```

Sau khi chuẩn bị dữ liệu, lệnh smoke run dự kiến là `python src/train.py --run-id smoke_001 --purpose "Kiểm tra pipeline"`. Chọn mã mới nếu đã tồn tại. Xem [training.md](docs/training.md) để biết đầu ra hiện có và điều kiện trước normal run. Kiểm tra nguồn gốc file từ đây bằng `python tools/check_provenance.py --strict`.

## 📁 Cấu Trúc Tài Liệu Đặc Tả (`docs/`)
1. **[`docs/lab_1_description.md`](./docs/lab_1_description.md)**: Đặc tả bài toán, mục tiêu hệ thống, chỉ số KPIs & chính sách quản lý Checkpoints.
2. **[`docs/data.md`](./docs/data.md)**:  dữ liệu FashionMNIST, Transforms, DataLoaders.
3. **[`docs/model.md`](./docs/model.md)**: Kiến trúc MLP kế thừa `torch.nn.Module`; hồ sơ model trước và sau huấn luyện.
4. **[`docs/training.md`](./docs/training.md)**: Cấu hình giải thuật tối ưu SGD/Adam, Loss function, quy trình smoke test.
5. **[`docs/evaluation.md`](./docs/evaluation.md)**: Báo cáo định lượng hệ thống (Accuracy, Loss, thời gian & tài nguyên).
6. **[`docs/demo.md`](./docs/demo.md)**: Kịch bản và mã nguồn suy luận nhanh với ảnh mẫu.
7. **[`docs/plan_progress_status.md`](./docs/plan_progress_status.md)**: Lộ trình triển khai, bảng phân rã nhiệm vụ (TODO / DONE).
