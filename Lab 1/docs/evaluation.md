# Performance Evaluation & System Benchmarks — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Performance Evaluation & System Benchmarks — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T09:38:18+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Trạng thái: **chưa có kết quả huấn luyện/đánh giá để báo cáo**. `src/evaluate.py` đã có; các bảng dưới chờ số liệu run thật. Chi tiết đầu ra theo [3.5](requirements/3_5_evaluate.md) và [đặc tả gốc §8](lab_1_description.md).

## 1. Training & Validation Progress Log

| Epoch | Train Loss | Val Loss | Val Accuracy (0–1) | Checkpoint Flag | Notes |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | - | - | - | Baseline | - |
| **...**| - | - | - | ... | - |

---

## 2. Quantitative System Benchmarks
- **Peak Validation Accuracy**: [Chờ đo, tỷ lệ 0–1]
- **Average Training Speed**: [Giây / Epoch]
- **Inference Latency**: [ms / Sample]
- **Memory Footprint**: [MB / GB]

Lấy train/val từ `outputs/<run_id>/history.csv`; test accuracy/loss từ `metrics.json` và đối chiếu `predictions.csv`. `performance.json` phải có cả accuracy (gồm theo lớp), thời gian (train, từng epoch, suy luận) và tài nguyên (tham số, kích thước checkpoint, GPU peak memory hoặc `null` trên CPU, môi trường). Timestamp dùng ISO 8601; thời gian dùng giây, riêng latency là ms/ảnh.

Mỗi normal run đánh giá đủ 10.000 ảnh test, smoke dùng 256 ảnh. Chọn checkpoint bằng validation; test chỉ dùng đánh giá cuối. Không thêm hoặc sửa file trong run đã chốt manifest: đánh giá phải được ghép vào vòng đời run trước khi chốt.

| Run | Thay đổi | Best val accuracy | Test accuracy / loss | Train seconds | Inference ms/ảnh | Nguồn |
| --- | --- | --- | --- | --- | --- | --- |
| `run_001` | Cơ sở | Chưa có | Chưa có | Chưa có | Chưa có | Chưa chạy |
| `run_002` | Hidden `[256,128]` | Chưa có | Chưa có | Chưa có | Chưa có | Chưa chạy |
| `run_003` | lr 0.01 | Chưa có | Chưa có | Chưa có | Chưa có | Chưa chạy |
| `run_004` | Adam lr 0.001 | Chưa có | Chưa có | Chưa có | Chưa có | Chưa chạy |

Sau khi chạy, bổ sung `outputs/compare_runs.csv`, `compare_loss.png`, đồ thị train/val loss và 16 ảnh dự đoán mỗi run. Nhận xét dựa trên validation, overfitting và đánh đổi accuracy/thời gian/tài nguyên; dẫn đường dẫn bằng chứng thật.

---

## 3. Checkpoint Artifacts & Model Registry
- `final_epoch`: `checkpoints/<run_id>/final_epoch.pth`.
- `best_val_loss`: `checkpoints/<run_id>/best_val_loss.pth`.
- `best_val_acc`: `checkpoints/<run_id>/best_val_acc.pth`.
- `best_val_pause`: `checkpoints/<run_id>/best_val_pause.pth` (kèm cờ `early_stopped` trong index).
- `best`: `checkpoints/<run_id>/best.pth`, bản sao `best_val_acc`, dùng đánh giá và demo.
- Metadata: `checkpoints/<run_id>/checkpoint_index.json`.

Evaluation dùng `load_model(run_id, device, checkpoint="best")`; cách gọi cũ không có đối số checkpoint vẫn tải final_epoch để tương thích.
