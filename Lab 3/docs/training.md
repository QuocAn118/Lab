# Cấu Hình Huấn Luyện (Training Specification) — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Cấu Hình Huấn Luyện (Training Specification) — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-13T08:56:14+07:00 |
| Updated by | AI agent (thực thi kế hoạch của HoangDang) |
| Phạm vi | Tài liệu kế thừa bản QuocAn; nội dung lịch sử, xem bộ 01–12 cho bản chính |

> Tài liệu của bản triển khai nhánh QuocAn (`src/exercise_*`, `notebooks/Lab_3_quocan.ipynb`). Tài liệu chính của bài là bộ `01_`–`12_`.

Exercise 1 không huấn luyện. Toàn bộ file này áp dụng cho Exercise 2.

Mã nguồn: `src/exercise_2/training.py`. Tham số: `configs/exercise_2.yaml`.

Bổ sung tích hợp ngày 2026-11-12 bởi HoangDang theo `08_implementation.md` mục 10. Lệnh train từ chối thư mục checkpoint có dữ liệu hoặc báo cáo đã có mục training. Phiên mới cần đổi đường dẫn checkpoint và reports trong config. Bảng số liệu lịch sử của bản Quốc An vẫn được đọc theo nguồn báo cáo tương ứng.

## 1. Siêu Tham Số

| Tham số | Giá trị | Ghi chú |
| :--- | :---: | :--- |
| Framework | Hugging Face `Trainer` | |
| Optimizer | `AdamW` | Mặc định của `Trainer` |
| Learning rate | $2 \times 10^{-5}$ | |
| LR scheduler | `linear` | Giảm tuyến tính, `warmup_steps = 0` |
| Batch size | 16 | Train và validation |
| Số epoch tối đa | 5 | Có thể dừng sớm từ epoch 3 |
| Weight decay | $0.01$ | |
| `max_grad_norm` | $1.0$ | Giới hạn độ lớn gradient |
| Hàm mất mát | Cross-Entropy | Tích hợp trong model |
| Seed | 42 | `seed` và `data_seed` |
| `logging_steps` | 20 | Ghi train loss mỗi 20 bước |
| Mixed precision | `fp16` | Chỉ bật khi có GPU |

## 2. Smoke Test
- **Cách chạy**: `python -m pytest tests/test_smoke.py -s`, trước khi train thật.
- **Quy mô**: 2 batch, mỗi batch 16 mẫu, dùng dữ liệu và model thật.
- **Tiêu chí đạt**:
  - Logits có shape đúng $(16, 2)$.
  - Loss là số hữu hạn.
  - Gradient của lớp `classifier` tồn tại và khác 0.
  - Chạy đủ 2 batch, không hết bộ nhớ.
  - Khi có GPU, VRAM cao nhất không vượt 4,0 GB.

## 3. Quy Trình Huấn Luyện
1. Chạy `python -m exercise_2.training` trên tập 1.000 mẫu train.
2. Sau mỗi epoch, đánh giá trên tập validation và cập nhật các mốc checkpoint.
3. Khi kết thúc, sao chép `best_val_acc` thành `best` và ghi nhật ký vào mục `training` của `reports/metrics.json`.

Tập test không được dùng trong quá trình này.

## 4. Checkpoint & Early Stopping
- `eval_strategy = "epoch"`, `save_strategy = "epoch"`.
- `load_best_model_at_end = True`, `metric_for_best_model = "accuracy"`, `greater_is_better = True`.
- **Early stopping**: `early_stopping_patience = 2`, `early_stopping_threshold = 0.01`.
  - Một epoch chỉ được tính là cải thiện nếu val accuracy tăng **hơn** $1\%$ so với mức tốt nhất đang lưu.
  - Dừng sau 2 lần đánh giá liên tiếp không cải thiện.
  - Với 200 mẫu validation, mỗi mẫu tương đương $0.5\%$, nên cần đoán đúng thêm ít nhất 3 mẫu mới tính là cải thiện.

## 5. Xử Lý Tình Huống

| Tình huống | Cách xử lý |
| :--- | :--- |
| Colab ngắt phiên | Chạy lại từ đầu, không tiếp tục từ giữa chừng. Mốc `best` cũ chỉ bị thay khi lần chạy mới hoàn tất. |
| Không có GPU | In cảnh báo và chạy bằng CPU, tắt `fp16`. KPI tốc độ và VRAM ghi "Không đo được". |
| Một epoch đạt nhiều mốc tốt nhất | Mỗi mốc vẫn lưu trong thư mục riêng, kể cả khi trọng số giống nhau. |
| Lỗi khi đang lưu checkpoint | Ghi vào thư mục tạm rồi mới thay thế, nên mốc cũ không bị hỏng. |
| Chưa có checkpoint `best` | `evaluation.py` và `demo.py` dừng và báo cần chạy huấn luyện trước. |

## Nguồn và phạm vi

- Nhập từ nhánh QuocAn, đầu nhánh `32ca8b1`; không diễn giải kết quả cũ thành số đo sau chuẩn hóa.
- [Bản chính](01_lab_3_description.md), [nhật ký đồng bộ](../../BRANCH_SYNC_2026-11-11.md). Lần cập nhật hiện tại chỉ chuẩn hóa thông tin và phân loại tài liệu.
