# Cấu Hình Huấn Luyện (Training Specification) — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Cấu Hình Huấn Luyện (Training Specification) — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Tài liệu kế thừa bản QuocAn; nội dung lịch sử, xem bộ 01–12 cho bản chính |

> Tài liệu của bản triển khai nhánh QuocAn (`src/exercise_*`, notebook chạy pipeline `notebooks/Lab_3_colab.ipynb`; `notebooks/Lab_3_quocan.ipynb` là notebook cũ tham khảo). Tài liệu chính của bài là bộ `01_`–`12_`.

> Vị trí hiện hành: docs/docs demo/. Bộ chính 01–12 ở thư mục cha, theo full split 20.000/5.000/25.000 và tối đa 3 epochs. Các bảng số liệu dưới đây giữ kết quả lịch sử Quốc An (1.000/200/200, tối đa 5 epochs), không phải kết quả mới của đợt chuẩn hóa. Các đường dẫn src/, configs/, checkpoints/, reports/, notebooks/ và lệnh bên dưới tính từ Lab 3/. Xem [hướng dẫn phân biệt](README.md).


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
| Colab ngắt phiên | Không resume optimizer. Đổi cả checkpoints_dir và reports_dir sang thư mục phiên mới rồi chạy lại; không xóa checkpoint cũ để vượt kiểm tra ghi đè. |
| Không có GPU | In cảnh báo và chạy bằng CPU, tắt `fp16`. KPI tốc độ và VRAM ghi "Không đo được". |
| Một epoch đạt nhiều mốc tốt nhất | Mỗi mốc vẫn lưu trong thư mục riêng, kể cả khi trọng số giống nhau. |
| Lỗi khi đang lưu checkpoint | Code ghi thư mục tạm trước khi thay thế. Chưa có rollback nếu đổi tên lỗi sau khi xóa mốc cũ; cần giữ bản sao trước khi chạy. Chưa khắc phục code trong đợt chuẩn hóa tài liệu. |
| Chưa có checkpoint `best` | `evaluation.py` và `demo.py` dừng và báo cần chạy huấn luyện trước. |

## 6. Chạy phiên mới sau nhập nhánh

Lệnh mặc định dùng checkpoints/ và reports/ đã có kết quả lịch sử nên bị chặn. Tạo bản sao configs/exercise_2.yaml với cả checkpoints_dir và reports_dir mới, sau đó truyền cùng --config cho mọi bước data/model/training/evaluation/demo. Không xóa dữ liệu lịch sử để chạy lại.

Notebook Colab lịch sử chưa chuyển các lệnh !python sang cơ chế dừng khi subprocess thất bại. Vì vậy Run All không bảo đảm đã tạo kết quả mới; phải kiểm tra mã thoát và nguồn report. Đây là giới hạn đã biết, chưa sửa code trong đợt chuẩn hóa tài liệu.

## Nguồn và phạm vi

- Nhập từ nhánh QuocAn, đầu nhánh `32ca8b1`; không diễn giải kết quả cũ thành số đo sau chuẩn hóa.
- [Bản chính](../01_lab_3_description.md), [nhật ký đồng bộ](../../../BRANCH_SYNC_2026-11-11.md). Lần cập nhật hiện tại chỉ chuẩn hóa thông tin và phân loại tài liệu.
