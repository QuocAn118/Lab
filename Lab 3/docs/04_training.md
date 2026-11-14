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
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Siêu tham số hiện hành

| Tham số | Giá trị bản chính |
| --- | --- |
| Framework / optimizer | Hugging Face Trainer, PyTorch, `adamw_torch` (AdamW) |
| Learning rate | 2e-5 |
| Batch train / eval | 16 / 16 |
| Số epoch tối đa | 3 |
| Weight decay / gradient clip | 0.01 / 1.0 |
| Seed | 42 |
| Max length / padding | 256 / động theo batch |
| Loss | Cross entropy trên logits |
| Precision | FP16 khi CUDA, FP32 khi CPU |

Đây là cấu hình đã chốt cho notebook chính và CLI full. Các lựa chọn cũ batch 32, 3–5 epoch hoặc chọn model theo loss không áp dụng cho bản chính. Pipeline Quốc An có cấu hình riêng, xem [training.md](docs%20demo/training.md).

## 2. Smoke, quick và full

CLI smoke dùng 32 train / 16 validation / 16 test, batch 16, `max_steps=2`: kiểm loss/gradient hữu hạn và khác 0, lưu/nạp checkpoint. Không dùng score smoke để kết luận chất lượng. Quick dùng 1.000/200/200, tối đa 3 epochs. Full dùng 20.000/5.000/25.000, tối đa 3 epochs và có thể dừng sớm.

Mỗi lần train là một run mới, khởi tạo từ pretrained; không dùng trọng số smoke cho full. Các mức dữ liệu nhỏ chỉ dùng kiểm cơ chế, không đổi cấu hình full đã chốt.

## 3. Chọn model và early stopping

Validation đánh giá theo epoch, chọn model có accuracy cao nhất; hòa accuracy giữ checkpoint sớm hơn trong bản chính. Early stopping có patience 2, ngưỡng cải thiện **hơn 0.01 accuracy tuyệt đối** (1 điểm phần trăm), so với best trước lần đánh giá hiện tại. Cải thiện nhỏ vẫn cập nhật best nhưng không reset bộ đếm patience.

`best_val_loss` là checkpoint tham khảo theo loss trong CLI, không thay tiêu chí chọn `best`. Run full lịch sử đủ 3 epochs: điều kiện patience kích hoạt ở bước cuối nhưng không rút ngắn lịch train, nên `early_stopped=false`.

## 4. Cấu hình pipeline CLI đầy đủ

CLI dùng `eval_strategy="epoch"`, `save_strategy="no"`, không bật `load_best_model_at_end` của Trainer. Callback tự lưu `final_epoch`, `best_val_loss`, `best_val_acc`, `best_val_pause`; sau train sao chép `best_val_acc` thành `best`. `final_epoch` giữ trọng số cuối trước khi nạp best. `best_val_pause` là lần validation cuối, không tự chứng minh đã dừng sớm.

Model và tokenizer được lưu để suy luận; không cam kết resume optimizer. Chi tiết callback, cờ dừng và đo epoch wall tại [08_implementation.md §3–5](08_implementation.md).

## 5. Notebook chính

Notebook `Lab 3/Lab_3 (1).ipynb` dùng Trainer và EarlyStoppingCallback trực tiếp: `eval_strategy=save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="accuracy"`, `greater_is_better=True`, `save_total_limit=2`. Không dùng callback quản lý năm checkpoint của CLI; callback thời gian nằm trong `notebook_helpers.py`.

Mặc định `TRAIN=False`: nạp checkpoint có trọng số từ `outputs/full/checkpoints/best`, sau đó thử snapshot nhập `outputs/imported-thanhbinh-40acced/full/checkpoints/best`. Thiếu trọng số thì báo lỗi và hướng dẫn tải Git LFS, không tự train. Để huấn luyện mới, đặt `TRAIN=True` và chạy từ Step 1. Model tốt nhất xuất ra `outputs/simple-training/<run>/best`; mỗi run có thư mục mới.

## 6. Ghi thời gian huấn luyện

Mỗi lượt train mới lưu `training_time.json` trong thư mục run: thời điểm bắt đầu/kết thúc có múi giờ, tổng số giây và trạng thái hoàn tất hoặc gián đoạn. `time.perf_counter()` đo riêng `trainer.train()`, gồm công việc validation/save nằm trong lệnh đó; không gồm tải dữ liệu/model trước train hoặc đánh giá test sau train.

Mỗi checkpoint có `training_time.json` ghi epoch, step, thời điểm lưu và số giây đã train tới lúc lưu. Thời lượng epoch của notebook tính đến callback sau khi lưu checkpoint; CLI ghi epoch wall trong callback validation, gồm các checkpoint do callback đó lưu. Tổng kết notebook giữ lịch sử mọi epoch đã chạy dù checkpoint bị xoay vòng; `best` có bản tổng kết run và đường dẫn checkpoint tốt nhất.

Nếu chỉ đánh giá checkpoint cũ thì đọc thời gian đã có, không tạo thời lượng train mới. Snapshot nhập full có trọng số và lịch sử loss nhưng không có `training_time.json`; metadata thời gian khôi phục nằm riêng tại `outputs/full/training_time.json`, với `status=recovered_from_documentation`. Các mốc chưa có bằng chứng để `null`, không suy từ ngày sửa file. Notebook có thể báo checkpoint nhập chưa có log thời gian; nguồn khôi phục xem [timing_sources.md](../outputs/full/timing_sources.md).

Tiện ích notebook gồm `train_and_save`, `show_training_time`, `plot_loss`; report được ghi theo từng thao tác, kể cả khi train bị ngắt. Xem [05_evaluation.md](05_evaluation.md) và [06_verification.md](06_verification.md).

## Nguồn và phạm vi

- [01_lab_3_description.md](01_lab_3_description.md), [08_implementation.md §1–6, §11](08_implementation.md): yêu cầu và quyết định triển khai của bản chính.
- [10_sources.md S-01–S-02](10_sources.md): Trainer và callback; [11_AI_Usage.md](11_AI_Usage.md): khai nguồn.
- Bản kế thừa trong Git giữ các lựa chọn ban đầu. Chuẩn hóa tài liệu không thay tham số, thuật toán, trọng số hoặc kết quả lịch sử.
