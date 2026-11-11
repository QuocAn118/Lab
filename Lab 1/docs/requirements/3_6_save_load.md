# 3.6. Lưu và tải lại mô hình

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |

## Giao diện và đầu vào

- `save_model(model, run_id)` lưu `final_epoch.pth` theo cấu hình đã ghi trong `outputs/<run_id>/config.json`. Mã run được kiểm tra; mở file bằng `xb` để không ghi đè.
- `load_model(run_id, device, checkpoint="final_epoch")` hỗ trợ `final_epoch` hoặc `best`; từ chối lựa chọn khác. Cách gọi cũ giữ nguyên hành vi tải checkpoint cuối.
- `verify_reload(model, sample_batch, run_id, checkpoint="final_epoch")` yêu cầu tensor 16 ảnh validation đầu tiên `[16,1,28,28]`.

## Lưu, tải và kiểm chứng

1. Checkpoint chỉ chứa state_dict; config là nguồn kiến trúc, transforms và tên lớp. Không hỗ trợ resume optimizer.
2. Nạp với `weights_only=True`, `map_location=device`, `load_state_dict(..., strict=True)`, sau đó `eval()`.
3. So sánh trên CPU, tắt gradient, yêu cầu logits `[16,10]` hữu hạn; nhãn giống nhau và sai khác logits lớn nhất không quá `1e-6`.
4. Train kiểm final trước; tiếp theo khởi tạo model từ bản sao best-state độc lập giữ trong bộ nhớ và kiểm `best`. `reload_check.json` cuối ghi checkpoint best. Không so hai lần tải cùng một file để thay cho phép kiểm serialization.
5. Báo cáo có run_id, checkpoint, device, checked_samples, labels_match, max_abs_logit_difference, tolerance, passed. Không đạt thì Run thất bại và vẫn giữ artifact để truy vết.

Năm checkpoint và metadata nằm ở `checkpoints/<run_id>/`. `best` sao chép byte từ best_val_acc; final và pause cùng trọng số tại lần validation cuối là có chủ đích. Test/demo luôn chọn best.

## Kiểm tra

`tests/test_pipeline.py` kiểm best khác final, mặc định final còn tương thích, từ chối đường dẫn sai và không ghi đè checkpoint. `tests/test_train.py` kiểm train/early stopping/checkpoint/reload. Phép kiểm trên GPU được bổ sung bởi smoke/normal run thật; xem [verification.md](../verification.md).

## Nguồn và phạm vi

- [Đặc tả gốc](../lab_1_description.md), [toàn vẹn](../team_plan_integrity.md). Giữ tác giả code kế thừa trong Git; thay đổi API/kiểm chứng do HoangDang tích hợp với hỗ trợ OpenAI Codex.
