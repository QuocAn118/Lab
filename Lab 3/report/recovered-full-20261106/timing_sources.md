# Nguồn thời gian được bổ sung

Các file training_time.json trong run này được khôi phục từ bảng thời gian ở docs/06_verification.md, không phải một lần train mới. Tổng thời gian 334.46 giây; các epoch 113.11, 111.24 và 109.95 giây đã được làm tròn trong báo cáo. Các thời điểm bắt đầu/kết thúc/lưu checkpoint chưa có log gốc nên để null.

Áp dụng cho training_time.json của run và các checkpoint có model.safetensors: best, best_val_acc, best_val_loss. best_val_pause hiện chỉ có config.json, không có trọng số; không bổ sung dữ liệu như một checkpoint hoàn chỉnh. Không sửa trọng số, tokenizer hoặc metric đã đo.

## Nguồn và phạm vi

- docs/06_verification.md §1–2 (tính từ Lab 3/): thời lượng và epoch chọn best.
- docs/04_training.md §6: quy tắc ghi thời gian, phân biệt log mới và số khôi phục từ báo cáo.
- docs/08_implementation.md: nguồn và phạm vi triển khai.
