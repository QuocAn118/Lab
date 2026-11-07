# Hiệu chỉnh metadata dừng sớm

Full run chạy đủ 3 epochs, 3,750 optimizer steps. Điều kiện patience=2 kích hoạt ở đúng bước cuối nên lịch huấn luyện không bị rút ngắn. Bản callback đầu gộp hai ý này vào early_stopped. Bản hoàn thiện tách early_stopping_triggered=true và early_stopped=false.

Hiệu chỉnh chỉ cờ dừng của metrics.json, epoch_history.json và checkpoint_policy.json; không thay trọng số, split, loss, accuracy, F1, latency, RAM, GPU hoặc log gốc. Bản JSON trước hiệu chỉnh giữ ở metadata_originals/. lab3_source.py và code_sha256 vẫn là source thực đã chạy; postprocessing_code_sha256 trong manifest là code đã sửa cờ cho các lần sau. Kiểm thử riêng bao phủ dừng trước lịch và điều kiện tại bước cuối.

## Nguồn và phạm vi

- docs/implementation.md §3, docs/training.md §4 (tính từ Lab 3/): chỉ gọi là dừng sớm khi thực sự rút ngắn lịch.
- docs/AI_Usage.md §2: báo trung thực, lưu bằng chứng trước/sau.
- Lệnh đo và số liệu gốc trong run.log, run_config.json và metadata_originals/ cùng run.

## Hiệu chỉnh trình bày biểu đồ

Dùng chữ trắng ở ô màu tối của confusion matrix để đọc rõ. Hai PNG được render lại từ đúng trainer_history.json và confusion_matrix trong metrics.json; không thay số liệu. Nguồn: docs/evaluation.md R-07, docs/AI_Usage.md (tính từ Lab 3/).
