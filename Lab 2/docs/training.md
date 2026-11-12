# Báo cáo TV3 — Huấn luyện và TensorBoard, Lab 2

Đặc tả kỹ thuật: [training_specification.md](training_specification.md).

Cập nhật review 2026-11-12: [checkpoint LFS, source khớp manifest và ghép main](TV3_REVIEW_RESOLUTION.md). Code hiện tại đã tích hợp main; source thực sự dùng cho bảng kết quả dưới đây được lưu riêng, không thay manifest cũ.

Ngày xuất: 2026-11-11T12:56:59+07:00. OpenAI Codex hỗ trợ triển khai, chạy và tổng hợp theo yêu cầu người dùng.

## Kết quả bốn cấu hình

Nguồn: `reports/metrics.json`, mục `training.runs`; so sánh bằng validation, không dùng test.

| Run | Epoch | Best epoch | Val accuracy | Val loss | Giây/epoch | Stop epoch | VRAM GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| resnet18_baseline | 5 | 4 | 84.80% | 0.4216 | 27.28 | 5 | 0.895 |
| densenet121 | 6 | 5 | 88.24% | 0.3455 | 64.67 | — | 1.311 |
| resnet18_lr1e-2 | 6 | 5 | 83.48% | 0.5671 | 28.61 | 6 | 0.895 |
| resnet18_bs128 | 6 | 6 | 85.08% | 0.4092 | 27.80 | — | 0.896 |

## Nhận xét Exercise 1 và 2

- `densenet121` chênh +3.44 điểm phần trăm validation accuracy so với baseline.
- `resnet18_lr1e-2` chênh -1.32 điểm phần trăm validation accuracy so với baseline.
- `resnet18_bs128` chênh +0.28 điểm phần trăm validation accuracy so với baseline.

Mỗi cấu hình chạy một lần với seed 42 trên NVIDIA GeForce RTX 4060 Laptop GPU. Kết quả không đại diện cho benchmark nhiều seed hoặc GPU T4 của Colab.

## Exercise 3 — TensorBoard

Cả bốn run có đủ `Loss/train`, `Loss/val`, `Accuracy/val`, `LearningRate`; số điểm theo epoch được đối chiếu với summary.

- Log: `reports/runs/<run>/`.
- Kiểm tra event: [tensorboard_check.json](../reports/tv3/tensorboard_check.json).
- Kiểm tra checkpoint: [checkpoint_check.json](../reports/tv3/checkpoint_check.json).
- Biểu đồ validation: [tv3_val_accuracy.png](../reports/figures/tv3_val_accuracy.png).
- [Ảnh chụp TensorBoard](../reports/figures/tensorboard.png), tạo bằng `tools/capture_tensorboard.py`.

## Kiểm chứng

- 41 test logic/integration bổ sung và ba smoke test pretrained đạt trên CPU; log: `reports/tv3/20261111-122708/`.
- 41 test đạt trong 8.98s và ba smoke test đạt trong 39.28s trên GPU; log: `reports/tv3/20261111-123359/`.
- Bốn run có trạng thái completed; log, môi trường và hash source: `reports/tv3/20261111-123519/manifest.json`.
- Ba mốc checkpoint mỗi run khớp epoch/validation metrics; đủ event TensorBoard theo số epoch thực tế.

## Checkpoint và bàn giao

Mỗi run có `final_epoch.pth`, `best_val_loss.pth`, `best_val_acc.pth` trong `checkpoints/<run>/`. Early stopping dùng patience 3 và mức tăng phải hơn 0.01. Checkpoint accuracy ưu tiên accuracy cao, hòa thì loss thấp.

Không chạy đánh giá test hoặc tạo `best.pth` trong phần TV3. Đủ 12 trọng số được bàn giao tại `checkpoints/<run>/` bằng **Git LFS**, giữ nguyên hash; xem [biên bản review](TV3_REVIEW_RESOLUTION.md). Run lịch sử chưa có schema config phiên mới của main; không tự điền thêm config/người chạy hoặc coi code mới là code đã train.

## Cách kiểm tra lại

```powershell
$env:PYTHONPATH = 'src'
& .venv/Scripts/python.exe tools/run_tv3.py --stage verify
& .venv/Scripts/python.exe tools/export_tv3.py
```

Log kiểm chứng và nguồn code được lưu trong `reports/tv3/<timestamp>/`. Công cụ train từ chối ghi đè run đã có. Chi tiết ghép phụ thuộc xem [TV3_TRAINING_GUIDE.md](TV3_TRAINING_GUIDE.md).
