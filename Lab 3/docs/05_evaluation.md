# Đánh giá và tiêu chí nghiệm thu — Lab 3

## 1. Hợp đồng đo lường

Đánh giá `best_val_acc` trên validation và test sau khi nạp từ đĩa. Chỉ validation dùng chọn checkpoint/early stopping. Lưu test accuracy, precision, recall, binary F1 (POSITIVE), macro F1 (trung bình hai lớp), confusion matrix và dự đoán từng mẫu.

| Chỉ tiêu | Ngưỡng nội bộ | Điều kiện |
| --- | ---: | --- |
| Validation / test accuracy | >=85% | Full split |
| Test macro F1 | >=.85 | Full split |
| Epoch wall time | <=180 giây | Chỉ so trực tiếp Colab T4 |
| Inference mean latency | <=25 ms | Batch 1, tokenize/transfer/forward; 3 warm-up, 20 lần |
| Peak GPU allocated | <=4 GiB | PyTorch allocation, khác tổng VRAM hệ điều hành |
| Host observed RSS | <=6 GiB | Cao nhất tại callback; có thể bỏ lỡ đỉnh giữa hai mẫu |

Đơn vị bộ nhớ chốt GiB (1024³ bytes). Latency trên một review test có token count; không khái quát mọi độ dài. Epoch wall gồm train/eval/save; không gồm nạp lại và test sau train.

## 2. File bằng chứng mỗi run

- `run_config.json`: mode, seed, splits/counts/fingerprints/index hashes, revision và môi trường.
- `trainer_history.json`, `epoch_history.json`: loss/metric, quyết định callback.
- `metrics.json`: số đo và KPI true/false hoặc N/A. `test_predictions.csv` có dòng nguồn đầu (đọc bằng `comment='#'`).
- `loss_curve.png`, `confusion_matrix.png`, `report.md`: hình và báo cáo cùng run.
- `checkpoint_policy.json`: nguồn best, có thực sự early stop không.
- `provenance.json`, `artifacts.md`: nguồn và hash artifact.

Xem [06_verification.md](06_verification.md) cho kết quả thực tế. Không suy ra kết quả mới từ output notebook cũ; notebook hoàn thiện bỏ output cũ vì thiếu thông tin tái lập, có phần lệch code.

## Nguồn và phạm vi

- Tiêu chí nghiệm thu kế thừa [01_lab_3_description.md](01_lab_3_description.md) và đặc tả đánh giá hiện có trong dự án.
- [08_implementation.md](08_implementation.md): phương pháp đo; [10_sources.md](10_sources.md): metric definitions.
- [11_AI_Usage.md](11_AI_Usage.md): chỉ báo kết quả có bằng chứng.
