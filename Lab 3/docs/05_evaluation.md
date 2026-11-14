# Đánh giá và tiêu chí nghiệm thu — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Đánh giá và tiêu chí nghiệm thu — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Đo lường

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

Xem [06_verification.md](06_verification.md) cho kết quả thực tế. Không suy ra kết quả mới từ output cũ. Notebook chính giữ output của lần Run All ngày 2026-11-12; sửa tài liệu hoặc di chuyển notebook không tạo ra phép đánh giá mới.

## 3. Báo cáo theo từng lượt chạy

Mỗi lần chạy được ghi riêng trong `report/<run-id>/`. File log console, cấu hình, lịch sử train, thời gian, kết quả test, dự đoán từng mẫu và ảnh được tách theo tên tại [08_implementation.md §9](08_implementation.md). Dùng `summary.md` để đọc nhanh, `manifest.json` để kiểm hash. Không chép score smoke thành kết quả full và không coi đánh giá checkpoint có sẵn là train mới.

Các file được nhập từ output cũ có bản kê nguồn/hash. Phần full mất log chỉ có dữ liệu còn giữ hoặc số đã báo cáo; danh sách thiếu được ghi rõ. Report mới ghi stdout/stderr trong phạm vi train/đánh giá, không khẳng định đã thu được console trước khi thao tác bắt đầu.

## 4. Đánh giá lại ngày 2026-11-12

Đã chạy notebook tám cell với checkpoint full nhập tại `outputs/imported-thanhbinh-40acced/full/checkpoints/best`, trên RTX 5060 Laptop. Tập test gồm 25.000 mẫu, accuracy 0,91012, macro F1 0,9101182914, binary F1 0,9097264071 và loss 0,3465386629. Thời gian predict của lượt này là 48,3776 giây. Số đo và nguồn checkpoint được lưu trong report mới.

Confusion matrix có 11.431 / 1.069 ở hàng NEGATIVE và 1.178 / 11.322 ở hàng POSITIVE. Loss curve được vẽ từ lịch sử nằm cạnh checkpoint nhập, có đường dẫn nguồn trong `loss_history.json`. Hai ảnh đã được mở để kiểm tra bố cục. [Metrics](../report/evaluate-20261112-204226-805293/metrics.json), [loss history](../report/evaluate-20261112-204226-805293/loss_history.json), [manifest](../report/evaluate-20261112-204226-805293/manifest.json).

Người thực hiện HoangDang. Đây là đánh giá lại checkpoint có sẵn. Thời gian train lịch sử và KPI tham chiếu T4 được đọc theo đúng nguồn và thiết bị đo tương ứng.

## 5. Vị trí kết quả theo thao tác

| Thao tác | Kết quả cần đọc |
| --- | --- |
| CLI train | `outputs/<run>/metrics.json`, `test_predictions.csv`, `checkpoint_policy.json`; report có bản sao và manifest |
| CLI demo | `report/cli-demo-<run>/demo.json`; không tạo metrics.json của lượt train |
| Notebook Step 7 | `report/evaluate-<run>/metrics.json`, `predictions.csv`, `confusion_matrix.json`, `demo.json`; loss_history.json ghi nguồn loss cũ |
| Notebook Exercise 1 / CLI infer | `report/<run>/inference.json` và config.json |

Đường dẫn report được in khi thao tác chạy; xem [danh mục report](../report/README.md). Run ngày 2026-11-12 là nguồn số liệu đánh giá lại mới nhất đã lưu, không phải full training mới. Bản Quốc An có reports/metrics.json riêng và bộ KPI theo docs demo/evaluation.md; không trộn với report/ của bản chính.

## Nguồn và phạm vi

- Tiêu chí nghiệm thu kế thừa [01_lab_3_description.md](01_lab_3_description.md) và đặc tả đánh giá hiện có trong dự án.
- [08_implementation.md](08_implementation.md): phương pháp đo; [10_sources.md](10_sources.md): metric definitions.
- [11_AI_Usage.md](11_AI_Usage.md): chỉ báo kết quả có bằng chứng.
