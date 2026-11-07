# Báo cáo Lab 3

Mỗi lượt chạy có một thư mục riêng. Chạy notebook như bình thường; chương trình tự in đường dẫn report. Trọng số model được giữ trong `../outputs/`, không nhân bản vào report.

## Lượt đã lưu

| Report | Phạm vi |
| --- | --- |
| [Đánh giá model full](evaluate-20261107-091215-529975/summary.md) | Đo lại 25.000 mẫu test; [metrics](evaluate-20261107-091215-529975/metrics.json), [từng dự đoán](evaluate-20261107-091215-529975/predictions.csv) |
| [Inference pretrained](infer-20261107-091207-806869/summary.md) | Exercise 1, câu đầu vào và dự đoán thực |
| [Smoke mới](cli-smoke-20261107-091134-855303/summary.md) | 2 optimizer steps; kiểm tra ghi log, không dùng làm kết quả full |
| [Smoke cũ](imported-timing-cli-smoke-20261107011835/summary.md) | Bản sao bằng chứng cũ |
| [Smoke cũ đã kiểm tra](imported-timing-cli-smoke-20261107011835-checked/summary.md) | Bản sao bằng chứng cũ sau chỉnh cách ghi thời gian |
| [Full cũ, bằng chứng còn có](recovered-full-20261106/summary.md) | Thời lượng khôi phục từ tài liệu; ghi rõ file thiếu |

## Danh mục từng file

| File | Nội dung |
| --- | --- |
| `summary.md` | Loại lượt chạy, trạng thái và danh mục file |
| `run.json` | Thời điểm có múi giờ, trạng thái, phạm vi ghi log |
| `config.json` | Cấu hình model, dữ liệu và tham số thực tế |
| `environment.json` | Python, thư viện và môi trường chạy |
| `stdout.log` | Nội dung console được ghi trực tiếp |
| `stderr.log` | Cảnh báo, thanh tiến trình và nội dung stderr |
| `python.log` | Log Python/thư viện, có timestamp |
| `events.jsonl` | Sự kiện từng bước, có timestamp |
| `training_time.json` | Thời gian train, từng epoch và trạng thái |
| `trainer_history.json` | Lịch sử loss/metrics thực tế |
| `checkpoints.json` | Đường dẫn và tình trạng checkpoint |
| `metrics.json`, `predictions.csv` | Kết quả đo và từng nhãn dự đoán |
| `confusion_matrix.png`, `loss_curve.png` | Biểu đồ từ dữ liệu hiện có |
| `demo.json`, `inference.json` | Câu đầu vào và kết quả dự đoán |
| `error.json`, `traceback.txt` | Chi tiết lỗi khi có lỗi hoặc ngắt |
| `SOURCES.md`, `source_snapshot/` | Nguồn Markdown và bản sao code/đặc tả tại lúc chạy |
| `manifest.json` | Đường dẫn, kích thước, SHA-256 và nguồn từng file |

Chỉ tạo file phù hợp với thao tác thực hiện. Report đánh giá không được coi là lượt train mới. Notebook ghi log trong từng thao tác infer/train/evaluate; việc chuẩn bị dữ liệu/model trước thao tác không nằm trong log đó. `training_time.json` đo riêng `trainer.train()`; `run.json` có thể bao gồm cả thời gian xuất model và báo cáo.

Các thư mục `imported-*` sao chép bằng chứng cũ nguyên byte. `original_provenance.json` là manifest lịch sử của output gốc; `manifest.json` xác minh bản sao trong report hiện tại. `availability.json` ghi file không có ở output gốc. Thư mục `recovered-full-*` chỉ chứa phần còn có và thời gian khôi phục từ tài liệu; không có stdout/stderr hoặc thời điểm train gốc khi chưa có bằng chứng. Ngày nhập report không phải ngày train.

Kiểm tra hash từng file, chạy từ thư mục gốc dự án:

```powershell
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' audit --report 'Lab 3/report/<ten-luot-chay>'
```

## Nguồn và phạm vi

- [Thiết kế report](../docs/08_implementation.md), [đánh giá](../docs/05_evaluation.md), [thời gian train](../docs/04_training.md).
- [Quy tắc truy nguồn](../docs/11_AI_Usage.md), [bằng chứng kiểm chứng](../docs/06_verification.md).
- Tài liệu này mô tả file do chương trình thực sự tạo; không khẳng định mọi lượt chạy đều có toàn bộ file trong bảng.
