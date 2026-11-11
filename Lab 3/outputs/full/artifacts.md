# Nguồn artifact của run

Lệnh: `lab3.py train --mode full --output Lab 3/outputs/full`

Manifest gốc được ghi tại `provenance.json`, nhưng tệp đó hiện không có trong bản sao này; chưa thể kiểm tra đầy đủ hash từ các tệp hiện có.
Manifest không tự hash chính nó; artifacts.md là chỉ mục, tránh chuỗi hash đệ quy.

## Nguồn và phạm vi

Nguồn gốc tính từ thư mục Lab 3/: `docs/implementation.md`, `docs/AI_Usage.md`, `docs/data.md`, `docs/model.md`, `docs/training.md`, `docs/evaluation.md`, `docs/demo.md`.
Trọng số/cấu hình/tokenizer do thư viện lưu có định dạng nguyên bản; nguồn của chúng ghi ở manifest.

## Thời gian bổ sung

Xem [timing_sources.md](timing_sources.md) và [training_time.json](training_time.json). Số liệu được khôi phục từ báo cáo có nguồn, không phải timestamp đo lại.
