# Nguồn artifact của run

Lệnh: `lab3.py infer --text Bộ phim này rất hay! --text The movie was boring. --output Lab 3/outputs/module-a`

SHA-256, kích thước và nguồn từng file: [provenance.json](provenance.json).
Manifest không tự hash chính nó; artifacts.md là chỉ mục, tránh chuỗi hash đệ quy.

## Nguồn và phạm vi

Nguồn gốc tính từ thư mục Lab 3/: `docs/implementation.md`, `docs/AI_Usage.md`, `docs/data.md`, `docs/model.md`, `docs/training.md`, `docs/evaluation.md`, `docs/demo.md`.
Trọng số/cấu hình/tokenizer do thư viện lưu có định dạng nguyên bản; nguồn của chúng ghi ở manifest.
