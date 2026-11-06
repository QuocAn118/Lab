# Bảng truy vết — Lab 3

| Yêu cầu | Nguồn Markdown | Triển khai | Kiểm chứng |
| --- | --- | --- | --- |
| R-01 Module A | `lab_3_description.md` §1; `model.md` §4 | `infer_pretrained`, `validate_texts` | CLI infer thật; test chuỗi rỗng |
| R-02 Dữ liệu | `data.md` §3 | `prepare_data`, `stratified_indices` | Chọn mẫu tái lập; run_config split/index hashes |
| R-03 Model/tokenize | `model.md` §4; `data.md` §3 | `train_run`, `prepare_data` | Smoke pretrained; test save/reload model nhỏ |
| R-04 Huấn luyện | `training.md` §4 | `RunConfig`, `train_run` | Tiny Trainer integration; smoke và full |
| R-05 Smoke | `training.md` §4 | `max_steps=2`, loss/grad checks | 2 bước optimizer; gradient hữu hạn khác 0 |
| R-06 Checkpoint/stop | `implementation.md` §3 | `StopPolicy`, `CheckpointCallback` | Ngưỡng/tie/NaN; checkpoint nạp được |
| R-07 Metrics/resources | `evaluation.md` §1–2 | `compute_metrics`, `measure_latency`, `plot_results` | Known-label metrics; báo cáo cùng run |
| R-08 Demo | `demo.md` §2–3 | `load_demo` | Save/reload safetensors; CLI demo |
| R-09 Nguồn MD | `AI_Usage.md` §2 | `record_provenance`, `audit_sources` | CLI audit; artifact hash |

Các hàm trong bảng thuộc `lab3.py`.

## Phạm vi notebook rút gọn

Theo `implementation.md` §6, notebook có một ô Exercise 1 độc lập và bảy ô Step 1–7 cho Exercise 2, theo thứ tự notebook kế thừa ở commit `755af75`. Giữ train/validation/test độc lập, model tốt nhất và nguồn Markdown. Chính sách 5 checkpoint, audit và benchmark chi tiết thuộc pipeline CLI đã có; không lặp lại trong nội dung notebook đơn giản.

## File do AI tạo/sửa trong lần này

| File | Nguồn chính | Vai trò |
| --- | --- | --- |
| `lab3.py` | `implementation.md` và bảng trên | Code chính |
| `test.py` | `implementation.md` §4; `verification.md` | Test offline |
| `requirements.txt`, `requirements-lock.txt`, `.gitignore` | `implementation.md` §4; `sources.md` | Môi trường/phạm vi gói nộp |
| `README.md` | `lab_3_description.md`, `implementation.md`, `verification.md` | Chạy và bàn giao |
| `AGENTS.md` | `AI_Usage.md`, `implementation.md` | Quy tắc chỉnh sửa về sau |
| `../Lab_3 (1).ipynb` | `implementation.md` §6 và bộ đặc tả | 8 ô code: Exercise 1 + 7 bước Exercise 2; 2 hàm nhỏ; API thư viện trực tiếp, không class tùy biến và không phụ thuộc lab3.py |
| `docs/*.md` | Mục Nguồn và phạm vi từng file | Đặc tả, nguồn và báo cáo |
| `outputs/<run>/**` | `artifacts.md`, `provenance.json` mỗi run | Bằng chứng; không đưa trọng số vào Git |

Căn cứ yêu cầu truy nguồn được ghi nhận tại `AI_Usage.md` §1. Liên kết hai chiều dùng điều hướng, không chứng minh nguồn gốc bằng tham chiếu vòng. Nguồn kỹ thuật cuối là tài liệu chính thức trong `sources.md`.

## Nguồn và phạm vi

- [implementation.md](implementation.md): R-01–R-09 chốt trước code.
- [AI_Usage.md](AI_Usage.md): quy tắc; [sources.md](sources.md): nguồn ngoài repo.
