# Bảng truy vết — Lab 3

| Yêu cầu | Nguồn Markdown | Triển khai | Kiểm chứng |
| --- | --- | --- | --- |
| R-01 Module A | `01_lab_3_description.md` §1; `03_model.md` §4 | `infer_pretrained`, `validate_texts` | CLI infer thật; test chuỗi rỗng |
| R-02 Dữ liệu | `02_data.md` §3 | `prepare_data`, `stratified_indices` | Chọn mẫu tái lập; run_config split/index hashes |
| R-03 Model/tokenize | `03_model.md` §4; `02_data.md` §3 | `train_run`, `prepare_data` | Smoke pretrained; test save/reload model nhỏ |
| R-04 Huấn luyện | `04_training.md` §4 | `RunConfig`, `train_run` | Tiny Trainer integration; smoke và full |
| R-05 Smoke | `04_training.md` §4 | `max_steps=2`, loss/grad checks | 2 bước optimizer; gradient hữu hạn khác 0 |
| R-06 Checkpoint/stop | `08_implementation.md` §3 | `StopPolicy`, `CheckpointCallback` | Ngưỡng/tie/NaN; checkpoint nạp được |
| R-07 Metrics/resources | `05_evaluation.md` §1–2 | `compute_metrics`, `measure_latency`, `plot_results` | Known-label metrics; báo cáo cùng run |
| R-08 Demo | `07_demo.md` §2–3 | `load_demo` | Save/reload safetensors; CLI demo |
| R-09 Nguồn MD | `11_AI_Usage.md` §2 | `record_provenance`, `audit_sources` | CLI audit; artifact hash |

Các hàm trong bảng thuộc `lab3.py`.

## Phạm vi notebook rút gọn

Bổ sung Exercise 1 ngày 2026-11-07: nguồn `08_implementation.md` phần bổ sung Exercise 1 và `07_demo.md` §1; triển khai ô `lab3-exercise-0` trong notebook; 10 câu Việt/Anh, bảng kỳ vọng/dự đoán/score, gộp nhãn chỉ khi đối chiếu. Hướng dẫn nhận xét tại `huong_dan_trinh_bay.md`. Kiểm tra JSON và giữ nguyên bảy ô Exercise 2; chưa chạy suy luận mới do Python cục bộ không khả dụng. Output cũ của riêng ô Exercise 1 đã xóa, không sử dụng làm bằng chứng cho nội dung bổ sung.

Theo `08_implementation.md` §6, notebook có một ô Exercise 1 độc lập và bảy ô Step 1–7 cho Exercise 2, theo thứ tự notebook kế thừa ở commit `755af75`. Giữ train/validation/test độc lập, model tốt nhất và nguồn Markdown. Chính sách 5 checkpoint, audit và benchmark chi tiết thuộc pipeline CLI đã có; không lặp lại trong nội dung notebook đơn giản.

## File do AI tạo/sửa trong lần này

| File | Nguồn chính | Vai trò |
| --- | --- | --- |
| `docs/huong_dan_trinh_bay.md` | `08_implementation.md` §6–7; `06_verification.md`; đặc tả dữ liệu/model/training/evaluation | Thứ tự đọc 01–12, bảng tên cũ/tên mới và giải thích bảy bước; không bổ sung yêu cầu giảng viên |
| `lab3.py` | `08_implementation.md` và bảng trên | Code chính |
| `test.py` | `08_implementation.md` §4; `06_verification.md` | Test offline |
| `requirements.txt`, `requirements-lock.txt`, `.gitignore` | `08_implementation.md` §4; `10_sources.md` | Môi trường/phạm vi gói nộp |
| `README.md` | `01_lab_3_description.md`, `08_implementation.md`, `06_verification.md` | Chạy và bàn giao |
| `AGENTS.md` | `11_AI_Usage.md`, `08_implementation.md` | Quy tắc chỉnh sửa về sau |
| `../Lab_3 (1).ipynb` | `08_implementation.md` §6 và bộ đặc tả | 8 ô code: Exercise 1 + 7 bước Exercise 2; 2 hàm nhỏ; API thư viện trực tiếp, không class tùy biến và không phụ thuộc lab3.py |
| `docs/*.md` | Mục Nguồn và phạm vi từng file | Đặc tả, nguồn và báo cáo |
| `outputs/<run>/**` | `artifacts.md`, `provenance.json` mỗi run | Bằng chứng; không đưa trọng số vào Git |

Căn cứ yêu cầu truy nguồn được ghi nhận tại `11_AI_Usage.md` §1. Liên kết hai chiều dùng điều hướng, không chứng minh nguồn gốc bằng tham chiếu vòng. Nguồn kỹ thuật cuối là tài liệu chính thức trong `10_sources.md`.

## Nguồn và phạm vi

- [08_implementation.md](08_implementation.md): R-01–R-09 chốt trước code.
- [11_AI_Usage.md](11_AI_Usage.md): quy tắc; [10_sources.md](10_sources.md): nguồn ngoài repo.
