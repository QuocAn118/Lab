# Project Roadmap & Progress Status — Lab 1

## 1. Task Breakdown & Delivery Pipeline

| Task ID | Work Item | Status | Owner | Deliverable & Evidence |
| :---: | :--- | :---: | :---: | :--- |
| **TASK-01** | Thiết lập kiến trúc tài liệu đặc tả `docs/` | **DONE** | Team | 7 file đặc tả hoàn chỉnh |
| **TASK-02** | Khởi tạo Data Pipeline & DataLoaders | **DONE** | TV1 (Codex hỗ trợ, chờ nhóm review) | `data/prepare_data.py`, `data/data_pipeline.py`; `data/reports/data_summary.json`, `batch_check.json`, `split_indices.json`, `data_samples.png` |
| **TASK-03** | Xây dựng kiến trúc nơ-ron (MLP/CNN) | **TODO** | Team | Lớp `nn.Module` forward flow |
| **TASK-04** | Thực thi Smoke Test (Dry Run) | **TODO** | Team | Xác thực tensor shapes |
| **TASK-05** | Huấn luyện Full Run & Quản lý Checkpoints | **TODO** | Team | Đồ thị loss & 5 checkpoints |
| **TASK-06** | Đo lường Benchmarks & Nghiệm thu | **TODO** | Team | Báo cáo `evaluation.md` |

---

## 2. Engineering Changelog
- `2026-11-10` — Theo yêu cầu người dùng, gom module 3.1–3.2 vào `data/` (code, `reports/`, `tests/`, dataset gốc). Cập nhật import trong train, đường dẫn báo cáo, Git ignore và công cụ lưu vết; giữ lịch sử nguồn gốc khi chuyển file. Tài liệu chung tiếp tục ở `docs/`.
- `2026-11-10` — Codex: triển khai 3.1–3.2, đồng bộ giao diện ba loader theo đặc tả chung. Tải FashionMNIST thật; split 54.000/6.000/10.000, giao train/val bằng 0; batch [64,1,28,28], float32 [0,1]. `python -m pytest "Lab 1/tests" "Lab 1/data/tests" -q`: 8 passed (gồm kiểm tra dữ liệu và train hiện có). Chưa chạy full training. Kiểm tra provenance toàn Lab còn lỗi có sẵn ở model.py, model_io.py và model_summary.txt; phần data đã đăng ký. Chờ review của thành viên nhóm.
- `2026-11-04`: Hoàn thiện khung tài liệu đặc tả hệ thống Module 01 theo chuẩn MLOps.
