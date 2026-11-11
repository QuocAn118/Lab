# Project Roadmap & Progress Status — Lab 1

Last updated: 2026-11-10 — Codex (AI hỗ trợ, chờ nhóm review).

Trạng thái căn cứ vào code và bằng chứng đang có trong repo. “Có code” không đồng nghĩa đã nghiệm thu pipeline. Các mốc chạy trước đây trong changelog được giữ làm lịch sử, không phải kết quả chạy lại ở lần rà này.

## 1. Task Breakdown & Delivery Pipeline

| Task ID | Work Item | Status | Owner | Deliverable & Evidence |
| :---: | :--- | :---: | :---: | :--- |
| **TASK-01** | Đồng bộ đặc tả và hướng dẫn | **ĐÃ RÀ, CHỜ NHÓM REVIEW** | TV5 | Mô tả gốc, README, tài liệu thành phần, 3.1–3.6, phân công; ghi rõ phần chưa triển khai |
| **TASK-02** | Data Pipeline & ba DataLoader | **CÓ CODE VÀ BÁO CÁO** | TV1 | `data/prepare_data.py`, `data/data_pipeline.py`; `data/reports/` có summary, batch, split và ảnh mẫu kèm sidecar. Cần chuẩn bị dataset gốc trên máy trước khi chạy lại test |
| **TASK-03** | MLP và model I/O | **CÓ CODE, CÒN THIẾU** | TV2 | `FashionMLP`, `build_model`; summary 669.706 tham số. `load_model`/`verify_reload` chỉ dùng `final_epoch`, chưa hỗ trợ/kiểm tra `best`; provenance model còn thiếu |
| **TASK-04** | Smoke run toàn pipeline | **TODO** | TV3, TV5 | 512/256/256 mẫu, 1 epoch; đủ đầu ra và kiểm tra tải lại `best`; chưa có manifest nghiệm thu |
| **TASK-05** | Train + checkpoint và normal run | **CÓ CODE TRAIN; CHƯA CÓ NORMAL RUN** | TV3 | `src/train.py`, `tests/test_train.py`; train/val, early stopping, 5 checkpoint. Chưa có lịch sử hoặc manifest normal run để báo cáo |
| **TASK-06** | Evaluation, performance, đồ thị, demo và nghiệm thu | **TODO** | TV4, TV5 | Chưa có `src/evaluate.py`, `src/verify.py`, script đồ thị/demo, `metrics.json`, `performance.json`, `compare_runs.csv`; `verification.md`, `traceability.md` cũng chưa có |

## 1.1. Kiểm chứng lần rà tài liệu này

- Đã đọc code data/model/model I/O/train và kiểm thử hiện có; đối chiếu chữ ký hàm, cấu hình, checkpoint và đầu ra.
- Kiểm tra tĩnh đạt: 17 file Markdown không còn marker conflict, 47 liên kết nội bộ tồn tại, 7 file Python parse được, registry JSON hợp lệ và không trùng entry; chỉ số split đã lưu đủ 54.000/6.000, không giao nhau, phủ đủ 60.000 mẫu.
- Đã giải quyết marker merge conflict trong các file Lab 1 liên quan, giữ cấu trúc `data/` và lịch sử review. Git vẫn đang ở trạng thái merge chưa hoàn tất; chưa stage hay commit thay người dùng.
- `python -m pytest "Lab 1/tests/test_train.py" -q`: dừng ở bước thu thập test vì Python hiện tại thiếu `torch` (`ModuleNotFoundError`). Chưa chạy smoke/normal run hoặc tải lại dataset.
- `check_provenance.py --strict`: còn 3 lỗi có sẵn: `src/model.py`, `src/model_io.py` chưa đăng ký nguồn gốc; `outputs/model_summary.txt` thiếu sidecar. Không sửa tay báo cáo thực nghiệm hoặc tạo bằng chứng chạy thay cho lần chạy gốc.
- Hash nguồn trong sổ đăng ký được cập nhật sau khi rà; đây là xác nhận đối chiếu tài liệu/code, không phải xác nhận runtime đã pass. Phần `best`, evaluation và demo vẫn chưa hoàn thành.

---

## 2. Engineering Changelog
- `2026-11-10` — Codex: đồng bộ mô tả chính, README, data/model/training/evaluation/demo, yêu cầu 3.1–3.6 và phân công. Thống nhất `data/*.py`, `data/reports/`, ba loader, MLP `[512,512]`, smoke 512/256/256, tối đa 10 epoch, early stopping 2 epoch/0.01, 5 checkpoint và đường dẫn theo run. Gỡ marker conflict của Lab 1 để tài liệu và công cụ lưu vết đọc được; giữ lịch sử registry. Ghi rõ model I/O mới kiểm tra `final_epoch`, thiếu phần `best`/evaluation/demo, 3 lỗi provenance và giới hạn môi trường thiếu torch. Chờ nhóm review.
- `2026-11-10` — Theo yêu cầu người dùng, gom module 3.1–3.2 vào `data/` (code, `reports/`, `tests/`, dataset gốc). Cập nhật import trong train, đường dẫn báo cáo, Git ignore và công cụ lưu vết; giữ lịch sử nguồn gốc khi chuyển file. Tài liệu chung tiếp tục ở `docs/`.
- `2026-11-10` — Codex: triển khai 3.1–3.2, đồng bộ giao diện ba loader theo đặc tả chung. Tải FashionMNIST thật; split 54.000/6.000/10.000, giao train/val bằng 0; batch [64,1,28,28], float32 [0,1]. `python -m pytest "Lab 1/tests" "Lab 1/data/tests" -q`: 8 passed (gồm kiểm tra dữ liệu và train hiện có). Chưa chạy full training. Kiểm tra provenance toàn Lab còn lỗi có sẵn ở model.py, model_io.py và model_summary.txt; phần data đã đăng ký. Chờ review của thành viên nhóm.
- `2026-11-04`: Hoàn thiện khung tài liệu đặc tả hệ thống Module 01 theo chuẩn MLOps.
