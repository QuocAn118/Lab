# Phân công nhóm và bảo đảm toàn vẹn — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Phân công nhóm và bảo đảm toàn vẹn — Lab 1 |
| Created date | 2026-11-07 |
| Created by | HoangDang |
| Last updated | 2026-11-07 — bản đầu |
| Updated by | HoangDang |
| Áp dụng cho | Nhóm 5 người, toàn bộ Lab 1 |
| Tài liệu gốc | [lab_1_description.md](lab_1_description.md) |

## 1. Nguyên tắc

Theo hướng dẫn của cô: **không bắt buộc tự viết code**, nhưng nhóm phải **bảo đảm tính toàn vẹn của dữ liệu, kết quả và tài liệu**.

1. Code do AI sinh vẫn có người chịu trách nhiệm. Người phụ trách module phải giải thích được code và trả lời được khi bị hỏi.
2. Không có bằng chứng thì không ghi vào tài liệu. Số liệu chưa chạy ghi "chưa có", không ghi số ước lượng.
3. [lab_1_description.md](lab_1_description.md) là nguồn gốc. Muốn thay đổi đặc tả thì sửa file đó trước, rồi mới sửa code và các file khác.
4. Không sửa tay file trong `outputs/` và `checkpoints/`, không ghi đè run cũ. Kết quả sai thì chạy run mới với `run_id` mới.

## 2. Ba lớp toàn vẹn

| Lớp | Yêu cầu | Cách thực hiện | Bằng chứng |
| --- | --- | --- | --- |
| **Dữ liệu** | Dataset gốc không bị sửa | Tính SHA-256 các file IDX trong `data/FashionMNIST/raw/` | `outputs/data_summary.json` |
| | Split cố định, tái lập được | Chia 54.000/6.000 bằng seed 42, lưu chỉ số mẫu | `outputs/split_indices.json` |
| | Không rò rỉ dữ liệu | Train ∩ val = ∅; test không dùng để train hay chọn checkpoint | `outputs/data_summary.json` (trường `overlap_train_val = 0`) |
| **Kết quả** | Mọi con số đến từ run thật | Mỗi run có manifest gồm hash từng file, git commit, người chạy, thời điểm bắt đầu/kết thúc | `outputs/<run_id>/manifest.json` |
| | Số liệu tự nhất quán | Tính lại accuracy từ `predictions.csv` và so với `metrics.json`; kiểm tra hash khớp manifest | `outputs/<run_id>/verify_report.json` |
| | So sánh công bằng | Mọi normal run chạy trên cùng một máy; ghi thiết bị trong `performance.json` | `outputs/compare_runs.csv` |
| **Tài liệu** | Các file khớp nhau và khớp code | Mỗi số liệu trong docs có link tới file nguồn trong `outputs/` | Các file `docs/*.md` |
| | Truy vết được | Bảng yêu cầu → code → kiểm tra → bằng chứng | `docs/traceability.md` |
| | Biết ai sửa gì, vì sao | Mỗi file có bảng thông tin (mục 0); mọi thay đổi ghi changelog | `docs/plan_progress_status.md` |
| | Khai báo AI | Ghi phần AI sinh, prompt chính, người review, phần đã sửa | `docs/AI_Usage.md` |

## 3. Phân công

Mỗi người làm trọn vòng cho module của mình: **đặc tả → prompt nhỏ → AI sinh code → review theo đặc tả → chạy kiểm tra → lưu bằng chứng → cập nhật tài liệu**.

| Vai trò | Thành viên | Module | Code (AI hỗ trợ) | Tài liệu phụ trách | Kiểm chứng bắt buộc |
| --- | --- | --- | --- | --- | --- |
| TV1 — Data | | 3.1, 3.2 | `src/prepare_data.py`, `src/data_pipeline.py` | `data.md`, `requirements/3_1`, `3_2` | Hash file gốc; đủ 54.000/6.000/10.000; không trùng mẫu; batch `[B,1,28,28]` |
| TV2 — Model | | 3.3, 3.6 | `src/model.py`, `src/model_io.py` | `model.md`, `requirements/3_3`, `3_6` | Output `[2,10]`; 669.706 tham số; tải lại khớp, logits lệch ≤ 1e-6 |
| TV3 — Training | | 3.4, checkpoint | `src/train.py` | `training.md`, `requirements/3_4` | Smoke run đạt; đủ 5 checkpoint và `checkpoint_index.json`; từ chối `run_id` trùng |
| TV4 — Evaluation | | 3.5, performance | `src/evaluate.py` | `evaluation.md`, `requirements/3_5` | Đủ 10.000 dòng dự đoán; accuracy CSV khớp JSON; `performance.json` đủ 3 nhóm |
| TV5 — Lead và toàn vẹn | | Thí nghiệm, demo, kiểm chứng | `src/verify.py`, script vẽ đồ thị | `lab_1_description.md`, `demo.md`, `plan_progress_status.md`, `verification.md`, `traceability.md`, `AI_Usage.md` | Chạy `verify.py` cho mọi run; đối chiếu docs ↔ code ↔ outputs; checklist mục 7 |

Review chéo: TV1 ↔ TV2, TV3 ↔ TV4, TV5 review khi ghép. Không ai tự merge phần của mình.

## 4. Giao tiếp giữa các module

Chốt ở buổi họp đầu. Muốn đổi phải báo cả nhóm và TV5 cập nhật `lab_1_description.md` trước.

```python
# TV1
create_dataloaders(config) -> (train_loader, val_loader, test_loader)
# TV2
build_model(model_config) -> torch.nn.Module
save_model(model, path) -> None
load_model(config_path, checkpoint_path, device) -> torch.nn.Module
verify_reload(model, reloaded, batch) -> dict
```

| Thành phần dùng chung | Quy định |
| --- | --- |
| `config.json` | `schema_version`, `run_id`, `created_by`, `started_at`, `dataset`, `classes`, `model`, `data`, `training` |
| `history.csv` | `epoch, train_loss, train_accuracy, val_loss, val_accuracy, epoch_seconds` |
| `manifest.json` | `run_id`, `git_commit`, `created_by`, `started_at`, `finished_at`, danh sách `{path, sha256, bytes}` |
| Thư mục | `outputs/<run_id>/`, `checkpoints/<run_id>/` theo [lab_1_description.md §10](lab_1_description.md) |
| Đơn vị | Accuracy dạng tỷ lệ 0–1; thời gian tính bằng giây; timestamp ISO 8601 |

Để không phải đợi nhau: TV3 thử vòng train bằng DataLoader chứa tensor ngẫu nhiên; TV4 thử đánh giá bằng model chưa train; TV2 thử bằng `torch.randn(2, 1, 28, 28)`.

## 5. Các giai đoạn và điều kiện chuyển bước

| Giai đoạn | Việc | Chủ trì | Chuyển bước khi |
| --- | --- | --- | --- |
| 1. Description + Discussion | Đọc đề, chốt mục 4 và phân công | TV5 | Mục 4 được cả nhóm đồng ý, ghi vào changelog |
| 2. Detail | Mỗi người hoàn thiện `requirements/` của mình | Từng TV | Đặc tả có input, xử lý, output, đường dẫn, cách kiểm tra |
| 3. Implementation | Prompt nhỏ → AI sinh code → review → kiểm tra riêng module | Từng TV | Kiểm chứng bắt buộc ở mục 3 đạt; có khai báo AI |
| 4. Smoke run | Ghép code, chạy `smoke_001` | TV3, TV5 | `verify.py --run-id smoke_001` báo PASS |
| 5. Normal run | Chạy `run_001`–`run_004` trên **cùng một máy** | TV3 | Mỗi run có manifest và PASS |
| 6. Check | Tổng hợp, cập nhật docs, đối chiếu đặc tả | TV4, TV5 | Checklist mục 7 đạt |

## 6. Quy tắc làm việc

### 6.1. Git

- Mỗi người một nhánh theo module, ví dụ `lab1-data`, `lab1-model`.
- Merge qua pull request có người review theo mục 3.
- Không commit `data/` và `*.pth`. Hash trong manifest là bằng chứng thay cho file nặng.
- Commit message ghi rõ module và nội dung, ví dụ `feat(lab1-data): save split indices`.

### 6.2. Dùng AI

- Mỗi prompt chỉ giao một nhiệm vụ nhỏ, kèm đặc tả từ `requirements/` và tiêu chí chấp nhận.
- Đọc và chạy thử code AI sinh trước khi commit; không dán nguyên khối mà không review.
- Ghi vào `AI_Usage.md` theo mẫu:

| Ngày | Người | Module / file | Công cụ AI | Prompt chính (tóm tắt) | Đã review / sửa gì | Kiểm tra đã chạy |
| --- | --- | --- | --- | --- | --- | --- |
| | | | | | | |

### 6.3. Sửa tài liệu

- Mỗi người chỉ sửa file thuộc phần mình. Muốn sửa file người khác thì báo chủ file hoặc TV5.
- Sửa xong cập nhật dòng `Last updated` và `Updated by` ở mục 0 của file đó.
- Thay đổi ảnh hưởng nhiều file thì ghi vào changelog trong `plan_progress_status.md`.

## 7. Checklist trước khi nộp

TV5 chạy và đánh dấu; mỗi mục phải có bằng chứng đi kèm.

**Dữ liệu**
- [ ] Hash file IDX khớp giữa các máy của thành viên.
- [ ] `split_indices.json` tồn tại; train/val không trùng mẫu.
- [ ] Không dùng test ở bất kỳ bước chọn checkpoint nào (kiểm tra trong `train.py`).

**Kết quả**
- [ ] Mỗi run báo cáo có `manifest.json` và `verify_report.json` PASS.
- [ ] Các normal run cùng thiết bị trong `performance.json`.
- [ ] Không có run bị ghi đè; `run_id` không trùng.

**Tài liệu**
- [ ] Mọi số liệu trong `evaluation.md`, `model.md` có link tới file nguồn và khớp giá trị.
- [ ] `traceability.md` phủ đủ yêu cầu 3.1–3.6, mục 6.1, 7 và 8 của đặc tả.
- [ ] `AI_Usage.md` có đủ các phần do AI sinh.
- [ ] Mọi file `docs/` có bảng thông tin cập nhật; changelog đầy đủ.
- [ ] Mỗi thành viên giải thích được module của mình.

## 8. Tài liệu bổ sung so với danh sách cơ bản

| File | Chủ file | Nội dung |
| --- | --- | --- |
| `team_plan_integrity.md` | TV5 | File này |
| `verification.md` | TV5 | Kết quả kiểm chứng thực tế từng run và giới hạn |
| `traceability.md` | TV5 | Yêu cầu → code → kiểm tra → bằng chứng |
| `AI_Usage.md` | TV5 (mọi người cùng ghi) | Khai báo sử dụng AI |
