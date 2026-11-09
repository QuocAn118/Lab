# Phân công nhóm và bảo đảm toàn vẹn — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Phân công nhóm và bảo đảm toàn vẹn — Lab 1 |
| Created date | 2026-11-07 |
| Created by | HoangDang |
| Last updated | 2026-11-07 — thêm mục 9 Lưu vết |
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

**Lưu vết**
- [ ] `python "Lab 1/tools/check_provenance.py" --strict` báo PASS.
- [ ] Mọi run (kể cả smoke, lỗi) có manifest và đã được commit.
- [ ] Mọi file trọng số của run báo cáo có trên Drive, hash khớp manifest.
- [ ] Mỗi thành viên giải thích được module của mình.

## 8. Tài liệu bổ sung so với danh sách cơ bản

| File | Chủ file | Nội dung |
| --- | --- | --- |
| `team_plan_integrity.md` | TV5 | File này |
| `verification.md` | TV5 | Kết quả kiểm chứng thực tế từng run và giới hạn |
| `traceability.md` | TV5 | Yêu cầu → code → kiểm tra → bằng chứng |
| `AI_Usage.md` | TV5 (mọi người cùng ghi) | Khai báo sử dụng AI |
| `provenance.json` | TV5 (mọi người cùng ghi qua công cụ) | Sổ đăng ký nguồn gốc file, mục 9.2 |

## 9. Lưu vết

Mục tiêu: với **bất kỳ file nào** trong `Lab 1/` phải trả lời được: *ai tạo, lúc nào, từ đặc tả nào, bằng cách nào, đã sửa những gì*. Với **bất kỳ lần chạy nào**, kể cả smoke hoặc chạy lỗi, phải biết: *ai chạy, code nào, cấu hình nào, trên máy nào, ra file gì*.

| Cần lưu vết | Công cụ | Nơi lưu | Kiểm tra tự động |
| --- | --- | --- | --- |
| Mọi thay đổi nội dung | git: commit, pull request | Lịch sử git | `git log`, `git blame` |
| File nào tạo từ markdown nào | Sổ đăng ký + header `Spec:` trong code | `docs/provenance.json`, đầu mỗi file `.py` | `tools/check_provenance.py` |
| Mọi lần train / smoke | `src/run_tracking.py` — `Run` | `outputs/<run_id>/manifest.json` | `tools/check_provenance.py` |
| File đầu ra không thuộc run | `src/run_tracking.py` — `record_artifact` | `outputs/<file>.prov.json` | `tools/check_provenance.py` |

### 9.1. Thay đổi (git)

- Mỗi người commit bằng tài khoản của mình (`git config user.name`). Làm chung thì thêm dòng `Co-authored-by: Tên <email>` cuối commit.
- Commit nhỏ, mỗi commit một việc; message theo mẫu `loại(lab1-module): nội dung`, ví dụ `docs(lab1-data): cập nhật split 54k/6k`.
- Không để thay đổi chưa commit qua ngày. Không dùng `git push --force` và không sửa lịch sử đã push.
- Merge vào `main` qua pull request; mô tả PR ghi: đặc tả liên quan, phần AI sinh, người review, kết quả `check_provenance.py`.
- Changelog trong `plan_progress_status.md` ghi các thay đổi ảnh hưởng nhiều file hoặc thay đổi quyết định.

### 9.2. Nguồn gốc file (markdown → file)

Mọi file thuộc `README.md`, `.gitignore`, `docs/**/*.md`, `src/*.py`, `tools/*.py` phải có một mục trong `docs/provenance.json`:

```json
{
  "path": "src/model.py",
  "sources": [{"path": "docs/requirements/3_3_build_model.md", "sha256": "..."}],
  "method": "ai-assisted",
  "created_by": "TV2",
  "recorded_at": "2026-11-08T09:00:00+07:00",
  "status": "active",
  "note": "Sinh bằng <công cụ AI>, TV1 review",
  "reviews": [{"at": "...", "by": "TV2", "sources_changed": ["docs/requirements/3_3_build_model.md"], "note": "..."}]
}
```

- `sources` là file markdown (kèm mục `#§x` nếu cần) hoặc `external:<mô tả>` cho nguồn ngoài repo như đề bài, slide, hướng dẫn của cô.
- `sha256` là hash của nguồn **tại lần rà gần nhất**. Khi đặc tả đổi, công cụ báo `nguồn đã đổi` cho mọi file sinh từ nó; người phụ trách rà lại file đó rồi ghi nhận bằng `--ack`. Mỗi lần ghi nhận được thêm vào `reviews`.
- File code phải có dòng `Spec: <đường dẫn>` trong 15 dòng đầu cho mỗi nguồn; công cụ báo lỗi nếu header không khớp sổ đăng ký.
- Không sửa tay `provenance.json`; dùng lệnh:

```powershell
# Đăng ký file mới
python "Lab 1/tools/check_provenance.py" --add src/model.py --source "docs/requirements/3_3_build_model.md" --by TenBan --method ai-assisted --note "Sinh bằng <công cụ AI>, TV1 review"
# Ghi nhận đã rà lại sau khi đặc tả đổi
python "Lab 1/tools/check_provenance.py" --ack src/model.py --by TenBan --note "Cập nhật theo §6 mới"
# Kiểm tra toàn bộ
python "Lab 1/tools/check_provenance.py"
```

### 9.3. Lần chạy (train, smoke, đánh giá)

Mọi script tạo kết quả phải chạy trong `Run`:

```python
from run_tracking import Run

with Run("run_001", purpose="Cấu hình cơ sở", config=config,
         specs=["docs/requirements/3_4_train.md"]) as run:
    train(..., out_dir=run.out_dir, ckpt_dir=run.ckpt_dir)
    run.note("early stopping tại epoch 7")
```

`manifest.json` của mỗi run ghi:

| Nhóm | Trường |
| --- | --- |
| Định danh | `run_id`, `kind` (normal/smoke), `purpose`, `command` |
| Người và thời gian | `created_by`, `started_at`, `finished_at`, `duration_seconds` |
| Trạng thái | `status`: running → completed / failed / interrupted; `error` nếu lỗi |
| Code | `git.commit`, `git.branch`, `git.dirty`, `git.dirty_files`; hash từng file `src/*.py` và đặc tả nguồn của nó |
| Đặc tả và cấu hình | `specs` kèm hash; `config_sha256` |
| Môi trường | Python, PyTorch, torchvision, máy, CPU/GPU |
| Đầu ra | Đường dẫn, kích thước, SHA-256 của **mọi** file trong `outputs/<run_id>/` và `checkpoints/<run_id>/`; cờ `ignored_by_git` cho file trọng số |
| Sự kiện | `notes` ghi ngay lúc xảy ra, giữ lại kể cả khi run bị kill |

Quy tắc:

- **Normal run (`run_xxx`) chỉ chạy được khi code đã commit**; `Run` từ chối nếu còn thay đổi chưa commit. Smoke run được chạy khi chưa commit nhưng ghi lại danh sách file đang sửa.
- `run_id` đã tồn tại thì dừng, không ghi đè.
- **Không xóa run nào**, kể cả run lỗi hoặc smoke. Commit thư mục `outputs/<run_id>/` (và `checkpoint_index.json`) ngay sau khi chạy xong, với message `run(lab1): run_001 completed — <mục đích>`.
- Sửa, thêm hoặc xóa file trong run sau khi kết thúc sẽ bị công cụ phát hiện qua hash.
- File trọng số `.pth` không lên git; tải lên Drive chung theo đường dẫn `Lab1/checkpoints/<run_id>/` và ghi link vào `model.md`. Hash trong manifest dùng để xác nhận file trên Drive đúng là file đã chạy.

### 9.4. File đầu ra lẻ và file bị bỏ qua

- File trong `outputs/` không thuộc run nào (ví dụ `data_summary.json`, `compare_runs.csv`) phải gọi `record_artifact(path, specs=[...], inputs=[...])` ngay sau khi ghi. Tạo lại file thì lần cũ được giữ trong `history`.
- `.gitignore` của Lab 1 chỉ bỏ qua môi trường, cache, `data/` và file trọng số. Mọi file JSON/CSV/PNG/TXT trong `outputs/` đều được commit làm bằng chứng.

### 9.5. Tự động kiểm tra khi commit

Hook `.githooks/pre-commit` chạy `check_provenance.py` mỗi khi commit có file trong `Lab 1/` và chặn commit nếu có lỗi. Mỗi thành viên bật một lần sau khi clone:

```powershell
git config core.hooksPath .githooks
```

Không dùng `git commit --no-verify` để bỏ qua hook. Nếu buộc phải bỏ qua, ghi lý do vào changelog.
