# Bảng truy vết — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Bảng truy vết — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-06T22:10:18+07:00 ; HoangDang ; 77cd208 |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

| Yêu cầu | Nguồn Markdown | Triển khai | Kiểm chứng |
| --- | --- | --- | --- |
| R-01 Module A | `01_lab_3_description.md` §2; `03_model.md` §3.5 | `infer_pretrained`, `validate_texts` | CLI infer thật; test chuỗi rỗng |
| R-02 Dữ liệu | `02_data.md` §3 | `prepare_data`, `stratified_indices` | Chọn mẫu tái lập; run_config split/index hashes |
| R-03 Model/tokenize | `03_model.md` §3.2–3.3; `02_data.md` §2–3 | `train_run`, `prepare_data` | Smoke pretrained; test save/reload model nhỏ |
| R-04 Huấn luyện | `04_training.md` §4 | `RunConfig`, `train_run` | Tiny Trainer integration; smoke và full |
| R-05 Smoke | `04_training.md` §4 | `max_steps=2`, loss/grad checks | 2 bước optimizer; gradient hữu hạn khác 0 |
| R-06 Checkpoint/stop | `08_implementation.md` §3 | `StopPolicy`, `CheckpointCallback` | Ngưỡng/tie/NaN; checkpoint nạp được |
| R-07 Metrics/resources | `05_evaluation.md` §1–2 | `compute_metrics`, `measure_latency`, `plot_results` | Known-label metrics; báo cáo cùng run |
| R-08 Demo | `07_demo.md` §2–3; `08_implementation.md` §10 | `load_demo`, `src/exercise_2/demo.py` | Save/reload safetensors; CLI demo; chặn ghi đè mục demo đã có |
| R-10 Thời gian train/checkpoint | `04_training.md` §6 | `notebook_helpers.py`: `TrainingTime`, `train_and_save`; CLI `CheckpointCallback`, `train_run` | Hai lượt train riêng, đủ 3 epoch, lỗi giữa lượt; 5 checkpoint CLI có thời gian |
| R-09 Nguồn MD | `11_AI_Usage.md` §2 | `record_provenance`, `audit_sources` | CLI audit; artifact hash |
| R-11 Report riêng/từng file | `08_implementation.md` §9; `05_evaluation.md` §3 | `reporting.py`: `RunReport`, `collect_output`, `verify_report`; tích hợp notebook/CLI | `test_reporting.py`: console nguyên nội dung, hash, nhập log cũ, ngắt, train/evaluate model nhỏ; chạy notebook thực |
| R-12 Nạp checkpoint sau nhập nhánh | `08_implementation.md` §10 | `notebook_helpers.py`: `available_checkpoint`; notebook Step 3 | Config thiếu trọng số, checkpoint nhập còn đủ, pointer LFS và Run All trên dữ liệu thật |

Các hàm trong bảng thuộc `lab3.py`, trừ các tiện ích có ghi rõ file.

Hồ sơ model: [03_model.md](03_model.md) được rà soát theo [08_implementation.md §8](08_implementation.md), gồm metadata người/ngày, purpose, initial model (architecture, input/output, source file) và trained model (thay đổi sau train, kết quả hoàn tất, saving path). Tên HoangDang đã được xác nhận cho hai vai trò; ngày cập nhật hồ sơ được tách khỏi ngày huấn luyện trọng số. Kiểm tra đối chiếu trực tiếp config, số tham số và tình trạng checkpoint.


## Phạm vi notebook rút gọn

Theo `08_implementation.md` §6, notebook có một ô Exercise 1 độc lập và bảy ô Step 1–7 cho Exercise 2, theo thứ tự notebook kế thừa ở commit `755af75`. Giữ train/validation/test độc lập, model tốt nhất và nguồn Markdown. Chính sách 5 checkpoint, audit và benchmark chi tiết thuộc pipeline CLI đã có; không lặp lại trong nội dung notebook đơn giản.

## File do AI tạo/sửa trong lần này

| File | Nguồn chính | Vai trò |
| --- | --- | --- |
| `docs/docs demo/huong_dan_trinh_bay.md`, `docs/docs demo/README.md` | `08_implementation.md` §6–7, §11; `06_verification.md`; đặc tả dữ liệu/model/training/evaluation | Thứ tự đọc 01–12, phân biệt bản chính và tham khảo; giải thích bảy bước |
| `lab3.py` | `08_implementation.md` và bảng trên | Code chính |
| `notebook_helpers.py` | `08_implementation.md` §6; `04_training.md` §6; `05_evaluation.md` | Tách phần ghi thời gian/lưu model/vẽ loss khỏi notebook, giữ hành vi |
| `reporting.py`, `test_reporting.py` | `08_implementation.md` §9; `05_evaluation.md` §3; `04_training.md` §6 | Ghi report theo thao tác, kiểm hash và kiểm thử hành vi |
| `.gitattributes`, `.gitignore` | `08_implementation.md` §4, §9; `11_AI_Usage.md` §2 | Giữ byte log/report khi checkout; quản lý trọng số đã chọn qua LFS; loại optimizer/cache khỏi Git |
| `report/README.md`, `report/<run>/**` | `08_implementation.md` §9; `SOURCES.md`, `manifest.json` từng run | Danh mục từng file, nguồn/hash; giữ bằng chứng cũ riêng |
| `test.py` | `08_implementation.md` §4; `06_verification.md` | Test offline |
| `requirements.txt`, `requirements-lock.txt`, `.gitignore` | `08_implementation.md` §4; `10_sources.md` | Môi trường/phạm vi gói nộp |
| `README.md` | `01_lab_3_description.md`, `08_implementation.md`, `06_verification.md` | Chạy và bàn giao |
| `AGENTS.md` | `11_AI_Usage.md`, `08_implementation.md` | Quy tắc chỉnh sửa về sau |
| `Lab_3 (1).ipynb` | `08_implementation.md` §6 và bộ đặc tả | 8 ô code: Exercise 1 + 7 bước Exercise 2; hàm ngắn, thêm callback ghi thời gian; API thư viện trực tiếp và không phụ thuộc lab3.py |
| `docs/*.md` | Mục Nguồn và phạm vi từng file | Đặc tả, nguồn và báo cáo |
| `outputs/<run>/**` | `artifacts.md`, `provenance.json` mỗi run | Bằng chứng; trọng số đã chọn dùng Git LFS, giữ nguồn/hash của run |

Căn cứ yêu cầu truy nguồn được ghi nhận tại `11_AI_Usage.md` §1. Liên kết hai chiều dùng điều hướng, không chứng minh nguồn gốc bằng tham chiếu vòng. Nguồn kỹ thuật cuối là tài liệu chính thức trong `10_sources.md`.

## Bản nhập từ nhánh QuocAn — 2026-11-11

| File nhập | Nguồn đặc tả được giữ từ nhánh QuocAn | Phạm vi |
| --- | --- | --- |
| `src/exercise_1/sentiment_analysis.py` | `lab_3_description.md` | Exercise 1 riêng |
| `src/exercise_2/{common,data,model,training,evaluation,demo}.py` | `lab_3_description.md`, `data.md`, `model.md`, `training.md`, `evaluation.md`, `demo.md` | Pipeline Exercise 2 riêng; giữ nguyên code nguồn |
| `configs/exercise_2.yaml`, `requirements_quocan.txt`, `tests/`, `notebooks/` | `lab_3_description.md`, `plan_progress_status.md` | Cấu hình, môi trường, kiểm thử và notebook của bản nhập |
| `checkpoints/`, `reports/` | Commit gốc `7fea49d`, `186582e` và lịch sử trong nhật ký đồng bộ | Artifact kế thừa; không được diễn giải là kết quả chạy mới |

Đối chiếu tác giả, thời điểm commit, hash nhánh và quyết định giữ bản hiện hành tại [BRANCH_SYNC_2026-11-11.md](../../BRANCH_SYNC_2026-11-11.md). Phần nhập ngày 2026-11-11 do HoangDang tích hợp. Kiểm thử logic và smoke hai batch của bản Quốc An đã đạt trong lần kiểm 45 test ngày 2026-11-13; điều đó không đồng nghĩa đã chạy lại toàn bộ pipeline Quốc An hoặc Colab Run All trên môi trường mới.

Đổi tên ngày 2026-11-13 (T4 kế hoạch hoàn thiện của HoangDang), AI agent thực hiện: `notebooks/Lab_3_quocan.ipynb` thay tên cũ `Lab_3 (1).ipynb` trong thư mục `notebooks/`, giữ nguyên byte notebook nhập từ QuocAn; căn cứ `08_implementation.md` §7 và `11_AI_Usage.md` §2, kiểm bằng Git rename/hash, pytest và source audit. Notebook chính khi đổi tên T4 còn ở repo gốc; từ lần sắp xếp ngày 2026-11-14, vị trí hiện hành là Lab 3/Lab_3 (1).ipynb.

## Chuẩn hóa ngày 2026-11-14

| Phạm vi | Căn cứ | File và kiểm chứng |
| --- | --- | --- |
| Bản chính / tham khảo | 08_implementation.md §11; 01_lab_3_description.md §5 | README, bộ 01–12, docs demo/README.md và hướng dẫn trình bày; kiểm cấu hình và liên kết |
| Đường dẫn sau sắp xếp | 08_implementation.md §7, §11 | lab3.py audit, reporting.py snapshot_sources, test.py/test_reporting.py; header Sources của src/, tests/, config và requirements_quocan.txt |
| Checkpoint và kết quả | 03_model.md §4.3; 05_evaluation.md §5 | Model nạp từ snapshot nhập; metadata thời gian cũ giữ riêng; phân biệt demo.json với metrics.json |
| Tính toàn vẹn | 11_AI_Usage.md §2; 08_implementation.md §11 | Notebook và artifact lịch sử giữ nguyên byte so với trước chỉnh; kết quả tại 06_verification.md §8 |

Các nguồn trong bảng nhập Quốc An bên trên được giữ theo tên gốc của nhánh; vị trí hiện hành là `docs/docs demo/<tên-file>`. Bảy file đó và notebook Lab_3_quocan.ipynb giữ vai trò tham khảo; notebook Lab_3_colab.ipynb là bản điều phối pipeline src/ có output T4 lịch sử. Hướng dẫn trình bày trong docs demo vẫn thuộc bản chính.

## Nguồn và phạm vi

- Snapshot local ThanhBinh được nhập ngày 2026-11-11 tại [outputs/imported-thanhbinh-40acced/](../outputs/imported-thanhbinh-40acced/README.md). `IMPORT.json` giữ ánh xạ 176 file và nguồn commit; 20 trọng số dùng LFS. Đây là bản nhập lịch sử, không phải lần huấn luyện mới. Người tích hợp: HoangDang.

- [08_implementation.md](08_implementation.md): R-01–R-09 chốt trước code.
- [11_AI_Usage.md](11_AI_Usage.md): quy tắc; [10_sources.md](10_sources.md): nguồn ngoài repo.
