# Lab 1 — Phân loại ảnh trang phục FashionMNIST bằng PyTorch

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Mô tả bài Lab 1 — FashionMNIST MLP |
| Created date | 2026-11-04 |
| Created by | HoangDang |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Chương tham chiếu | Chương 2 — Artificial Neural Networks |
| Trạng thái | Hoàn thành kỹ thuật: 17 test, smoke thật và bốn normal run PASS; có sản phẩm đầu vào/đầu ra và gói bàn giao cục bộ |

File này là tài liệu gốc của Lab 1. Các file khác trong `docs/` mô tả chi tiết từng phần và phải khớp với file này. Khi có mâu thuẫn, sửa theo file này hoặc cập nhật file này trước, rồi ghi lý do vào [plan_progress_status.md](plan_progress_status.md).

## 1. Bài toán

### 1.1. Bối cảnh

Phân loại ảnh là bài toán xác định ảnh thuộc lớp nào trong một tập lớp cho trước. Lab 1 dùng FashionMNIST để thực hành mạng neural nhiều lớp (MLP) và quy trình học có giám sát bằng PyTorch.

### 1.2. Vấn đề cần giải quyết

Xây dựng, huấn luyện và đánh giá một MLP nhận ảnh trang phục và dự đoán lớp tương ứng. Bài làm cần có đủ các bước: chuẩn bị dữ liệu, tiền xử lý, xây dựng mạng, huấn luyện, chọn checkpoint theo validation, đánh giá trên test, lưu và tải lại mô hình.

### 1.3. Đầu vào

| Nội dung | Giá trị |
| --- | --- |
| Dataset | FashionMNIST (torchvision) |
| Ảnh | Ảnh xám 28 × 28, một kênh |
| Số lớp | 10, nhãn từ 0 đến 9, giữ đúng ánh xạ tên lớp của dataset |
| Phân vùng gốc | 60.000 train, 10.000 test |
| Phân vùng dùng trong bài | 54.000 train, 6.000 validation (tách từ train gốc, seed 42), 10.000 test |

### 1.4. Đầu ra

- Mạng trả về 10 logits cho mỗi ảnh. Nhãn dự đoán là chỉ số có logit lớn nhất.
- Mỗi lần chạy có cấu hình, lịch sử train/validation, checkpoint, chỉ số test, dự đoán từng ảnh và báo cáo performance.
- Mọi kết quả của một lần chạy được lưu theo cùng một `run_id` để đối chiếu và tải lại.

## 2. Mục tiêu

1. **Hiểu cơ chế học của mạng neural:** lớp vào, lớp ẩn, lớp ra, trọng số, bias, hàm kích hoạt; giải thích được forward → loss → backward → cập nhật trọng số và vai trò của learning rate.
2. **Dùng các thành phần cơ bản của PyTorch:** tensor, Dataset, transforms, DataLoader, `nn.Module`, Autograd, optimizer, `state_dict`.
3. **Đánh giá đúng cách:** validation để chọn mô hình, test chỉ để đánh giá cuối; đo cả chất lượng, thời gian và tài nguyên.
4. **Làm việc có hồ sơ:** mỗi lần chạy truy vết được cấu hình, kết quả, người chạy và thời điểm.

## 3. Phạm vi

| Trong phạm vi | Ngoài phạm vi |
| --- | --- |
| MLP trên ảnh làm phẳng 784 chiều | CNN, transfer learning (thuộc các lab sau) |
| Transform `ToTensor` | Data augmentation |
| SGD (cơ sở) và các thí nghiệm ở mục 7 | Tìm siêu tham số tự động |
| 5 loại checkpoint ở mục 6 | Resume huấn luyện từ checkpoint (không lưu trạng thái optimizer) |
| Chạy trên CPU hoặc một GPU CUDA | Huấn luyện phân tán, triển khai web/API |

## 4. Hệ thống tài liệu

| File | Nội dung | Trạng thái |
| --- | --- | --- |
| `lab_1_description.md` | File này: bài toán, phạm vi, quy ước, tiêu chí hoàn thành | Đã chuẩn hóa |
| [data.md](data.md) | Dataset, phân vùng train/val/test, transforms, DataLoader | Đã đối chiếu code và báo cáo data |
| [model.md](model.md) | Hồ sơ model theo mục 9 | Đã có model trained và hồ sơ chọn theo validation |
| [training.md](training.md) | Siêu tham số, smoke run, normal run, checkpoint | Smoke và bốn normal run đã chạy/kiểm chứng |
| [evaluation.md](evaluation.md) | Chỉ số accuracy, thời gian, tài nguyên; cách đọc kết quả | Đã có kết quả test thật cho bốn run |
| [demo.md](demo.md) | Nạp checkpoint `best` để dự đoán ảnh mẫu | Đã có CLI, ảnh mẫu và ví dụ input/output thật |
| [plan_progress_status.md](plan_progress_status.md) | Công việc, trạng thái, nhật ký thay đổi | Đã cập nhật trạng thái và phần còn thiếu |
| [requirements/](requirements/) | Đặc tả chi tiết từng yêu cầu 3.1–3.6 | Đã đồng bộ; ghi rõ khoảng cách với code |
| [team_plan_integrity.md](team_plan_integrity.md) | Phân công 5 người, quy tắc toàn vẹn dữ liệu/kết quả/tài liệu, checklist nộp | Đã đồng bộ đường dẫn và giao diện |
| [AI_Usage.md](AI_Usage.md) | Ghi nhận hỗ trợ AI đã biết | Giữ nguồn tác giả và hỗ trợ triển khai |
| [verification.md](verification.md) | Lệnh kiểm tra và kết quả thực tế | Cập nhật sau từng giai đoạn |
| [traceability.md](traceability.md) | Yêu cầu → code → kiểm tra → sản phẩm | Đã có bản đồ triển khai |

Trạng thái nghiệm thu hiện tại:

- Model I/O hỗ trợ final/best. Train so sánh checkpoint `best` tải từ đĩa với bản sao trọng số tốt nhất giữ trong bộ nhớ trên 16 ảnh validation đầu tiên.
- Đã có evaluation, demo, đồ thị, compare và verifier. Bằng chứng chạy thật có trong `verification.md`: smoke và cả bốn normal run PASS; mỗi normal run đánh giá đủ 10.000 ảnh.
- Báo cáo data đã có trong `data/reports/`; dataset gốc trong `data/FashionMNIST/` không lưu trên Git, cần chuẩn bị lại trên máy chưa có dữ liệu.

## 5. Quy trình thực hiện

Mỗi phần việc đi qua các bước sau. Không chạy normal run khi smoke run chưa đạt.

| Bước | Làm gì | Kết quả cần có | Thư mục workflow |
| --- | --- | --- | --- |
| 1. Description | Đọc đề, viết mô tả bài toán và phạm vi | File này | `01-define`, `04-describe` |
| 2. Discussion | Chốt các lựa chọn còn mở (split, siêu tham số, checkpoint) với nhóm/cô | Quyết định ghi vào `plan_progress_status.md` | `02-design` |
| 3. Detail | Viết đặc tả từng yêu cầu: input, xử lý, output, đường dẫn, cách kiểm tra | `requirements/3_x_*.md` | `03-plan` |
| 4. Implementation | Viết code theo đặc tả, mỗi file một trách nhiệm | `src/*.py` | `05-generate`, `06-review` |
| 5. Small prompt | Giao cho AI từng nhiệm vụ nhỏ, có tiêu chí chấp nhận | Prompt và kết quả được review | `04-describe` |
| 6. Smoke run | Chạy thử với dữ liệu rất nhỏ để bắt lỗi | `outputs/smoke_xxx/` đạt kiểm tra | `07-run`, `08-test` |
| 7. Normal run | Huấn luyện đầy đủ với cấu hình đã chốt | `outputs/run_xxx/`, `checkpoints/run_xxx/` | `07-run` |
| 8. Check | Đối chiếu kết quả với file này và tiêu chí mục 11 | Cập nhật `evaluation.md`, `model.md` | `14-definition-of-done` |

### 5.1. Smoke run

- Mục đích: kiểm tra pipeline chạy hết từ đầu đến cuối, không dùng để báo cáo kết quả.
- Dữ liệu: 512 ảnh train, 256 ảnh validation, 256 ảnh test; 1 epoch; cùng code và cấu hình với normal run.
- Mã lần chạy: `smoke_` và ba chữ số, ví dụ `smoke_001`.
- Đạt khi: shape đúng, loss hữu hạn, đủ các file đầu ra ở mục 10, tải lại checkpoint `best` thành công.

### 5.2. Normal run

- Dùng toàn bộ 54.000 / 6.000 / 10.000 ảnh.
- Mã lần chạy: `run_` và ba chữ số, ví dụ `run_001`.
- Khởi tạo model mới; không dùng trọng số của smoke run.

## 6. Yêu cầu thực hiện (phần 3 của đề)

### Giao diện tích hợp hoàn thiện

`train.py` thực hiện toàn bộ pipeline trong cùng `with Run(...)`: huấn luyện → kiểm tra tải lại checkpoint cuối và `best` → `evaluate_run(config, run, test_loader, summary)` → `verify_run(run_id, finalized=False)` → ghi `verify_report.json` → chốt manifest. Sau khi chốt, `verify.py --run-id ...` chỉ đọc và kiểm hash, không sửa run.

`load_model(run_id, device, checkpoint="final_epoch")` và `verify_reload(model, sample_batch, run_id, checkpoint="final_epoch")` giữ cách gọi cũ; đối số `checkpoint="best"` dùng cho test/demo. Train giữ bản sao trọng số tốt nhất trong bộ nhớ để so sánh độc lập với checkpoint `best` tải từ đĩa. `reload_check.json` cuối cùng ghi phép kiểm tra `best`.

Demo độc lập `python src/demo.py --run-id run_001 --index 0` đọc ảnh test tại chỉ số đã chọn và in logits, nhãn dự đoán, nhãn thật; không thêm file vào run cũ. `compare_runs.py --run-ids ...` tổng hợp ít nhất ba normal run đã kiểm chứng, tạo CSV/đồ thị ở gốc `outputs/` kèm sidecar nguồn.

Các giao diện này là hợp đồng triển khai cho lần hoàn thiện hiện tại; trạng thái nghiệm thu cuối ghi trong `verification.md`.

Chi tiết từng yêu cầu nằm trong `docs/requirements/`. Bảng dưới là tóm tắt.

| Yêu cầu | File đặc tả | File code dự kiến | Đầu ra chính | Kiểm tra |
| --- | --- | --- | --- | --- |
| 3.1. Tải và chuẩn bị dữ liệu | [3_1](requirements/3_1_prepare_data.md) | `data/prepare_data.py` | `data/FashionMNIST/raw/`, `data/reports/data_summary.json`, `data_samples.png`, `split_indices.json` | Đủ 60.000/10.000 mẫu, 10 lớp; split 54.000/6.000 không trùng; 16 ảnh mẫu có nhãn |
| 3.2. Transforms và DataLoader | [3_2](requirements/3_2_transforms_dataloader.md) | `data/data_pipeline.py` — `create_dataloaders(config)` | train/val/test loader trong bộ nhớ; `data/reports/batch_check.json` | Ảnh `[B, 1, 28, 28]` float32, nhãn `[B]` int64; train xáo trộn, val/test giữ thứ tự |
| 3.3. Xây dựng mạng | [3_3](requirements/3_3_build_model.md) | `src/model.py` — `FashionMLP`, `build_model(model_config)` | `outputs/model_summary.txt` | Input `[2, 1, 28, 28]` → output hữu hạn `[2, 10]`; không có Softmax trước CrossEntropyLoss |
| 3.4. Huấn luyện | [3_4](requirements/3_4_train.md) | `src/train.py` | `config.json`, `history.csv`, các checkpoint ở mục 6.1 | Mỗi epoch có train và val; từ chối `run_id` đã tồn tại |
| 3.5. Đánh giá | [3_5](requirements/3_5_evaluate.md) | `src/evaluate.py` | `metrics.json`, `predictions.csv`, `performance.json` | Normal: 10.000 dòng; smoke: 256 dòng; accuracy tính lại từ CSV khớp JSON |
| 3.6. Lưu và tải lại | [3_6](requirements/3_6_save_load.md) | `src/model_io.py` — `save_model(model, run_id)`, `load_model(run_id, device)`, `verify_reload(model, sample_batch, run_id)` | `reload_check.json` (hiện kiểm tra `final_epoch`) | Nhãn giống nhau; chênh lệch logits tối đa ≤ 1e-6; `passed=true`; còn phải hỗ trợ và kiểm tra `best` |

### Cấu hình cơ sở

| Tham số | Giá trị |
| --- | --- |
| Kiến trúc | MLP: 784 → 512 → 512 → 10, ReLU (669.706 tham số) |
| Transform | `ToTensor` |
| Batch size | 64 |
| Loss | CrossEntropyLoss |
| Optimizer | SGD, learning rate 0.001, momentum 0.9 |
| Số epoch tối đa | 10 |
| Early stopping | patience 2, ngưỡng cải thiện val accuracy 0.01 (1 điểm %) |
| Seed | 42 (cho split, khởi tạo trọng số và xáo trộn) |
| Thiết bị | CUDA nếu có, nếu không dùng CPU; ghi thiết bị thực tế vào `config.json` |

Đây là lựa chọn triển khai của bài làm, không phải giá trị bắt buộc trong slide.

Early stopping so với val accuracy tốt nhất trước epoch hiện tại: tăng ít nhất 0.01 thì reset bộ đếm, ngược lại tăng bộ đếm; dừng khi bộ đếm đạt 2. Việc lưu `best_val_acc` vẫn ghi nhận mọi cải thiện dương; khi hòa giữ checkpoint sớm hơn. `epoch_seconds` và `elapsed_train_seconds` chỉ tính train, không tính validation hoặc kiểm tra tải lại.

### 6.1. Chính sách checkpoint

Validation được đánh giá sau mỗi epoch. Mỗi lần chạy lưu 5 checkpoint dạng `state_dict` (`.pth`):

| Checkpoint | Khi nào lưu | Ý nghĩa |
| --- | --- | --- |
| `final_epoch.pth` | Sau epoch cuối cùng đã chạy | Trọng số tại thời điểm kết thúc |
| `best_val_loss.pth` | Mỗi khi val loss thấp hơn mức tốt nhất trước đó | Model có val loss nhỏ nhất |
| `best_val_acc.pth` | Mỗi khi val accuracy cao hơn mức tốt nhất trước đó | Model có val accuracy lớn nhất |
| `best_val_pause.pth` | Tại lần validation cuối, kèm cờ `early_stopped` | Thời điểm dừng. Nếu 2 epoch liên tiếp val accuracy cải thiện dưới 0.01 thì dừng sớm (`early_stopped=true`); nếu chạy đủ epoch thì `early_stopped=false` |
| `best.pth` | Sau khi huấn luyện xong | Bản sao `best_val_acc.pth`; dùng cho đánh giá test và demo. Khi hòa, giữ checkpoint sớm hơn |

- `best_val_pause` và `final_epoch` cùng trọng số là có chủ đích: cả hai ghi trạng thái tại lần validation cuối; khác vai trò và metadata dừng sớm.
- Tất cả nằm trong `checkpoints/<run_id>/`.
- `checkpoints/<run_id>/checkpoint_index.json` ghi cho từng checkpoint: `epoch`, `val_loss`, `val_accuracy`, `saved_at` (ISO 8601), `elapsed_train_seconds`.
- Checkpoint chỉ chứa trọng số, không chứa trạng thái optimizer, nên không dùng để resume.
- Test **không** được dùng để chọn checkpoint.

## 7. Thí nghiệm và phân tích (phần 4 của đề)

### 7.1. Thử nghiệm kiến trúc hoặc siêu tham số

Mỗi thí nghiệm chỉ đổi **một** yếu tố so với `run_001`, các yếu tố còn lại giữ nguyên. Nghiệm thu tối thiểu gồm `run_001` và ít nhất hai run trong `run_002`–`run_004`; kế hoạch khuyến nghị chạy đủ bốn run.

| Run | Thay đổi so với cơ sở | Mục đích |
| --- | --- | --- |
| `run_001` | Cấu hình cơ sở | Mốc so sánh |
| `run_002` | Lớp ẩn `[256, 128]` | Ảnh hưởng của số tham số |
| `run_003` | Learning rate 0.01 | Ảnh hưởng của learning rate |
| `run_004` | Optimizer Adam, learning rate 0.001 | So sánh SGD và Adam |

Có thể thêm run nếu cần, nhưng phải ghi lý do vào `plan_progress_status.md` trước khi chạy.

### 7.2. Đồ thị loss

- Vẽ train loss và val loss theo epoch từ `history.csv`, lưu `outputs/<run_id>/loss_curve.png`.
- Đánh dấu epoch của `best_val_acc` và `best_val_loss` trên đồ thị.
- Vẽ thêm `outputs/compare_loss.png` để đặt val loss các run cạnh nhau.

### 7.3. Ảnh, nhãn dự đoán và nhãn thật

- Lấy 16 ảnh test cố định (seed 42), hiển thị nhãn thật và nhãn dự đoán; tô đỏ ảnh đoán sai.
- Lưu `outputs/<run_id>/prediction_samples.png`.

### 7.4. So sánh và nhận xét

Tổng hợp vào `outputs/compare_runs.csv` và bảng trong `evaluation.md` theo ba nhóm ở mục 8. Nhận xét cần trả lời: run nào tốt nhất trên validation, có overfitting không (val loss tăng trong khi train loss giảm), và đánh đổi giữa accuracy với thời gian/tài nguyên.

## 8. Đánh giá performance

Mọi run được đánh giá theo ba nhóm. Kết quả ghi vào `outputs/<run_id>/performance.json`.

| Nhóm | Chỉ số | Cách đo |
| --- | --- | --- |
| **Accuracy** | Test accuracy, test loss | Checkpoint `best` trên 10.000 ảnh test cho normal, 256 ảnh cho smoke; accuracy lưu dạng tỷ lệ 0–1 |
| | Best val accuracy, best val loss, epoch tương ứng | Từ `history.csv` |
| | Accuracy theo từng lớp | Tính từ `predictions.csv` |
| **Thời gian xử lý** | Thời gian bắt đầu, kết thúc, tổng thời gian train | Timestamp ISO 8601 và số giây (`time.perf_counter`) |
| | Thời gian mỗi epoch | Cột `epoch_seconds` trong `history.csv` |
| | Thời gian suy luận | Tổng thời gian đánh giá test và ms/ảnh; đồng bộ CUDA trước khi đo nếu dùng GPU |
| **Tài nguyên** | Số tham số, kích thước file `best.pth` (MB) | Đếm `numel()`; đọc kích thước file |
| | Bộ nhớ GPU đỉnh | `torch.cuda.max_memory_allocated()`; ghi `null` nếu chạy CPU |
| | Thiết bị và môi trường | Tên CPU/GPU, phiên bản Python, PyTorch, torchvision |

Thời gian đánh giá không được tính vào thời gian train. Không lấy ngày sửa file làm thời điểm huấn luyện.

## 9. Hồ sơ model (`model.md`)

`model.md` phải có các mục sau và được cập nhật sau mỗi lần huấn luyện được dùng để báo cáo:

1. **Thông tin:** Name; Created date (năm-tháng-ngày) và người tạo; Last updated và người cập nhật gần nhất.
2. **Purpose:** model dùng để làm gì, cho dữ liệu nào.
3. **Initial model:** kiến trúc, input/output (`[B, 1, 28, 28]` → `[B, 10]`), số tham số, file nguồn (`src/model.py`).
4. **Trained model:** run được chọn, ngày huấn luyện, mô tả hoàn thành (số epoch đã chạy, có dừng sớm không, val/test accuracy), đường dẫn lưu checkpoint.

## 10. Quy ước tổ chức và lưu trữ

- Mọi đường dẫn tính từ thư mục gốc `Lab 1`; code xác định đường dẫn từ vị trí file, không phụ thuộc thư mục đang chạy lệnh.
- Cây dưới là cấu trúc đích; các file ghi “dự kiến” chưa có trong repo.
- Tài liệu `.md`; mã nguồn `.py`; JSON và CSV dùng UTF-8.
- `outputs/<run_id>/` và `checkpoints/<run_id>/` phải chưa tồn tại khi bắt đầu; nếu trùng thì dừng, chọn mã mới, không ghi đè.
- Cùng một lần chạy dùng cùng `run_id` cho cấu hình, kết quả và checkpoint.
- `config.json` ghi thêm `created_by` (người chạy) và `started_at`.
- Lưu vết theo [team_plan_integrity.md §9](team_plan_integrity.md): mọi run chạy qua `src/run_tracking.py` và có `manifest.json`; mọi file đăng ký nguồn gốc trong `docs/provenance.json`; kiểm tra bằng `tools/check_provenance.py`.

```text
Lab 1/
├── README.md
├── .gitignore
├── tools/
│   └── check_provenance.py        # Kiểm tra lưu vết
├── docs/
│   ├── provenance.json            # Sổ đăng ký nguồn gốc file
│   ├── team_plan_integrity.md
│   ├── lab_1_description.md
│   ├── data.md
│   ├── model.md
│   ├── training.md
│   ├── evaluation.md
│   ├── demo.md
│   ├── plan_progress_status.md
│   ├── verification.md
│   ├── traceability.md
│   └── requirements/
│       ├── 3_1_prepare_data.md
│       ├── 3_2_transforms_dataloader.md
│       ├── 3_3_build_model.md
│       ├── 3_4_train.md
│       ├── 3_5_evaluate.md
│       └── 3_6_save_load.md
├── src/
│   ├── run_tracking.py            # Đã có: lưu vết run và file đầu ra
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py               # Test/performance và đồ thị
│   ├── verify.py                 # Kiểm số liệu và manifest
│   ├── demo.py                   # Ảnh mẫu và CLI suy luận chỉ đọc
│   ├── compare_runs.py           # CSV/đồ thị so sánh
│   └── model_io.py
├── tests/test_train.py            # Kiểm thử train bằng tensor giả
├── data/
│   ├── README.md
│   ├── prepare_data.py
│   ├── data_pipeline.py
│   ├── tests/test_data.py
│   ├── FashionMNIST/raw/         # Dữ liệu tải về, không commit
│   └── reports/                  # Mỗi báo cáo có <file>.prov.json
│       ├── data_summary.json
│       ├── split_indices.json
│       ├── data_samples.png
│       └── batch_check.json
├── outputs/                       # Mỗi file lẻ có <file>.prov.json đi kèm
│   ├── model_summary.txt
│   ├── compare_runs.csv
│   ├── compare_loss.png
│   └── <run_id>/                  # run_xxx hoặc smoke_xxx
│       ├── manifest.json          # Ai chạy, commit, đặc tả, môi trường, hash mọi file
│       ├── config.json
│       ├── history.csv            # epoch, train_loss, train_accuracy, val_loss, val_accuracy, epoch_seconds
│       ├── metrics.json
│       ├── predictions.csv
│       ├── performance.json
│       ├── reload_check.json
│       ├── verify_report.json    # Kiểm số liệu trước khi chốt manifest
│       ├── loss_curve.png
│       └── prediction_samples.png
└── checkpoints/<run_id>/          # Không commit file .pth
    ├── final_epoch.pth
    ├── best_val_loss.pth
    ├── best_val_acc.pth
    ├── best_val_pause.pth
    ├── best.pth
    └── checkpoint_index.json
```

## 11. Sản phẩm nộp và tiêu chí hoàn thành (phần 5 của đề)

| Sản phẩm | Nội dung | Hoàn thành khi |
| --- | --- | --- |
| 5.1. Code Python | `data/*.py`, `src/*.py` theo mục 6 | Smoke run và `run_001` chạy hết không lỗi từ thư mục `Lab 1` |
| 5.2. Báo cáo ngắn | `evaluation.md`: bảng so sánh mục 7.4 và nhận xét | Mọi số liệu lấy từ file trong `outputs/`, có dẫn đường dẫn |
| 5.3. Đồ thị loss | `loss_curve.png` mỗi run và `compare_loss.png` | Có cả train và val loss |
| 5.4. Ảnh dự đoán | `prediction_samples.png` | 16 ảnh, có nhãn thật và nhãn dự đoán |

Lab 1 được xem là hoàn thành khi:

- [x] Smoke run đạt các điều kiện ở mục 5.1.
- [x] `run_001` và ít nhất hai run trong `run_002`–`run_004` có đủ file đầu ra ở mục 10.
- [x] Mỗi run báo cáo có `verify_report.json` PASS; sau khi chốt manifest, kiểm tra hash ở chế độ chỉ đọc.
- [x] `reload_check.json` của mỗi run có `passed=true`.
- [x] `performance.json` có đủ ba nhóm accuracy, thời gian, tài nguyên.
- [x] `model.md`, `evaluation.md`, `plan_progress_status.md` đã cập nhật theo kết quả thực tế.
- [x] Đã đối chiếu kết quả với file này; mọi thay đổi so với đặc tả được ghi lý do.

## Nguồn và phạm vi

- Slide Chương 2 — Artificial Neural Networks (tài liệu môn học).
- [PyTorch Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) và [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
- [torchvision FashionMNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).
- Khung tài liệu, chính sách checkpoint và nhóm chỉ số performance theo hướng dẫn của cô và cách tổ chức đã dùng ở Lab 3.

## 12. Sản phẩm thực tế ngày 2026-11-11

Model được chọn bằng validation: **run_004**, val 88,10%, test 87,70%. Xem [báo cáo](evaluation.md), [kiểm chứng](verification.md), [demo input/output](demo.md). Thư mục `outputs/run_001`–`run_004` và `checkpoints/` đã có trên máy. ZIP cục bộ `deliverables/Lab1-complete-20261111.zip` chứa source/input/output/trọng số; không nằm trong Git. Không đánh đồng kiểm chứng kỹ thuật với việc đã upload Drive hoặc xác nhận danh sách thành viên.
