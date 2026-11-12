# Cấu Hình Huấn Luyện (Training Specification) — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Cấu Hình Huấn Luyện (Training Specification) — Lab 2 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T11:13:30+07:00 |
| Updated by — nguồn đặc tả/test nền | Nguyễn Quốc An (Quoc An), `3a86e93`, `00b8d04`, `9096223` |
| Người tích hợp/chuẩn hóa | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Người cập nhật nguồn TV3 | Nguyễn Thanh Bình (Thanh Binh), `7961f49` và `92930e6`; hỗ trợ OpenAI Codex theo hồ sơ nguồn |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Mã nguồn: `src/lab_2/training.py`. Tham số: `configs/lab_2.yaml`.

Nguồn đặc tả training và bộ test nền: Nguyễn Quốc An (Quoc An). Đây là ghi nhận đóng góp đặc tả/test, không chuyển phần triển khai package do HoangDang/OpenAI Codex chuẩn bị tại `4176139` hoặc vai trò TV3 của Nguyễn Thanh Bình sang Quốc An. Ghi nhận bổ sung do HoangDang/OpenAI Codex đối chiếu Git ngày 2026-11-12.

> **Phụ trách**: TV3. Đặc tả từng hàm đã triển khai nằm ở mục 8.

> Đã nhận một phiên lịch sử của TV3 gồm bốn cấu hình: [biên bản nhập](../reports/imported-thanhbinh-92930e6/IMPORT.json), [báo cáo gốc giữ nguyên](../reports/imported-thanhbinh-92930e6/training_report.md). Đủ 12 checkpoint, 21 source snapshot và ba manifest đã được đối chiếu; bản gốc dùng RTX 4060, không phải T4. `reports/metrics.json` tham số lịch sử của Bảo được giữ nguyên; metrics training gốc nằm trong thư mục nhập. Người chạy cũ chưa có tên trong manifest nên không thay bằng tên người commit.

Phiên chạy lại mới dùng config trong `configs/sessions/`, với reports/checkpoints riêng. Chạy bằng `python tools/run_tv3.py --stage train --config <config phiên> --output-dir <thư mục audit mới>`. Công cụ từ chối ghi đè run/log; snapshot code và config được lưu khi bắt đầu. Lần chạy lại là kết quả mới trên máy HoangDang, tách khỏi kết quả nguồn của Bình.

## 1. Siêu Tham Số Chung

| Tham số | Giá trị | Ghi chú |
| :--- | :---: | :--- |
| Hàm mất mát | `nn.CrossEntropyLoss()` | |
| Optimizer | `torch.optim.AdamW` | Chỉ nhận các tham số huấn luyện được |
| Weight decay | $10^{-4}$ | |
| LR scheduler | `ReduceLROnPlateau(mode="min", patience=2)` | Theo dõi validation loss |
| Số epoch tối đa | 6 | Có thể dừng sớm từ epoch 4 |
| Batch size khi đánh giá | 128 | |
| Seed | 42 | |
| Ghi train loss | Mỗi 50 batch | |

## 2. Các Lần Chạy

Mỗi lần chạy chỉ đổi một yếu tố so với mô hình gốc, nên có thể biết kết quả thay đổi do yếu tố nào.

| Lần chạy | Mục đích | Model | Learning rate | Batch size |
| :--- | :--- | :---: | :---: | :---: |
| `resnet18_baseline` | Mô hình gốc | ResNet18 | $10^{-3}$ | 64 |
| `densenet121` | Exercise 1, đổi model | DenseNet121 | $10^{-3}$ | 64 |
| `resnet18_lr1e-2` | Exercise 2, đổi learning rate | ResNet18 | $10^{-2}$ | 64 |
| `resnet18_bs128` | Exercise 2, đổi batch size | ResNet18 | $10^{-3}$ | 128 |

## 3. Quy Trình Huấn Luyện

Mỗi lần chạy là một cell trong notebook: `python -m lab_2.training --run <tên lần chạy>`. Trong mỗi epoch:

1. Đưa phần được mở khóa về chế độ `train`.
2. Với từng batch: xóa gradient, tính đầu ra và loss, lan truyền ngược, cập nhật trọng số.
3. Đánh giá trên tập validation.
4. Cập nhật checkpoint, log TensorBoard và scheduler.
5. Kiểm tra điều kiện dừng sớm.

Kết thúc lần chạy, nhật ký được ghi vào `training.runs.<tên lần chạy>` trong `reports/metrics.json`, giữ nguyên kết quả của các lần khác. Tập test không được dùng trong quá trình này.

## 4. Smoke Test
- **Cách chạy**: `python -m pytest tests/test_smoke.py -s`, trước khi train thật.
- **Quy mô**: 2 batch, mỗi batch 64 ảnh $(64, 3, 224, 224)$, dùng dữ liệu và trọng số thật.
- **Trường hợp kiểm tra**: ResNet18 đóng băng, DenseNet121 đóng băng, ResNet18 mở khóa `layer4`.
- **Tiêu chí đạt**:
  - Đầu ra có shape đúng $(64, 10)$.
  - Loss là số hữu hạn.
  - Gradient của lớp phân loại tồn tại và khác 0.
  - Tầng đang đóng băng không bị thay đổi.
  - Chạy đủ 2 batch, không hết bộ nhớ.
  - Khi có GPU, VRAM cao nhất không vượt 4,0 GB.

## 5. TensorBoard (Exercise 3)

- **Nơi lưu log**: `reports/runs/<tên lần chạy>/`. Không ghi đè log hoặc checkpoint của run đã tồn tại; dùng config/mã phiên mới để thử lại.
- **Cách mở**: Cell cuối của notebook chạy `%tensorboard --logdir reports/runs`. Trên máy cá nhân dùng `tensorboard --logdir reports/runs`.
- **Bằng chứng**: Biểu đồ TensorBoard không được lưu cùng notebook, nên cần chụp ảnh và lưu tại `reports/figures/tensorboard.png`.

| Chỉ số | Khi ghi | Ý nghĩa |
| :--- | :--- | :--- |
| `Loss/train` | Mỗi 50 batch | Theo dõi quá trình học |
| `Loss/val` | Mỗi epoch | Phát hiện overfitting |
| `Accuracy/val` | Mỗi epoch | So sánh các lần chạy |
| `LearningRate` | Mỗi epoch | Thấy được lúc scheduler giảm learning rate |

## 6. Checkpoint & Early Stopping
- **Checkpoint**: Xem chính sách trong `lab_2_description.md` mục 7. Mỗi file `.pth` lưu trọng số kèm tên model, dropout, epoch, val accuracy và val loss.
- **Early stopping**:
  - Một epoch chỉ được tính là cải thiện nếu val accuracy tăng **hơn** 1 điểm phần trăm so với mức tốt nhất đang tính.
  - Dừng sau 3 epoch liên tiếp không cải thiện.
  - Với 2.500 ảnh validation, 1 điểm phần trăm tương đương 25 ảnh.

## 7. Xử Lý Tình Huống

| Tình huống | Cách xử lý |
| :--- | :--- |
| Colab ngắt phiên | Kiểm tra JSON/log. Nếu run dở đã có output, giữ nguyên và tạo phiên mới; không resume giữa epoch. |
| Không có GPU | In cảnh báo và chạy bằng CPU. KPI tốc độ và VRAM ghi "Không đo được". |
| Một epoch đạt nhiều mốc tốt nhất | Mỗi mốc vẫn lưu trong file riêng, kể cả khi trọng số giống nhau. |
| Lỗi khi đang lưu checkpoint | Ghi vào file tạm rồi mới thay thế, nên mốc cũ không bị hỏng. |
| Tên lần chạy không có trong cấu hình | Dừng và in danh sách tên hợp lệ. |
| Chưa chạy đủ 4 lần | `evaluation.py` dừng và báo còn thiếu lần nào. |
| Chưa có `checkpoints/best.pth` | `demo.py` dừng và báo cần chạy `evaluation.py` trước. |

---

## 8. Đặc Tả Mã Nguồn (TV3)

**File đã triển khai**: `src/lab_2/training.py`. **Điều kiện trước**: TV5 đã đẩy `common.py`; TV1, TV2 đã đẩy `data.py`, `model.py`.
**Đẩy sớm**: `is_better` và `select_best_run` nên được đẩy trước, vì `evaluation.py` của TV4 cần import `select_best_run`.

**Import ở đầu file** (tên phải đúng vì test thay thế các tên này):

```python
from lab_2.common import argument_parser, load_config, peak_ram_gb, project_path, read_report, update_report
from lab_2.data import load_splits, make_loader
from lab_2.model import build_model, save_checkpoint, set_train_mode, trainable_parameters
```

Hằng số: `LIBRARIES = ["torch", "torchvision", "tensorboard", "scikit-learn"]`.

Giao diện triển khai và test tương ứng như dưới đây. Các test nằm trong `tests/test_training.py`, riêng `train_run` nằm trong `tests/test_integration.py`.

#### `is_better(candidate, current)`
- **Đầu vào**: Hai dict có `accuracy`, `loss`. `current` có thể là `None`.
- **Trả về**: `True` nếu `current` là `None`; accuracy cao hơn; hoặc accuracy bằng nhau và loss thấp hơn.
- **Test**: `test_is_better_prefers_accuracy_then_lower_loss`.

#### `select_best_run(runs)`
- **Đầu vào**: `{tên: kết quả}`. Dùng `best_val_accuracy` làm accuracy và `best_val_accuracy_loss` làm loss rồi so sánh bằng `is_better`.
- **Trả về**: Tên lần chạy tốt nhất. Không đọc số liệu của tập test.
- **Test**: `test_select_best_run_uses_validation_results_only`.

#### `class EarlyStopping`
- **`__init__(self, patience, threshold)`**: Lưu hai giá trị, `best = None`, `waited = 0`.
- **`update(self, accuracy)`**: Trả về `True` nếu phải dừng.
  - Lần đầu, hoặc `round(accuracy - best, 10) > threshold`: gán `best = accuracy`, `waited = 0`, trả về `False`.
  - Ngược lại: tăng `waited`, trả về `waited >= patience`.
- **Lý do cần `round`**: Trong máy tính `0.71 - 0.70` lớn hơn `0.01` một chút, nhưng mức tăng đúng 1 điểm phần trăm không được tính là cải thiện.
- **Test**: `test_early_stopping_stops_after_patience_without_enough_improvement`, `test_early_stopping_resets_when_improvement_exceeds_threshold`.

#### `find_run(config, name)`
- **Trả về**: Phần tử của `config["runs"]` có tên `name`. Không có thì `raise SystemExit` kèm danh sách tên hợp lệ.
- **Test**: `test_find_run_returns_the_run_or_lists_valid_names`.

#### `train_one_epoch(model, loader, criterion, optimizer, device, writer, log_every, step)`
- **Trả về**: `(loss trung bình theo batch, step mới)`.
- **Cách làm**: Gọi `set_train_mode(model)` đầu hàm. Với mỗi batch: chuyển dữ liệu sang `device`, `zero_grad`, tính loss, `backward`, `optimizer.step`, tăng `step` thêm 1. Nếu `step % log_every == 0` thì `writer.add_scalar("Loss/train", loss.item(), step)`.
- **Test**: `test_train_one_epoch_returns_mean_loss_and_logs_every_n_batches`.

#### `train_run(config, run)`
- **Trả về**: Dict `summary`, cũng là nội dung ghi vào báo cáo.
- **Import trong thân hàm**: `from lab_2.evaluation import evaluate_model`. Không đặt ở đầu file vì `evaluation.py` đã import `training.py`, sẽ gây lỗi import vòng tròn.
- **Chuẩn bị**:
  - Có GPU thì `device = "cuda"` và gọi `torch.cuda.reset_peak_memory_stats()`. Không có thì `device = "cpu"` và in cảnh báo.
  - `torch.manual_seed(config["seed"])`.
  - `train, val, _ = load_splits(config)`. Tạo loader train bằng `make_loader(train, run["batch_size"], config, shuffle=True)`, loader val với `config["training"]["eval_batch_size"]`.
  - Gọi đúng 2 đối số: `build_model(run["model"], config["model"]["dropout"])`, vì test thay hàm này bằng một hàm chỉ nhận 2 đối số.
  - Optimizer, scheduler theo mục 1. `EarlyStopping` lấy `early_stopping_patience`, `early_stopping_threshold` từ `config["training"]`.
  - Nếu run đã có summary, checkpoint hoặc log thì báo `FileExistsError`; không xóa kết quả cũ. Mở `SummaryWriter` cho thư mục log mới.
- **Mỗi epoch, từ 1 đến `max_epochs`, đúng thứ tự**:
  1. Bấm giờ `train_one_epoch`, truyền `log_every_batches`. Thời gian đánh giá không tính vào `train_seconds`.
  2. `result = evaluate_model(model, val_loader, device)`. Đọc learning rate từ `optimizer.param_groups[0]["lr"]`.
  3. Lưu `final_epoch`. Lưu `best_val_loss` nếu loss thấp hơn hẳn mức tốt nhất. Lưu `best_val_acc` nếu `is_better(result, ...)`. Đường dẫn là `<checkpoints_dir>/<tên lần chạy>/<mốc>.pth`, gọi `save_checkpoint` với `name=run["model"]`, `unfreeze=[]`.
  4. Ghi `Loss/val`, `Accuracy/val`, `LearningRate` theo epoch.
  5. `scheduler.step(result["loss"])`.
  6. Thêm vào `epochs` và in một dòng tóm tắt.
  7. `stopper.update(result["accuracy"])` là `True` thì ghi `stop_epoch` và dừng.
- **Kết thúc**: Tạo `summary` với các khóa trong `lab_2_description.md` mục 10. Đọc `read_report(config).get("training", {"runs": {}})`, gán `["runs"][run["name"]] = summary`, rồi `update_report(config, "training", ...)`, nhờ vậy các lần chạy khác được giữ nguyên.
- **Test**: `test_train_run_writes_marks_log_and_report_and_stops_early`, chỉ chạy được khi TV4 đã có `evaluate_model`, `copy_best`.

#### `main()`
Tạo parser bằng `argument_parser(__doc__)`, thêm `--run` bắt buộc. Gọi `train_run(config, find_run(config, args.run))` rồi in epoch tốt nhất, val accuracy, số giây/epoch.

**Tiêu chí đạt**: `python -m pytest tests/test_training.py -q` cho `6 passed`; `python -m pytest tests/test_smoke.py -q -s` cho `3 passed` (cần mạng để tải dữ liệu và trọng số).

### Lưu vết phiên chạy

Summary lưu thêm `config`, `operator` (biến môi trường `LAB2_OPERATOR`), `started_at`, `finished_at` (ISO 8601 có múi giờ) và `code_commit` (null nếu không có metadata Git). Khai báo người chạy trước khi train. Test tích hợp dùng output tạm, không tạo summary thực nghiệm trong repo.
