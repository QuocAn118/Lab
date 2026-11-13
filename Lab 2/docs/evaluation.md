# Báo Cáo Đánh Giá (Performance Evaluation) — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Báo Cáo Đánh Giá (Performance Evaluation) — Lab 2 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-13T14:14:27+07:00 |
| Updated by | AI agent (thực thi yêu cầu kiểm tra của HoangDang) |
| Updated by — nguồn đặc tả/TV4 | Nguyễn Quốc An (Quoc An), `3a86e93`, `00b8d04`, `f9f2fd8` |
| Người cập nhật nguồn code evaluation | Nguyễn Quốc An (Quoc An), `32ca8b1`, 2026-11-10T21:18:59+07:00 |
| Người tích hợp/chuẩn hóa | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

> **Trạng thái hiện tại**: Phiên Colab Tesla T4 `colab-t4-20261113-browser-01` đã chạy đủ 4 cấu hình, evaluation và demo; notebook có 17 ô pipeline hoàn tất. Test accuracy **87,70%**, macro F1 **0,87573**; **7 KPI Đạt**. Kết quả T4 và bằng chứng ở mục bổ sung dưới đây. Các mục 1–7 giữ số liệu Windows và notebook lịch sử theo phiên đã ghi.

Mã nguồn: `src/lab_2/evaluation.py`.

**Ghi nhận nguồn TV4:** Nguyễn Quốc An đưa `evaluation.py` vào repo tại `32ca8b1`; bộ test evaluation có trong `00b8d04`. Theo khai báo của Quốc An trong tiến độ, code evaluation được lấy từ nhánh `lab2-ban-day-du` có, đã đối chiếu đặc tả, chạy 6 test và xem 3 biểu đồ. Đây là khai báo nguồn được giữ nguyên, không phải xác nhận độc lập về nhánh riêng đó. HoangDang tích hợp vào package và thêm các kiểm tra bảo vệ tại `4176139`. Ghi nhận bổ sung ngày 2026-11-12 do HoangDang đối chiếu Git; không thay tác giả nguồn hoặc ghi số đo mới.

> **Phụ trách**: TV4. Đặc tả từng hàm đã triển khai nằm ở mục 8. Các mục 1–7 ghi số thật của phiên Windows; phần bổ sung Colab dùng phiên riêng và số liệu T4 đã kiểm chứng.

> Đã chạy bốn cấu hình, chọn bằng validation rồi đánh giá test một lần và demo trong phiên riêng. Các đường dẫn output trong đặc tả là tương đối với `reports_dir`/`checkpoints_dir` của config; không dùng `reports/metrics.json` tham số lịch sử ở gốc để thay cho metrics phiên.

## Bổ sung Colab Tesla T4 — 2026-11-13

Nguồn [metrics phiên T4](../reports/sessions/colab-t4-20261113-browser-01/reports/metrics.json), [config](../configs/sessions/colab-t4-20261113-browser-01.yaml), [biên bản tiếp nhận](../reports/sessions/colab-t4-review-20261113-browser-01/intake_check.json). Source commit `735e5568f96819a9e52d25298ba5c34e213fa10b`; operator theo metrics là `OpenAI Codex, authorized by HoangDang`. Notebook có execution 1–17 cho các ô pipeline, ô xuất bổ sung thành công ở execution 19. [Notebook có output](../reports/sessions/colab-t4-review-20261113-browser-01/Lab_2_T4_verification_20261113.ipynb).

### So sánh Exercise 1 và 2

| Run | LR | Batch | Epoch thực tế | Best epoch | Validation accuracy | Giây train/epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `resnet18_baseline` | 0.001 | 64 | 5 | 4 | 84.80% | 43.68 |
| `densenet121` | 0.001 | 64 | 6 | 5 | 88.24% | 74.59 |
| `resnet18_lr1e-2` | 0.01 | 64 | 6 | 5 | 83.52% | 47.07 |
| `resnet18_bs128` | 0.001 | 128 | 6 | 6 | 85.08% | 48.81 |

Exercise 1: DenseNet121 cao hơn baseline **3,44 điểm phần trăm validation**, đạt 88,24% so với 84,80%. Thời gian train trung bình tăng từ 43,68 lên 74,59 giây/epoch trên T4 trong phiên này. Model tốt nhất được chọn bằng validation trước khi đánh giá test.

Exercise 2: ResNet18 với LR 0,01 đạt validation 83,52%, thấp hơn baseline **1,28 điểm phần trăm**. Batch 128 đạt 85,08%, cao hơn baseline **0,28 điểm phần trăm**, thời gian trung bình 48,81 giây/epoch so với 43,68 của baseline batch 64. Các so sánh dựa trên một seed 42 và cùng phiên T4.

### Test và KPI

DenseNet121 epoch 5 được đánh giá trên 5.000 ảnh test: accuracy **87,70%**, macro F1 **0,8757295828**, loss **0,3516814043**. Half-width khoảng tin cậy 95% của accuracy là **0,91 điểm phần trăm** theo công thức đang dùng.

| Chỉ số | Giá trị | Mục tiêu | Kết quả |
| --- | ---: | ---: | --- |
| Validation Accuracy | 88.24000% | 80% | Đạt |
| Test Accuracy | 87.70000% | 80% | Đạt |
| Test Macro F1 | 0.87573 | 0.8 | Đạt |
| Thời gian train (giây/epoch) | 74.59119 | 300 | Đạt |
| Độ trễ suy luận (ms/ảnh) | 20.38369 | 25 | Đạt |
| VRAM cao nhất (GB) | 1.31141 | 4 | Đạt |
| RAM cao nhất (GB) | 3.42210 | 6 | Đạt |

### Sản phẩm và giới hạn bằng chứng

- Metrics có đủ `data`, `model`, `training`, `evaluation`, `demo`; đủ 4 run và 13 checkpoint. [Manifest phiên](../reports/sessions/colab-t4-20261113-browser-01/FILES.json) đã kiểm 38 file, 10 source snapshot và hash trọng số; model best nạp CPU cho logits hữu hạn `[1,10]`.
- Bốn file event TensorBoard đã đối chiếu số epoch. [Ảnh TensorBoard](../reports/sessions/colab-t4-review-20261113-browser-01/tensorboard.png) hiển thị bốn run, Accuracy/val, Loss/train và Loss/val với smoothing 0. Ảnh được chụp từ giao diện cục bộ đọc event gốc T4.
- Các biểu đồ [validation](../reports/sessions/colab-t4-20261113-browser-01/reports/figures/val_accuracy_by_run.png), [loss](../reports/sessions/colab-t4-20261113-browser-01/reports/figures/loss_curve_best.png), [confusion matrix](../reports/sessions/colab-t4-20261113-browser-01/reports/figures/confusion_matrix.png) và [demo](../reports/sessions/colab-t4-20261113-browser-01/reports/figures/demo_probabilities.png) thuộc cùng phiên T4.
- Early stopping được tính lại từ các giá trị validation đã lưu với patience 3 và threshold 0,01, khớp cả bốn run. Baseline có `stop_epoch=5`; DenseNet121 và batch 128 chạy đủ 6 epoch; cấu hình LR 0,01 ghi `stop_epoch=6`, bằng giới hạn số epoch.
- AC-01: ô `run('lab_2.model')` có execution 6 nhưng `outputs=[]` trong notebook T4 đã lưu. Đã bổ sung [notebook Colab CPU và biên bản](../reports/sessions/model-architecture-colab-20261113/README.md), [stdout kiến trúc/từng tham số](../reports/sessions/model-architecture-colab-20261113/architecture.txt): một ô execution 1, không lỗi, cả ba cặp tổng/trainable khớp metrics T4. Source `model.py` trùng nguyên byte commit training; dùng `pretrained=False`, không train hoặc suy luận. Bằng chứng này chốt phần in kiến trúc của AC-01, không tạo kết quả train/benchmark T4 mới.
- `reports/metrics.json` ở gốc là artifact lịch sử chỉ có `model`. Báo cáo hiện tại đọc đường dẫn phiên ở trên theo config. Đặc tả mục 8 hiện đã mô tả `completed_runs`, loader rỗng/logits không hữu hạn, chặn evaluation lặp và config khác cấu hình train.

## 1. So Sánh Các Lần Chạy (Exercise 1 và 2) — Windows 2026-11-12

So sánh bằng validation, không dùng test. Một lần chạy mỗi cấu hình, seed 42; cùng RTX 5060 và cùng config phiên.

| Run | Model | LR | Batch | Epoch | Best epoch | Val accuracy | Val loss | Giây/epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `resnet18_baseline` | resnet18 | 0.001 | 64 | 5 | 4 | 84.76% | 0.4216 | 25.52 |
| `densenet121` | densenet121 | 0.001 | 64 | 6 | 5 | 88.24% | 0.3455 | 57.72 |
| `resnet18_lr1e-2` | resnet18 | 0.01 | 64 | 6 | 5 | 83.52% | 0.5671 | 25.31 |
| `resnet18_bs128` | resnet18 | 0.001 | 128 | 6 | 6 | 85.08% | 0.4092 | 25.32 |

Exercise 1: DenseNet121 cao hơn baseline 3.48 điểm phần trăm trong phiên này.
Exercise 2: LR 0,01 chênh -1.24 điểm; batch 128 chênh +0.32 điểm. Không kết luận thống kê nhiều seed từ một phiên.

[Đường validation theo epoch](../reports/sessions/rerun-thanhbinh-20261112-175308/reports/figures/val_accuracy_by_run.png). Log TensorBoard bốn run đã đối chiếu epoch/accuracy. Giao diện thật đã được chụp với smoothing 0 và hiển thị toàn bộ điểm dữ liệu. [Đối chiếu event](../reports/sessions/rerun-thanhbinh-20261112-175308/tensorboard_check.json), [ảnh TensorBoard](../reports/sessions/tensorboard-review-20261112-191218/tensorboard.png), [biên bản bổ sung](../reports/sessions/tensorboard-review-20261112-191218/REVIEW.json).

## 2. Đối Chiếu KPI

Nguồn: `evaluation.kpi`. Chất lượng và tài nguyên được đối chiếu với mục tiêu; hai KPI tốc độ tham chiếu GPU T4 chưa được nghiệm thu trên RTX 5060.

| Chỉ số | Giá trị | Mục tiêu | Kết quả |
| --- | ---: | ---: | --- |
| Validation Accuracy | 88.24% | 80% | Đạt |
| Test Accuracy | 87.68% | 80% | Đạt |
| Test Macro F1 | 0.8755 | 0.8 | Đạt |
| Thời gian train (giây/epoch) | Chưa đo trên T4 | 300 | Không đo được |
| Độ trễ suy luận (ms/ảnh) | Chưa đo trên T4 | 25 | Không đo được |
| VRAM cao nhất (GB) | 1.3124 | 4 | Đạt |
| RAM cao nhất (GB) | 2.6624 | 6 | Đạt |

Số đo thô trên RTX 5060 vẫn giữ: DenseNet121 57.72 giây train/epoch; latency trung bình 17.99 ms/ảnh, p95 20.64 ms/ảnh. Không coi đây là benchmark T4.

## 3. Kết Quả Trên Tập Test

Chọn DenseNet121 bằng validation; test không tham gia chọn model. Đã đánh giá một lần trong phiên mới trên 5.000 ảnh.

| Run được chọn | Epoch | Accuracy | Sai số 95% (điểm %) | Macro F1 | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `densenet121` | 5 | 87.68% | ±0.91 | 0.8755 | 0.3517 |

### Accuracy từng lớp

| Lớp | Accuracy |
| --- | ---: |
| airplane | 90.04% |
| automobile | 91.98% |
| bird | 80.39% |
| cat | 75.41% |
| deer | 80.53% |
| dog | 86.09% |
| frog | 94.85% |
| horse | 92.29% |
| ship | 89.86% |
| truck | 94.60% |

[Confusion matrix](../reports/sessions/rerun-thanhbinh-20261112-175308/reports/figures/confusion_matrix.png). Checkpoint gốc của Bình cũng được chọn bằng validation và kiểm lại riêng: [historical_checkpoint_recheck.json](../reports/sessions/rerun-thanhbinh-20261112-175308/historical_checkpoint_recheck.json); đây là phép kiểm tra mới trên trọng số cũ, không phải lần train mới hoặc kết quả test đã có trong báo cáo gốc.

## 4. Nhật Ký Lần Chạy Tốt Nhất

| Epoch | Train loss | Val loss | Val accuracy | LR | Giây train | Mốc lưu |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.7050 | 0.4438 | 85.32% | 0.001 | 58.02 | final_epoch, best_val_loss, best_val_acc |
| 2 | 0.4419 | 0.3764 | 86.96% | 0.001 | 58.52 | final_epoch, best_val_loss, best_val_acc |
| 3 | 0.4055 | 0.3641 | 87.48% | 0.001 | 56.95 | final_epoch, best_val_loss, best_val_acc |
| 4 | 0.3887 | 0.3551 | 87.80% | 0.001 | 56.89 | final_epoch, best_val_loss, best_val_acc |
| 5 | 0.3859 | 0.3455 | 88.24% | 0.001 | 57.61 | final_epoch, best_val_loss, best_val_acc |
| 6 | 0.3805 | 0.3459 | 87.92% | 0.001 | 58.34 | final_epoch |

Train loss là trung bình theo batch. DenseNet121 chạy đủ 6 epoch, không dừng sớm; checkpoint tốt nhất ở epoch 5.
[Đường loss](../reports/sessions/rerun-thanhbinh-20261112-175308/reports/figures/loss_curve_best.png).

## 5. Checkpoint

Đường dẫn dưới đây thuộc phiên mới, tách khỏi 12 trọng số gốc ở `Lab 2/checkpoints/`.

| Mốc | Trọng số | Epoch |
| --- | --- | ---: |
| `final_epoch` | [DenseNet121](../reports/sessions/rerun-thanhbinh-20261112-175308/checkpoints/densenet121/final_epoch.pth) | 6 |
| `best_val_loss` | [DenseNet121](../reports/sessions/rerun-thanhbinh-20261112-175308/checkpoints/densenet121/best_val_loss.pth) | 5 |
| `best_val_acc` | [DenseNet121](../reports/sessions/rerun-thanhbinh-20261112-175308/checkpoints/densenet121/best_val_acc.pth) | 5 |
| `best` | [Model được chọn](../reports/sessions/rerun-thanhbinh-20261112-175308/checkpoints/best.pth) | 5 |
| `best_val_pause` | Không có file riêng; DenseNet121 không dừng sớm | — |

[Đối chiếu 12 checkpoint theo run](../reports/sessions/rerun-thanhbinh-20261112-175308/checkpoint_check.json); `best.pth` là bản sao của best validation, tổng 13 file trọng số mới. [FILES.json](../reports/sessions/rerun-thanhbinh-20261112-175308/FILES.json) chốt hash toàn bộ artifact phiên.

## 6. Môi Trường Chạy

| GPU | torch | torchvision | tensorboard | scikit-learn |
| --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 5060 Laptop GPU | 2.9.1+cu128 | 0.24.1+cu128 | 2.21.0 | 1.9.1 |

Người chạy mới: HoangDang; code commit `5c03204`. Bắt đầu phiên 2026-11-12T18:01:16+07:00, hoàn tất 2026-11-12T18:22:42+07:00. Config và snapshot source có trong thư mục phiên; nguồn Bình dùng RTX 4060/torch 2.6.0+cu124 được giữ nguyên, không gán các số đo của phiên mới cho nguồn.

## 7. Lần Chạy Trước (Chỉ Để Tham Khảo)

Kết quả trong notebook cũ `notebooks/Lab 2.ipynb`. Lần chạy này khác đặc tả hiện tại nên **không dùng để nghiệm thu**: SGD, 2 epoch, ảnh $32 \times 32$, chuẩn hóa $(0.5, 0.5, 0.5)$, không có tập validation, so sánh các model trực tiếp trên toàn bộ 10.000 ảnh test.

| Model | Learning rate | Batch size | Test Accuracy |
| :--- | :---: | :---: | :---: |
| ResNet18 | 0,001 | 32 | 43,93% |
| DenseNet121 | 0,001 | 32 | 51,44% |
| ResNet18 | 0,01 | 32 | 38,56% |
| ResNet18 | 0,001 | 128 | 43,95% |

---

## 8. Đặc Tả Mã Nguồn (TV4)

**File đã triển khai**: `src/lab_2/evaluation.py`. **Điều kiện trước**: TV5 đã đẩy `common.py`; TV1, TV2 đã đẩy `data.py`, `model.py`; TV3 đã đẩy `select_best_run`.
**Người khác dùng lại**: `evaluate_model` (TV3), `style_axis`, `SERIES`, `SURFACE` (TV5).

**Đầu file**: Gọi `matplotlib.use("Agg")` trước khi import `matplotlib.pyplot`, vì `!python` trên Colab thừa hưởng backend inline của notebook.

**Hằng số**:

```python
LATENCY_SAMPLES = 200
LATENCY_WARMUP_RUNS = 5
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D")]
```

Bảng màu này đã qua validator. Mỗi màu đi với một hình marker riêng để người khó phân biệt màu vẫn đọc được biểu đồ.

Giao diện triển khai và test tương ứng như dưới đây. Các test nằm trong `tests/test_evaluation.py`.

#### `evaluate_model(model, loader, device)`
- **Trả về**: Dict `loss`, `accuracy`, `predictions` (list), `labels` (list).
- **Quy tắc**: Loader rỗng hoặc logits không hữu hạn báo lỗi. `model.eval()` và `torch.no_grad()`. `loss` là trung bình theo ảnh: cộng `cross_entropy(..., reduction="sum")` rồi chia cho số ảnh, vì batch cuối có thể nhỏ hơn.
- **Test**: `test_evaluate_model_computes_accuracy_loss_and_predictions`.

#### `kpi_status(value, target, higher_is_better)`
- **Trả về**: `"Không đo được"` nếu `value` là `None`; ngược lại `"Đạt"` hoặc `"Không đạt"`. Bằng đúng mục tiêu tính là đạt.
- **Test**: `test_kpi_status_handles_both_directions_and_missing_value`.

#### `completed_runs(config, report)`
- **Trả về**: `{tên: kết quả}` theo thứ tự trong `config["runs"]`, lấy từ `report["training"]["runs"]`.
- **Lỗi**: Thiếu lần nào thì `raise SystemExit` nêu tên lần đó. Báo cáo chưa có mục `training` cũng tính là thiếu.
- **Test**: `test_evaluation_stops_and_names_the_missing_runs`.

#### `measure_latency_ms(model, transform, samples, device)`
- **Đầu vào**: `samples` là list `(ảnh, nhãn)` từ `raw_samples`.
- **Trả về**: `{"mean": ..., "p95": ...}`, đơn vị mili giây.
- **Cách đo**: Chạy nháp `LATENCY_WARMUP_RUNS` ảnh đầu, không tính. Sau đó bấm giờ từng ảnh gồm `transform`, `unsqueeze(0)`, `.to(device)` và suy luận trong `no_grad`. Khi `device == "cuda"`, gọi `torch.cuda.synchronize()` trước khi dừng đồng hồ.
- **Test**: `test_latency_returns_mean_and_p95`.

#### `comparison_table(runs)`
- **Trả về**: List dict, một dòng mỗi lần chạy, có các khóa `run`, `purpose`, `model`, `learning_rate`, `batch_size`, `epochs_run`, `best_epoch`, `val_accuracy` (từ `best_val_accuracy`), `val_loss` (từ `best_val_accuracy_loss`), `seconds_per_epoch` (từ `mean_train_seconds_per_epoch`).
- **Test**: `test_comparison_table_has_one_row_per_run`.

#### `style_axis(axis, title, xlabel, ylabel)`
Nền `SURFACE`; tiêu đề căn trái màu `INK`; nhãn trục và số trên trục màu `MUTED`; lưới `GRID` nằm dưới dữ liệu; bỏ viền trên, phải, trái; viền dưới màu `GRID`.

#### `plot_loss_curve(run_name, run, path)`
Vẽ `train_loss` và `val_loss` theo epoch từ `run["epochs"]`, dùng `SERIES[0]`, `SERIES[1]`, có chú giải, trục y bắt đầu từ 0, đường dày 2, marker cỡ 8. Lưu với `dpi=150` rồi `plt.close`.

#### `plot_val_accuracy(runs, path)`
Mỗi lần chạy một đường `val_accuracy` theo epoch, dùng lần lượt các cặp trong `SERIES`, chú giải là tên lần chạy, trục x là số nguyên.

#### `plot_confusion_matrix(matrix, path)`
`imshow` với `cmap="Blues"`, hai trục ghi tên lớp, ô nào cũng ghi số. Chữ màu trắng nếu ô lớn hơn một nửa giá trị lớn nhất, ngược lại màu `INK`.

**Test của 3 hàm vẽ**: `test_plots_write_image_files`. Phải mở ảnh để xem bằng mắt, vì test chỉ kiểm tra file được tạo.

#### `copy_best(config, run_name)`
- **Trả về**: Đường dẫn `<checkpoints_dir>/best.pth`.
- **Cách làm**: `shutil.copyfile` từ `<checkpoints_dir>/<run_name>/best_val_acc.pth` sang file tạm, rồi `os.replace`.

#### `main()`
1. Chặn nếu báo cáo đã có `evaluation`; yêu cầu đủ 4 run và `config` trong từng summary khớp config hiện tại. Chọn `best_name = select_best_run(runs)` bằng validation.
2. Nếu có GPU thì `reset_peak_memory_stats`. Nạp `load_checkpoint(copy_best(config, best_name), device)`.
3. Lấy tập test từ `load_splits`, đánh giá với `make_loader(test, eval_batch_size, config)`.
4. Tính bằng `sklearn.metrics`, luôn truyền `labels=list(range(10))`, `zero_division=0` nếu hàm có tham số này:
   - `confusion_matrix`
   - `f1_score(average="macro")`
   - Accuracy từng lớp bằng `recall_score(average=None)`
5. Đo độ trễ bằng `measure_latency_ms(model, build_transform(config, train=False), raw_samples(test, LATENCY_SAMPLES), device)`.
6. Lập bảng KPI theo đúng thứ tự, mỗi dòng có `metric`, `value`, `target`, `status`:

| `metric` | `value` | `target` | Cao hơn là tốt |
| :--- | :--- | :--- | :---: |
| `Validation Accuracy` | `best_val_accuracy` của lần tốt nhất | `kpi.accuracy` | Có |
| `Test Accuracy` | Kết quả test | `kpi.accuracy` | Có |
| `Test Macro F1` | Kết quả test | `kpi.macro_f1` | Có |
| `Thời gian train (giây/epoch)` | `mean_train_seconds_per_epoch`; `None` nếu GPU train không phải T4 | `kpi.seconds_per_epoch` | Không |
| `Độ trễ suy luận (ms/ảnh)` | `latency["mean"]`; `None` nếu GPU đánh giá không phải T4 | `kpi.latency_ms` | Không |
| `VRAM cao nhất (GB)` | Lớn nhất của 4 lần chạy và bước này; `None` nếu không có số nào | `kpi.peak_vram_gb` | Không |
| `RAM cao nhất (GB)` | Lớn nhất của 4 lần chạy và `peak_ram_gb()` | `kpi.peak_ram_gb` | Không |

7. Vẽ vào `<reports_dir>/figures/`: `loss_curve_best.png`, `val_accuracy_by_run.png`, `confusion_matrix.png`. Tự tạo thư mục.
8. Ghi `update_report(config, "evaluation", ...)` với các khóa trong `lab_2_description.md` mục 10, rồi in bảng KPI và `Test Accuracy ± sai số`.

Kiểm chứng phần cứng ngày 2026-11-12: `evaluation.device_name` ghi tên thiết bị thật, `speed_kpi_reference` là `NVIDIA T4`. Trên RTX/CPU, hai KPI tốc độ mang trạng thái `Không đo được` theo chuẩn T4; số đo thô vẫn có ở `comparison.seconds_per_epoch` và `latency_ms`, không bị xóa hoặc đánh đồng phần cứng.

**Tiêu chí đạt**: `python -m pytest tests/test_evaluation.py -q` không có test lỗi. Kiểm lại ngày 2026-11-13 cùng `tests/test_handoff.py`: **25 passed**, 9,10 giây; không chạy lại training hoặc test set thật.
