# Báo Cáo Đánh Giá (Performance Evaluation) — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Báo Cáo Đánh Giá (Performance Evaluation) — Lab 2 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

> **Trạng thái**: Chưa có số liệu chính thức. Mọi ô "Chờ Colab" sẽ được điền từ `reports/metrics.json` sau lần chạy thật trên Colab.

Mã nguồn: `src/lab_2/evaluation.py`.

> **Phụ trách**: TV4. Đặc tả từng hàm cần viết nằm ở mục 8. Sau khi chạy Colab, TV4 điền số thật vào các mục dưới đây và viết nhận xét Exercise 1, 2.

## 1. So Sánh Các Lần Chạy (Exercise 1 và 2)

So sánh bằng tập validation, không dùng tập test. Nguồn: `evaluation.comparison`.

| Lần chạy | Mục đích | Model | LR | Batch | Số epoch đã chạy | Epoch tốt nhất | Val Accuracy | Val Loss | Giây/epoch |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `resnet18_baseline` | Mô hình gốc | ResNet18 | $10^{-3}$ | 64 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |
| `densenet121` | Exercise 1 | DenseNet121 | $10^{-3}$ | 64 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |
| `resnet18_lr1e-2` | Exercise 2 | ResNet18 | $10^{-2}$ | 64 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |
| `resnet18_bs128` | Exercise 2 | ResNet18 | $10^{-3}$ | 128 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |

- **Exercise 1**: Chờ Colab.
- **Exercise 2**: Chờ Colab.
- **Biểu đồ so sánh**: `reports/figures/val_accuracy_by_run.png` (có sau khi chạy).
- **Exercise 3**: `reports/figures/tensorboard.png` (ảnh chụp sau khi chạy).

## 2. Đối Chiếu KPI

Nguồn: `evaluation.kpi`. Điều kiện chạy: GPU T4, 22.500 ảnh train.

| Chỉ số | Giá trị đo | Mục tiêu | Kết quả | Cách đo |
| :--- | :---: | :---: | :---: | :--- |
| Validation Accuracy | Chờ Colab | $\ge 80.0\%$ | Chờ Colab | Của mốc `best` |
| Test Accuracy | Chờ Colab | $\ge 80.0\%$ | Chờ Colab | Mốc `best`, 5.000 ảnh test |
| Test Macro F1 | Chờ Colab | $\ge 0.80$ | Chờ Colab | Mốc `best`, 5.000 ảnh test |
| Thời gian train | Chờ Colab | $\le 300$ giây/epoch | Chờ Colab | Trung bình các epoch của lần chạy tốt nhất, không tính thời gian đánh giá |
| Độ trễ suy luận | Chờ Colab | $\le 25$ ms/ảnh | Chờ Colab | Tiền xử lý và suy luận từng ảnh trên GPU, trung bình 200 ảnh test, 5 lần chạy nháp đầu không tính |
| VRAM cao nhất | Chờ Colab | $\le 4.0$ GB | Chờ Colab | `torch.cuda.max_memory_allocated()`, đặt lại ở đầu mỗi tiến trình; lấy mức cao nhất của 4 lần chạy và đánh giá |
| RAM cao nhất | Chờ Colab | $\le 6.0$ GB | Chờ Colab | Mức cao nhất của tiến trình chính, không gồm tiến trình con của `DataLoader`; lấy mức cao nhất của 4 lần chạy và đánh giá |

## 3. Kết Quả Trên Tập Test

Tập test chỉ dùng một lần, trên mốc `best`. Nguồn: `evaluation.test_metrics`.

| Lần chạy được chọn | Epoch | Accuracy | Sai số (95%) | Macro F1 | Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |

Với 5.000 ảnh và accuracy quanh $80\%$, sai số khoảng $\pm 1{,}1$ điểm phần trăm. Nếu không đạt, vẫn ghi "Không đạt" và nêu lý do.

### Accuracy Từng Lớp

Nguồn: `evaluation.per_class_accuracy`.

| airplane | automobile | bird | cat | deer | dog | frog | horse | ship | truck |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |

### Confusion Matrix

Biểu đồ: `reports/figures/confusion_matrix.png` (có sau khi chạy).

## 4. Nhật Ký Lần Chạy Tốt Nhất

Số dòng phụ thuộc vào epoch dừng thực tế, tối đa 6 dòng. Nguồn: `training.runs`.

| Epoch | Train Loss | Val Loss | Val Accuracy | Learning Rate | Giây | Mốc đạt được |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Chờ Colab | | | | | | |

- **Train Loss** là trung bình trên toàn bộ batch của epoch.
- **Dừng sớm**: Chờ Colab.
- **Biểu đồ loss**: `reports/figures/loss_curve_best.png` (có sau khi chạy).

## 5. Checkpoint

| Mốc | Đường dẫn | Epoch | Giá trị |
| :--- | :--- | :---: | :---: |
| `final_epoch` | `checkpoints/<lần chạy>/final_epoch.pth` | Chờ Colab | |
| `best_val_loss` | `checkpoints/<lần chạy>/best_val_loss.pth` | Chờ Colab | Loss: Chờ Colab |
| `best_val_acc` | `checkpoints/<lần chạy>/best_val_acc.pth` | Chờ Colab | Accuracy: Chờ Colab |
| `best_val_pause` | Không có file | Chờ Colab | Có dừng sớm hay không |
| `best` | `checkpoints/best.pth` | Chờ Colab | Bản sao `best_val_acc` của lần chạy tốt nhất |

## 6. Môi Trường Chạy

Nguồn: `training.runs.<tên lần chạy>.device`, `library_versions`.

| GPU | torch | torchvision | tensorboard | scikit-learn |
| :--- | :---: | :---: | :---: | :---: |
| Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |

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

**File cần tạo**: `src/lab_2/evaluation.py`. **Điều kiện trước**: TV5 đã đẩy `common.py`; TV1, TV2 đã đẩy `data.py`, `model.py`; TV3 đã đẩy `select_best_run`.
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

Viết theo thứ tự dưới đây. Các test nằm trong `tests/test_evaluation.py`.

#### `evaluate_model(model, loader, device)`
- **Trả về**: Dict `loss`, `accuracy`, `predictions` (list), `labels` (list).
- **Quy tắc**: `model.eval()` và `torch.no_grad()`. `loss` là trung bình theo ảnh: cộng `cross_entropy(..., reduction="sum")` rồi chia cho số ảnh, vì batch cuối có thể nhỏ hơn.
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
1. `runs = completed_runs(config, read_report(config))`, `best_name = select_best_run(runs)`.
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
| `Thời gian train (giây/epoch)` | `mean_train_seconds_per_epoch`; `None` nếu `device` của lần đó là `"cpu"` | `kpi.seconds_per_epoch` | Không |
| `Độ trễ suy luận (ms/ảnh)` | `latency["mean"]`; `None` nếu không có GPU | `kpi.latency_ms` | Không |
| `VRAM cao nhất (GB)` | Lớn nhất của 4 lần chạy và bước này; `None` nếu không có số nào | `kpi.peak_vram_gb` | Không |
| `RAM cao nhất (GB)` | Lớn nhất của 4 lần chạy và `peak_ram_gb()` | `kpi.peak_ram_gb` | Không |

7. Vẽ vào `<reports_dir>/figures/`: `loss_curve_best.png`, `val_accuracy_by_run.png`, `confusion_matrix.png`. Tự tạo thư mục.
8. Ghi `update_report(config, "evaluation", ...)` với các khóa trong `lab_2_description.md` mục 10, rồi in bảng KPI và `Test Accuracy ± sai số`.

**Tiêu chí đạt**: `python -m pytest tests/test_evaluation.py -q` cho `6 passed`.
