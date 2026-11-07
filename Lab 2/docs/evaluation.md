# Báo Cáo Đánh Giá (Performance Evaluation) — Lab 2

> **Trạng thái**: Chưa có số liệu chính thức. Mọi ô "Chờ Colab" sẽ được điền từ `reports/metrics.json` sau lần chạy thật trên Colab.

Mã nguồn: `src/lab_2/evaluation.py`.

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
