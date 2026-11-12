# Lab 3 — báo cáo thực nghiệm

Mode: **smoke**; thiết bị: NVIDIA GeForce RTX 5060 Laptop GPU.
Smoke/quick không thay thế nghiệm thu full. Test không dùng chọn checkpoint.

| Chỉ số test | Giá trị |
| --- | ---: |
| test_loss | 0.695892 |
| test_model_preparation_time | 0.001000 |
| test_accuracy | 0.500000 |
| test_precision | 0.000000 |
| test_recall | 0.000000 |
| test_f1_binary | 0.000000 |
| test_f1_macro | 0.333333 |
| test_runtime | 0.029300 |
| test_samples_per_second | 546.694000 |
| test_steps_per_second | 34.168000 |

| Tiêu chí | Kết quả |
| --- | --- |
| test_accuracy_ge_0.85 | N/A: non-full run |
| test_macro_f1_ge_0.85 | N/A: non-full run |
| validation_accuracy_ge_0.85 | N/A: non-full run |
| latency_mean_le_25ms | True |
| gpu_peak_le_4GiB | True |
| host_observed_rss_le_6GiB | True |
| T4_epoch_le_180s | N/A: device is not Colab T4 |

## Epoch

| Epoch | Val loss | Val accuracy | Macro F1 | Wall seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.697144 | 0.5000 | 0.3333 | 1.29 |

![Loss](loss_curve.png)

![Confusion matrix](confusion_matrix.png)

Chi tiết số đo, phân lớp, latency, RAM/GPU và cấu hình: [metrics.json](metrics.json).
Nguồn/hash từng artifact: [artifacts.md](artifacts.md).

## Nguồn và phạm vi

- Đặc tả: `docs/08_implementation.md`, `docs/05_evaluation.md`, `docs/11_AI_Usage.md` (tính từ Lab 3/).
- Dữ liệu trực tiếp: `metrics.json`, `epoch_history.json`, `trainer_history.json` cùng run.
- Sinh bằng `lab3.py::render_report`; không sao chép số từ run khác.
