# Performance Evaluation & System Benchmarks — Lab 3

## 1. Training & Validation Progress Log

| Epoch | Train Loss | Val Loss | Val Accuracy | Val Precision | Val Recall | Val F1-Score | Checkpoint Flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | - | - | - | - | - | - | Baseline |
| **2** | - | - | - | - | - | - | `best_val_loss` |
| **3** | - | - | - | - | - | - | `best_val_acc` / `final_epoch` |

---

## 2. Quantitative System Benchmarks

| Performance Metric | Measured Value | Target Benchmark | Compliance Status |
| :--- | :--- | :--- | :---: |
| **Classification Accuracy** | [Chờ đo lường %] | $\ge 85.0\%$ | Pending |
| **Macro F1-Score** | [Chờ đo lường] | $\ge 0.85$ | Pending |
| **Training Speed** | [Giây / Epoch] | $\le 180\text{s}$ (GPU Colab T4) | Pending |
| **Inference Latency** | [ms / Sample] | $\le 25\text{ms}$ | Pending |
| **Peak GPU VRAM Footprint** | [MB / GB] | $\le 4.0\text{GB}$ | Pending |
| **Host System RAM Footprint** | [GB] | $\le 6.0\text{GB}$ | Pending |

---

## 3. Checkpoint Artifacts & Model Registry
- `final_epoch`: `checkpoints/final_epoch/`
- `best_val_loss`: `checkpoints/best_val_loss/` (Loss: ...)
- `best_val_acc`: `checkpoints/best_val_acc/` (Accuracy: ...)
- `best_val_pause`: Dừng tại Epoch ... (Early Stopping triggered: $\Delta < 1.0\%$)
- `best`: `checkpoints/best/` (Production Artifact được chọn đóng gói deploy)
