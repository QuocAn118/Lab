# Cấu Hình Huấn Luyện (Training Specification) — Lab 3

## 1. Siêu Tham Số Huấn Luyện (Hyperparameters)
- **Framework**: Hugging Face `Trainer` kết hợp PyTorch `torch`.
- **Optimizer**: `AdamW` (Decoupled Weight Decay).
- **Learning Rate ($\eta$)**: $2 \times 10^{-5}$ (chuẩn mực cho fine-tuning Transformer).
- **Batch Size**: 16 (hoặc 32 tùy theo dung lượng VRAM).
- **Số lượng Epochs**: $3 \sim 5$ epochs.
- **Weight Decay**: $0.01$.
- **Hàm mất mát (Loss Function)**: Cross-Entropy Loss.

## 2. Quy Trình Chạy Kiểm Thử (Smoke Test $\rightarrow$ Normal Run)
1. **Smoke Test**:
   - Chạy thử mồi với 1 batch nhỏ ($16$ mẫu) trong 1 epoch.
   - Xác nhận: Dữ liệu nạp đúng, forward pass không lỗi dimension, backward pass tính toán gradient mượt mà, không gặp lỗi CUDA out of memory.
2. **Normal Run**:
   - Huấn luyện trên toàn bộ tập dữ liệu đã chia.

## 3. Cấu Hình Lưu Checkpoints & Early Stopping
- `evaluation_strategy = "epoch"`.
- `save_strategy = "epoch"`.
- `load_best_model_at_end = True`.
- `metric_for_best_model = "accuracy"` (hoặc `"eval_loss"`).
- **Early Stopping Callback (Ngắt tạm dừng khi cải thiện $0 - 2\%$)**:
  - `patience = 2` epochs.
  - `threshold = 0.01` ($1\%$). Nếu trong 2 epochs liên tiếp mà `val_accuracy` tăng trưởng không quá $1\%$, tiến hành dừng huấn luyện để lưu `best_val_acc`.

