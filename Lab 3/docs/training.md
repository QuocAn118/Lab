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

## 4. Cấu hình pipeline CLI đầy đủ

Batch 16, tối đa 3 epochs, LR 2e-5, AdamW, decay .01, gradient clip 1.0, seed 42; CUDA dùng FP16, CPU dùng FP32. Smoke chạy 2 optimizer steps trên 32 mẫu; khởi tạo model mới cho full, không dùng trọng số smoke.

**Điều chỉnh kỹ thuật so với mục 3:** callback tự lưu 5 checkpoint suy luận, `save_strategy="no"` cho Trainer để tránh lưu lặp optimizer nhiều GB. Sau train, lưu `final_epoch` trước rồi nạp `best_val_acc` từ đĩa thành `best`; đạt hành vi chọn model tốt nhất nhưng không bật `load_best_model_at_end` của Trainer. Không hỗ trợ resume optimizer. Callback có kiểm thử riêng.

Early stopping dùng ngưỡng accuracy tuyệt đối > .01 (1 điểm phần trăm), patience 2; cập nhật best kể cả cải thiện nhỏ. `best_val_pause` là lần validation cuối, có cờ cho biết thực sự dừng sớm hay không. Chi tiết [implementation.md](implementation.md).

## 5. Notebook rút gọn

Notebook dùng `Trainer` và `EarlyStoppingCallback` trực tiếp. Cấu hình huấn luyện chính giữ nguyên; Trainer lưu tối đa 2 checkpoint theo epoch, nạp model có validation accuracy tốt nhất và lưu bản suy luận tại `outputs/simple-training/<run>/best`. Không dùng callback tùy biến của pipeline CLI. Nếu đã có model full, notebook mặc định nạp model để đánh giá; bật `TRAIN=True` để huấn luyện mới. Chi tiết tại [implementation.md §6](implementation.md).

## Nguồn và phạm vi

- Mục 1–3: đặc tả có sẵn; mục 4 là quyết định triển khai và ghi rõ khác biệt.
- [sources.md S-01–S-02](sources.md): Trainer/callback API.
- [AI_Usage.md](AI_Usage.md), [implementation.md](implementation.md): khai nguồn và hợp đồng kiểm thử.

