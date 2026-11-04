# Cấu Hình Huấn Luyện (Training Specification) — Lab 2

## 1. Siêu Tham Số Huấn Luyện
- **Optimizer**: `torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)`.
- **Learning Rate Scheduler**: `torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)`.
- **Loss Function**: `nn.CrossEntropyLoss()`.
- **Batch Size**: 64 (hoặc 128).
- **Epochs**: $10 \sim 15$ epochs.

## 2. Quy Trình Chạy Kiểm Thử
- **Smoke test**: Chạy thử 1 batch với kích thước ảnh $(64, 3, 224, 224)$ để kiểm tra không bị CUDA OOM.
- **Normal run**: Huấn luyện toàn bộ dữ liệu.
- **Lưu Checkpoints & Early Stopping**:
  - Tự động lưu `best_val_loss.pth` và `best_val_acc.pth`.
  - Dừng nếu `val_accuracy` cải thiện dưới 0% - 2% trong 3 epochs liên tiếp.

