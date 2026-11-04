# Cấu Hình Huấn Luyện (Training Specification) — Lab 1

## 1. Siêu Tham Số Huấn Luyện
- **Optimizer**: `torch.optim.SGD` (với Momentum = 0.9) hoặc `torch.optim.Adam`.
- **Learning Rate**: $1 \times 10^{-3}$.
- **Loss Function**: `nn.CrossEntropyLoss()`.
- **Epochs**: $10 \sim 20$ epochs.

## 2. Quy Trình Chạy Kiểm Thử
- **Smoke test**: Chạy thử 1 batch dữ liệu để đảm bảo không lỗi tensor shape.
- **Normal run**: Huấn luyện toàn bộ epochs.
- **Cơ chế lưu Checkpoint**: Tự động lưu `best_val_loss.pth` và `best_val_acc.pth`.
- **Early Stopping**: Dừng nếu độ chính xác không cải thiện quá 0% - 2% sau 3 epochs.

