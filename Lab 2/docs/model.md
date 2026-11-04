# Kiến Trúc Mô Hình (Model Specification) — Lab 2

## 1. Mô Hình Lựa Chọn
- **Kiến trúc Pre-trained**: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` (hoặc VGG-16, DenseNet-121).
- **Chiến lược Transfer Learning**:
  - Đóng băng (Freeze) toàn bộ các tầng trích xuất đặc trưng (`requires_grad = False`).
  - Thay thế tầng Fully Connected cuối cùng: `model.fc = nn.Linear(model.fc.in_features, 10)`.

## 2. Thử Nghiệm Tinh Chỉnh (Fine-Tuning)
- Mở khóa (Unfreeze) khối `layer4` của ResNet để huấn luyện với Learning Rate nhỏ ($10^{-5}$) nhằm tối ưu độ chính xác.

