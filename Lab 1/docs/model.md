# Kiến Trúc Mô Hình (Model Specification) — Lab 1

## 1. Kiến Trúc Mạng
- **Loại mô hình**: Multi-Layer Perceptron (MLP) hoặc CNN cơ bản kế thừa từ `torch.nn.Module`.
- **Cấu trúc tầng**:
  - `Input`: Làm phẳng ảnh $28 \times 28 \rightarrow 784$ chiều.
  - `Hidden Layer 1`: `Linear(784, 512)` + `ReLU()`.
  - `Hidden Layer 2`: `Linear(512, 256)` + `ReLU()`.
  - `Output`: `Linear(256, 10)` (10 classes).

## 2. Số Lượng Tham Số
- Ước tính tham số: $\approx 530,000$ tham số.
- Thiết bị chạy: CPU hoặc GPU (`cuda`).

