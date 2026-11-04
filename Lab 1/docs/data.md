# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 1

## 1. Nguồn Gốc Dataset
- **Tên dataset**: FashionMNIST (hoặc dataset được giao trong đề Lab 1).
- **Định dạng dữ liệu**: Ảnh xám $28 \times 28$ pixels, 10 lớp trang phục.
- **Phân chia tập dữ liệu**:
  - `Train`: 60,000 mẫu
  - `Test`: 10,000 mẫu

## 2. Tiền Xử Lý (Transforms)
- Chuyển ảnh thành Tensor: `transforms.ToTensor()`.
- Chuẩn hóa phân phối điểm ảnh: `transforms.Normalize((0.5,), (0.5,))`.
- Cấu hình DataLoader: `batch_size = 64`, `shuffle = True`.

