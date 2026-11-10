# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 1

## 1. Nguồn Gốc Dataset
- **Tên dataset**: FashionMNIST (hoặc dataset được giao trong đề Lab 1).
- **Định dạng dữ liệu**: Ảnh xám $28 \times 28$ pixels, 10 lớp trang phục.
- **Phân chia tập dữ liệu**:
  - `Train`: 60,000 mẫu
  - `Test`: 10,000 mẫu

## 2. Tiền Xử Lý (Transforms)
- Chuyển ảnh thành Tensor: `transforms.ToTensor()`.
- Cơ sở chỉ dùng `ToTensor()`, không Normalize; ảnh float32 [0,1].
- Cấu hình DataLoader: `batch_size = 64`, `shuffle = True`.

## 3. Triển khai 3.1 và 3.2

Last updated: 2026-11-10; Updated by: Codex (AI hỗ trợ, chờ nhóm review).

- `python data/prepare_data.py`: tải/kiểm tra dữ liệu, lưu `data/reports/data_summary.json`, `data_samples.png` (16 ảnh đầu), `split_indices.json`. SHA-256 tính trên byte gốc bốn tệp IDX.
- Chia train gốc thành 54.000 train / 6.000 validation bằng hoán vị seed 42; test 10.000 độc lập.
- `python data/data_pipeline.py --check`: mở dữ liệu không download; lưu `data/reports/batch_check.json`.
- `create_dataloaders(config)` trả về train, validation, test theo cấu hình `train.py`; chỉ train shuffle, `drop_last=False`, `num_workers=0`. `data.limits` dùng cho smoke run sau split.
- Batch ảnh `float32 [B,1,28,28]`, nhãn `int64 [B]`. Mọi đầu ra lẻ được lưu nguồn gốc `.prov.json`. Đường dẫn tính từ file script, độc lập thư mục chạy.

