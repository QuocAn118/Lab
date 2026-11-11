# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Dữ Liệu Thực Nghiệm (Data Specification) — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Nguồn Gốc Dataset
- **Tên dataset**: FashionMNIST.
- **Định dạng dữ liệu**: Ảnh xám $28 \times 28$ pixels, 10 lớp trang phục.
- **Phân chia tập dữ liệu**:
  - Train gốc: 60.000 mẫu; chia thành 54.000 train và 6.000 validation bằng seed 42.
  - `Test`: 10,000 mẫu

## 2. Tiền Xử Lý (Transforms)
- Chuyển ảnh thành Tensor: `transforms.ToTensor()`.
- Cơ sở chỉ dùng `ToTensor()`, không Normalize; ảnh float32 [0,1].
- Cấu hình DataLoader: `batch_size = 64`; chỉ train `shuffle=True`, validation/test `shuffle=False`.

## 3. Triển khai 3.1 và 3.2

- `python data/prepare_data.py`: tải/kiểm tra dữ liệu, lưu `data/reports/data_summary.json`, `data_samples.png` (16 ảnh đầu), `split_indices.json`. SHA-256 tính trên byte gốc bốn tệp IDX.
- Chia train gốc thành 54.000 train / 6.000 validation bằng hoán vị seed 42; test 10.000 độc lập.
- `python data/data_pipeline.py --check`: mở dữ liệu không download; lưu `data/reports/batch_check.json`.
- `create_dataloaders(config)` trả về train, validation, test theo cấu hình `train.py`; chỉ train shuffle, `drop_last=False`, `num_workers=0`. `data.limits` dùng cho smoke run sau split.
- Batch ảnh `float32 [B,1,28,28]`, nhãn `int64 [B]`. Mọi đầu ra lẻ được lưu nguồn gốc `.prov.json`. Đường dẫn tính từ file script, độc lập thư mục chạy.

## 4. Bằng chứng và giới hạn

- Báo cáo đã lưu: [data_summary.json](../data/reports/data_summary.json), [split_indices.json](../data/reports/split_indices.json), [batch_check.json](../data/reports/batch_check.json), [data_samples.png](../data/reports/data_samples.png).
- Đây là bằng chứng của lần chuẩn bị dữ liệu đã lưu, không phải kết quả huấn luyện. Dataset gốc không lưu trên Git; phải chạy bước 3.1 trên máy chưa có dữ liệu trước khi kiểm thử tích hợp.
- `data/tests/test_data.py` kiểm tra số mẫu, split không trùng, thứ tự validation/test, batch và giới hạn smoke. Trạng thái kiểm tra hiện tại xem [tiến độ](plan_progress_status.md).

