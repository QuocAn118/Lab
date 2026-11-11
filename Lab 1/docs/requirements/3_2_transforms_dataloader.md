# 3.2. Áp dụng transforms và tạo DataLoader

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | 3.2. Áp dụng transforms và tạo DataLoader |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-07T11:17:41+07:00 ; HoangDang ; d41a9fe |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Trạng thái: code và báo cáo batch đã có; kiểm thử dữ liệu thật yêu cầu chuẩn bị dataset gốc trên máy.

## 1. Mục đích

Chuyển ảnh thành tensor và tổ chức các mẫu thành batch để cung cấp cho huấn luyện và đánh giá.

## 2. File thực hiện

- File code: `data/data_pipeline.py`.
- Hàm `create_dataloaders(config)` trả về `(train_loader, val_loader, test_loader)` theo giao diện chung.
- Sử dụng `torchvision.datasets.FashionMNIST`, `ToTensor()` và `torch.utils.data.DataLoader`.
- Hàm được gọi bởi `src/train.py` và `src/evaluate.py`.

## 3. Đầu vào

- Dữ liệu do bước 3.1 lưu trong `data/FashionMNIST/raw/`.
- Cấu hình trong bộ nhớ gồm `batch_size`, `seed` và `transforms`.
- Khi huấn luyện, cấu hình do `train.py` chuẩn bị; khi đánh giá, cấu hình đọc từ `outputs/<run_id>/config.json`.
- Cấu hình cơ sở dự kiến: `batch_size = 64`, `seed = 42`, `transforms = ["ToTensor"]`. Đây là lựa chọn triển khai, không phải giá trị bắt buộc trong đề.

## 4. Các bước xử lý

1. Mở hai tập dữ liệu đã được chuẩn bị; không tự tải thêm dữ liệu trong bước này. Nếu thiếu dữ liệu, yêu cầu chạy bước 3.1.
2. Áp dụng `ToTensor()` cho train, validation và test: ảnh xám được chuyển thành tensor có dạng `[1, 28, 28]`, kiểu `float32`, giá trị trong khoảng `[0, 1]`.
3. Tạo DataLoader train với `shuffle=True`; thiết lập bộ sinh số ngẫu nhiên theo `seed`.
4. Tạo DataLoader validation và test với `shuffle=False` để giữ thứ tự mẫu.
5. Dùng `drop_last=False` để giữ cả batch cuối; dùng `num_workers=0` cho cấu hình cơ sở trên Windows.
6. Trả về ba DataLoader cho bên gọi; khi chạy kiểm tra riêng, lấy một batch và ghi thông tin kích thước.

### Phân vùng và cấu hình

Theo đặc tả chung §1.3: dùng `data/reports/split_indices.json` của bước 3.1 để chia 54.000 train / 6.000 validation (seed 42); giữ 10.000 test. Validation không shuffle. Chấp nhận cấu hình phẳng hoặc cấu hình lồng `data`/`training` của `train.py`; áp dụng `data.limits` sau split cho smoke run. Split cố định seed 42, shuffle theo seed cấu hình. Từ chối transforms ngoài `ToTensor` và limits không hợp lệ.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Nơi sử dụng hoặc lưu | Định dạng |
| --- | --- | --- |
| `train_loader`, `val_loader`, `test_loader` | Bộ nhớ của tiến trình gọi hàm | Đối tượng DataLoader; không lưu DataLoader hay toàn bộ tensor batch thành tệp. |
| Batch ảnh | Trả về khi duyệt DataLoader | Tensor `float32`, dạng `[B, 1, 28, 28]`. |
| Batch nhãn | Trả về cùng batch ảnh | Tensor `int64`, dạng `[B]`, giá trị từ 0 đến 9. |
| Báo cáo kiểm tra batch | `data/reports/batch_check.json` | JSON UTF-8, chỉ lưu khi chạy module ở chế độ kiểm tra riêng. |

`B` là số mẫu thực tế trong batch, có thể nhỏ hơn `batch_size` ở batch cuối.

`batch_check.json` gồm `batch_size_requested`, `image_shape`, `image_dtype`, `label_shape`, `label_dtype`, `pixel_min`, `pixel_max` của batch train được kiểm tra. Chạy kiểm tra lại sẽ ghi đè báo cáo này.

Quy trình cơ sở chỉ dùng `ToTensor()`. Nếu bổ sung chuẩn hóa hoặc transforms khác ở thí nghiệm sau, phải cập nhật cấu hình và dùng đúng transforms đó khi tải mô hình để dự đoán.

## 6. Cách chạy và kiểm tra

Chạy từ thư mục `Lab 1` sau bước 3.1:

```powershell
python data/data_pipeline.py --check
```

- Với batch đầu của cấu hình cơ sở, ảnh có dạng `[64, 1, 28, 28]`, nhãn có dạng `[64]`.
- Kiểm tra kiểu dữ liệu, khoảng giá trị ảnh và chỉ số nhãn đúng như mô tả.
- Tổng số mẫu khi duyệt hết mỗi DataLoader bằng số mẫu của tập tương ứng.
- Duyệt test_loader giữ đúng thứ tự mẫu để bước 3.5 gắn dự đoán với ảnh gốc.

Tham khảo API: [ToTensor — Torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html).
