# 3.1. Tải và chuẩn bị dữ liệu FashionMNIST

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | 3.1. Tải và chuẩn bị dữ liệu FashionMNIST |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-07T11:17:41+07:00 ; HoangDang ; d41a9fe |
| Last updated | 2026-11-11T10:16:25+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Theo đặc tả chung §1.3: tạo thêm `data/reports/split_indices.json` gồm `seed=42`, `train` (54.000 chỉ số), `val` (6.000 chỉ số), dùng `torch.randperm(60000)` với generator riêng. Giữ nguyên test. Báo cáo bổ sung `sha256` bốn tệp IDX (hash byte gốc), `split_samples`, `overlap_train_val`. Mọi đầu ra lẻ có sidecar từ `record_artifact`.

Trạng thái: code và báo cáo đã có; dataset gốc được lưu qua Git LFS trong repo. Sau clone/pull dùng `git lfs pull origin hoangdang`; bước chuẩn bị vẫn hỗ trợ tải lại nếu dữ liệu chưa có.

## 1. Mục đích

Tải dữ liệu, kiểm tra cấu trúc và khảo sát ảnh cùng nhãn thật trước khi xây dựng mô hình.

## 2. File thực hiện

- File code: `data/prepare_data.py`.
- Thư viện sử dụng: `torchvision.datasets.FashionMNIST`, thư viện JSON và công cụ vẽ ảnh.
- Tất cả đường dẫn tính từ thư mục `Lab 1`. Code phải xác định thư mục gốc từ vị trí file, không dựa vào thư mục đang chạy lệnh.

## 3. Đầu vào

- Bộ dữ liệu FashionMNIST được tải bằng `torchvision.datasets.FashionMNIST`, với `root` trỏ đến `Lab 1/data` và bật tải dữ liệu khi chưa có.
- Sử dụng hai phân vùng có sẵn: train để huấn luyện và test để đánh giá sau huấn luyện; không chuyển mẫu test sang tập train.
- Cần kết nối mạng khi dữ liệu chưa được tải đầy đủ.

## 4. Các bước xử lý

1. Tạo thư mục `data/` và `data/reports/` nếu chưa có.
2. Tải hoặc mở lại hai tập train và test đã có trên máy.
3. Kiểm tra số mẫu, kích thước ảnh, số lớp và ánh xạ chỉ số sang tên lớp.
4. Chọn 16 ảnh đầu tiên từ tập train để hiển thị cùng nhãn thật.
5. Tạo split train/validation cố định seed 42; ghi chỉ số, thông tin khảo sát và hình minh họa.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Đường dẫn | Định dạng và cách lưu |
| --- | --- | --- |
| Dữ liệu train và test | `data/FashionMNIST/raw/` | Các tệp ảnh và nhãn IDX do torchvision tải và giải nén; các tệp nén `.gz` có thể được giữ lại. Không tự chuyển thành CSV hay PNG riêng lẻ. |
| Thông tin dữ liệu | `data/reports/data_summary.json` | JSON UTF-8, ghi bằng thư viện JSON sau khi kiểm tra dữ liệu. |
| Ảnh khảo sát | `data/reports/data_samples.png` | PNG, bố cục 4 × 4 ảnh xám, mỗi ảnh có tên nhãn thật. Lưu bằng chức năng xuất hình của thư viện vẽ. |
| Chỉ số split | `data/reports/split_indices.json` | JSON gồm `seed`, `train`, `val`; chỉ số thuộc tập train gốc, không trùng nhau. |

Các tệp dữ liệu đã giải nén cần có trong `data/FashionMNIST/raw/`:

- `train-images-idx3-ubyte`: ảnh train.
- `train-labels-idx1-ubyte`: nhãn train.
- `t10k-images-idx3-ubyte`: ảnh test.
- `t10k-labels-idx1-ubyte`: nhãn test.

Các trường của `data_summary.json`:

- `dataset`: chuỗi `FashionMNIST`.
- `train_samples`, `test_samples`, `num_classes`: số nguyên.
- `image_shape`: danh sách `[28, 28]`, biểu diễn chiều cao và chiều rộng ảnh gốc.
- `classes`: danh sách tên lớp theo đúng thứ tự chỉ số nhãn do dataset cung cấp.

Khi chạy lại, sử dụng dữ liệu đã tải nếu hợp lệ; tái tạo các báo cáo và sidecar (giữ lịch sử nguồn gốc). Nếu dữ liệu lỗi hoặc thiếu, báo lỗi tải/kiểm tra; không ghi báo cáo thành công.

## 6. Cách chạy và kiểm tra

Chạy từ thư mục `Lab 1`:

```powershell
python data/prepare_data.py
```

- Kiểm tra tập train có 60.000 mẫu, tập test có 10.000 mẫu và có 10 lớp.
- Mở được `data_summary.json` và `data_samples.png`; ảnh hiển thị có nhãn tương ứng.
- Chạy lại được với dữ liệu đã tải mà không cần tải lại toàn bộ.
- Bước 3.2 sử dụng dữ liệu trong `data/`; không sử dụng hình PNG làm dữ liệu huấn luyện.

Tham khảo API: [FashionMNIST — Torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).
