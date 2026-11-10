# 3.1. Tải và chuẩn bị dữ liệu FashionMNIST

## 1. Mục đích

Tải dữ liệu, kiểm tra cấu trúc và khảo sát ảnh cùng nhãn thật trước khi xây dựng mô hình.

## 2. File thực hiện

- File code dự kiến: `src/prepare_data.py`.
- Thư viện sử dụng: `torchvision.datasets.FashionMNIST`, thư viện JSON và công cụ vẽ ảnh.
- Tất cả đường dẫn tính từ thư mục `Lab 1`. Code phải xác định thư mục gốc từ vị trí file, không dựa vào thư mục đang chạy lệnh.

## 3. Đầu vào

- Bộ dữ liệu FashionMNIST được tải bằng `torchvision.datasets.FashionMNIST`, với `root` trỏ đến `Lab 1/data` và bật tải dữ liệu khi chưa có.
- Sử dụng hai phân vùng có sẵn: train để huấn luyện và test để đánh giá sau huấn luyện; không chuyển mẫu test sang tập train.
- Cần kết nối mạng khi dữ liệu chưa được tải đầy đủ.

## 4. Các bước xử lý

1. Tạo thư mục `data/` và `outputs/` nếu chưa có.
2. Tải hoặc mở lại hai tập train và test đã có trên máy.
3. Kiểm tra số mẫu, kích thước ảnh, số lớp và ánh xạ chỉ số sang tên lớp.
4. Chọn 16 ảnh đầu tiên từ tập train để hiển thị cùng nhãn thật.
5. Ghi thông tin khảo sát và lưu hình minh họa.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Đường dẫn | Định dạng và cách lưu |
| --- | --- | --- |
| Dữ liệu train và test | `data/FashionMNIST/raw/` | Các tệp ảnh và nhãn IDX do torchvision tải và giải nén; các tệp nén `.gz` có thể được giữ lại. Không tự chuyển thành CSV hay PNG riêng lẻ. |
| Thông tin dữ liệu | `outputs/data_summary.json` | JSON UTF-8, ghi bằng thư viện JSON sau khi kiểm tra dữ liệu. |
| Ảnh khảo sát | `outputs/data_samples.png` | PNG, bố cục 4 × 4 ảnh xám, mỗi ảnh có tên nhãn thật. Lưu bằng chức năng xuất hình của thư viện vẽ. |

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

Khi chạy lại, sử dụng dữ liệu đã tải nếu hợp lệ; thay thế hai tệp khảo sát bằng kết quả mới của cùng bộ dữ liệu. Nếu dữ liệu lỗi hoặc thiếu, báo lỗi tải/kiểm tra; không ghi báo cáo thành công.

## 6. Cách chạy và kiểm tra

Lệnh dự kiến, chạy từ thư mục `Lab 1` sau khi code được triển khai:

```powershell
python src/prepare_data.py
```

- Kiểm tra tập train có 60.000 mẫu, tập test có 10.000 mẫu và có 10 lớp.
- Mở được `data_summary.json` và `data_samples.png`; ảnh hiển thị có nhãn tương ứng.
- Chạy lại được với dữ liệu đã tải mà không cần tải lại toàn bộ.
- Bước 3.2 sử dụng dữ liệu trong `data/`; không sử dụng hình PNG làm dữ liệu huấn luyện.

Tham khảo API: [FashionMNIST — Torchvision](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).
