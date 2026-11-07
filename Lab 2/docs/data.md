# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 2

Mã nguồn: `src/lab_2/data.py`. Tham số: mục `data` trong `configs/lab_2.yaml`.

## 1. Nguồn Gốc Dataset
- **Tên dataset**: CIFAR-10, tải qua `torchvision.datasets.CIFAR10`.
- **Nhiệm vụ**: Phân loại ảnh vào 10 lớp: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
- **Quy mô gốc**: 60.000 ảnh màu $32 \times 32 \times 3$, gồm 50.000 ảnh train và 10.000 ảnh test. Dataset không có sẵn tập validation.
- **Tải dữ liệu**: Server gốc thường chậm trên Colab nên notebook tải trước file `cifar-10-python.tar.gz` từ bản sao trên Hugging Face. `torchvision` vẫn kiểm tra MD5 của file; nếu không khớp sẽ tải lại từ server gốc.

## 2. Phân Chia Dữ Liệu

| Tập | Số ảnh | Lấy từ | Vai trò |
| :--- | :---: | :--- | :--- |
| `train` | 22.500 | Tách từ `train` gốc | Tính gradient và cập nhật trọng số |
| `val` | 2.500 | Tách từ `train` gốc, ngay sau tập `train` | Theo dõi loss, accuracy; so sánh các lần chạy; chọn checkpoint; dừng sớm |
| `test` | 5.000 | Tách từ `test` gốc | Chỉ đánh giá một lần trên mốc `best` |

- **Lý do chọn số ảnh**: Đề bài không quy định số mẫu. Nhóm dùng một nửa dữ liệu để giảm thời gian dùng GPU.
- **Tái lập**: Xáo trộn với `seed = 42` trước khi lấy mẫu.
- **Chống trùng mẫu**: `train` và `val` là hai đoạn không giao nhau sau khi xáo trộn. `test` lấy từ tập gốc khác.
- **Cân bằng lớp**: Không ép cân bằng, chỉ báo cáo số ảnh mỗi lớp trong từng tập vào mục `data` của `reports/metrics.json`.
- **Xử lý lỗi**: Nếu số ảnh yêu cầu lớn hơn số ảnh của dataset thì báo `ValueError`.

## 3. Tiền Xử Lý

Các bước dùng `torchvision.transforms` và chạy theo đúng thứ tự dưới đây. Augmentation phải chạy trước `Resize` vì `RandomCrop` làm việc trên ảnh $32 \times 32$.

| Thứ tự | Bước | `train` | `val`, `test`, demo |
| :---: | :--- | :---: | :---: |
| 1 | `RandomHorizontalFlip()` | Có | Không |
| 2 | `RandomCrop(32, padding=4)` | Có | Không |
| 3 | `Resize((224, 224))` | Có | Có |
| 4 | `ToTensor()` | Có | Có |
| 5 | `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` | Có | Có |

- **Lý do `Resize` và `Normalize`**: Backbone được huấn luyện trên ảnh ImageNet $224 \times 224$ với các giá trị chuẩn hóa này, nên đầu vào cần có cùng dạng.
- **Lý do tách augmentation**: Tập `val` và `test` phải ổn định giữa các lần đánh giá.

## 4. DataLoader

| Tham số | Train | Validation, test |
| :--- | :---: | :---: |
| `batch_size` | Theo từng lần chạy (64 hoặc 128) | 128 |
| `shuffle` | `True`, với seed cố định | `False` |
| `num_workers` | 2 | 2 |
| `pin_memory` | Chỉ bật khi có GPU | Chỉ bật khi có GPU |
