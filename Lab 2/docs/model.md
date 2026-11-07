# Kiến Trúc Mô Hình (Model Specification) — Lab 2

Mã nguồn: `src/lab_2/model.py`. Tham số: mục `model` trong `configs/lab_2.yaml`.

## 1. Model Sử Dụng

| | ResNet18 | DenseNet121 |
| :--- | :--- | :--- |
| **Vai trò** | Mô hình gốc, Exercise 2 | Exercise 1 |
| **Cách tải** | `torchvision.models.resnet18(weights="DEFAULT")` | `torchvision.models.densenet121(weights="DEFAULT")` |
| **Tầng cuối được thay** | `model.fc` | `model.classifier` |

- **Trọng số**: `weights="DEFAULT"` tải trọng số đã huấn luyện trên ImageNet. Đây là cách viết hiện tại của `pretrained=True` trong đề bài.
- **Lý do không dùng VGG**: Đề bài ghi "VGG, DenseNet, etc." nên không bắt buộc đủ. File trọng số của VGG16 nặng khoảng 530 MB, trong khi file của DenseNet121 chỉ 28,5 MB.

## 2. Xem Kiến Trúc

`python -m lab_2.model` thực hiện bước 3 của đề bài:
- In cấu trúc model bằng `print(model)`.
- In tên, kích thước và trạng thái (`train` hoặc `frozen`) của từng tham số.
- Ghi tổng số tham số và số tham số huấn luyện được vào mục `model` của `reports/metrics.json`.

## 3. Lớp Phân Loại

Tầng cuối 1.000 lớp của ImageNet được thay bằng khối sau:

| Thứ tự | Lớp | ResNet18 | DenseNet121 |
| :---: | :--- | :---: | :---: |
| 1 | `Dropout` | $p = 0.2$ | $p = 0.2$ |
| 2 | `Linear` | $512 \to 10$ | $1024 \to 10$ |

- **Đầu ra của model**: Logits của 10 lớp. Hàm mất mát Cross-Entropy nhận trực tiếp logits.
- **Softmax**: Chỉ dùng trong `demo.py` để đổi logits thành xác suất hiển thị.

## 4. Đóng Băng & Fine-Tuning

- **Mặc định (transfer learning)**: Đóng băng toàn bộ tầng trích xuất đặc trưng bằng `requires_grad = False`, chỉ huấn luyện lớp phân loại. Cả 4 lần chạy đều dùng cách này.
- **Fine-tuning**: `build_model(..., unfreeze=["layer4"])` mở khóa khối `layer4` của ResNet18. Để giảm thời gian dùng GPU, phần này chỉ được kiểm tra bằng smoke test trong `training.md` mục 4, không có lần chạy đầy đủ nên không có số accuracy.
- **BatchNorm**: Khi huấn luyện, chỉ phần được mở khóa ở chế độ `train`. BatchNorm của tầng đóng băng giữ chế độ `eval` để thống kê từ ImageNet không bị thay đổi.

## 5. Tham Số

Đo bằng cách đếm tham số sau khi thay lớp phân loại.

| Model | Tổng tham số | Huấn luyện được | Dung lượng 1 checkpoint |
| :--- | :---: | :---: | :---: |
| ResNet18, đóng băng | 11.181.642 | 5.130 | 44,8 MB |
| DenseNet121, đóng băng | 6.964.106 | 10.250 | 28,5 MB |
| ResNet18, mở khóa `layer4` | 11.181.642 | 8.398.858 | 44,8 MB |

## 6. Kỹ Thuật Có Trong Model

- **BatchNorm**: Có sẵn trong backbone của ResNet18 và DenseNet121, nhóm không tự thêm.
- **Dropout**: Có ở lớp phân loại, xem mục 3.
- **Weight decay**: Thuộc optimizer, xem `training.md` mục 1.
