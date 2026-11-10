# Kiến trúc mô hình (Model Architecture)
Đươc tạo vào: 10/11/2026       
Tác giả: Xuân Bảo 
## 1. Tổng quan

Dự án sử dụng **ResNet-18** làm mô hình cơ sở để thực hiện bài toán phân loại hình ảnh trên bộ dữ liệu CIFAR-10.

ResNet-18 là mạng nơ-ron tích chập (CNN), sử dụng các kết nối phần dư (Residual Connections) nhằm hỗ trợ quá trình huấn luyện mạng sâu và cải thiện khả năng trích xuất đặc trưng hình ảnh.

Mô hình được điều chỉnh từ bài toán phân loại ImageNet (1000 lớp) sang bài toán phân loại CIFAR-10 (10 lớp).

## 2. Kiến trúc mô hình

Mô hình được xây dựng bằng `torchvision.models.resnet18`.

Các thông số chính:

- Kiến trúc: ResNet-18
- Framework: PyTorch
- Kích thước đầu vào: `(batch_size, 3, 224, 224)`
- Số lớp đầu ra: 10
- Lớp phân loại cuối cùng: `nn.Linear(model.fc.in_features, 10)`

Lớp Fully Connected ban đầu được thay thế bằng một lớp mới có 10 đầu ra để phù hợp với 10 lớp của CIFAR-10.

## 3. Transfer Learning (Học chuyển giao)

Transfer Learning sử dụng trọng số đã được huấn luyện trước trên bộ dữ liệu ImageNet.

Trong quá trình này:

- Đóng băng toàn bộ các lớp trích xuất đặc trưng bằng cách đặt `requires_grad = False`.
- Chỉ cho phép lớp phân loại cuối cùng (`model.fc`) được cập nhật trọng số.

Phương pháp này giúp mô hình tận dụng những đặc trưng hình ảnh đã học được từ ImageNet, đồng thời học cách phân loại 10 lớp của CIFAR-10.

Hàm triển khai: `freeze_backbone(model)`.

## 4. Fine-Tuning (Tinh chỉnh mô hình)

Fine-Tuning cho phép cập nhật một số lớp đã được huấn luyện trước để mô hình thích nghi tốt hơn với bộ dữ liệu mới.

Cách triển khai:

- Đóng băng toàn bộ tham số của mô hình.
- Mở khóa khối tích chập cuối cùng (`layer4`).
- Giữ lớp phân loại cuối cùng (`model.fc`) ở trạng thái có thể huấn luyện.

Phương pháp này giúp mô hình điều chỉnh các đặc trưng cấp cao cho phù hợp với CIFAR-10, đồng thời giữ lại những đặc trưng cấp thấp đã học được.

Hàm triển khai: `unfreeze_layer4(model)`.

## 5. Quản lý tham số huấn luyện

Hàm `get_trainable_parameters(model)` trả về danh sách các tham số có thuộc tính `requires_grad = True`.

Hàm này hỗ trợ xác định chính xác những tham số cần đưa vào Optimizer trong từng chế độ huấn luyện.

## 6. Kiểm thử mô hình

File `model.py` có tích hợp kiểm thử cơ bản (Smoke Test) nhằm xác nhận:

- Mô hình có thể được khởi tạo thành công.
- Transfer Learning đóng băng đúng các lớp trích xuất đặc trưng.
- Fine-Tuning mở khóa đúng `layer4` và lớp phân loại.
- Mô hình nhận đầu vào có kích thước `(2, 3, 224, 224)`.
- Mô hình tạo đầu ra có kích thước `(2, 10)`.

### Kết quả kiểm thử

| Nội dung kiểm thử | Kết quả |
|---|---|
| Khởi tạo Transfer Learning | PASS |
| Khởi tạo Fine-Tuning | PASS |
| Kích thước đầu vào | `(2, 3, 224, 224)` |
| Kích thước đầu ra | `(2, 10)` |
| Số tham số được huấn luyện trong Transfer Learning | 5.130 |
| Số tham số được huấn luyện trong Fine-Tuning | 8.398.858 |
| Smoke Test | PASS |

## 7. Ghi chú

- Mặc định, mô hình sử dụng `pretrained=True` để tải trọng số đã được huấn luyện trước trên ImageNet.
- Trong quá trình Smoke Test, sử dụng `pretrained=False` để không cần tải trọng số từ Internet.
- Hình ảnh đầu vào cần được thay đổi kích thước và chuẩn hóa theo cấu hình tiền xử lý trong `data.md`.
- Các thông tin về cấu hình huấn luyện và đánh giá mô hình được trình bày riêng trong `training.md` và `evaluation.md`.