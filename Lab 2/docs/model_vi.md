# Kiến trúc mô hình (Model Architecture)
Đươc tạo vào: 10/11/2026     
Tác giả: Xuân Bảo    
Được cập nhật vào: 10/11/2026   
Tác giả: Xuân Bảo 

## 1. Tổng quan

Module Model chịu trách nhiệm triển khai các kiến trúc học sâu cho bài toán phân loại ảnh CIFAR-10, bao gồm 10 lớp.

Phần triển khai sử dụng PyTorch và Torchvision, hỗ trợ hai kiến trúc đã được huấn luyện trước:

- ResNet18
- DenseNet121

Cả hai kiến trúc đều sử dụng trọng số pretrained từ ImageNet thông qua Transfer Learning. Feature extractor mặc định được đóng băng, trong khi classification head ban đầu được thay thế bằng một head mới phù hợp với bài toán phân loại 10 lớp.

**File triển khai:** `Lab 2/model.py`

## 2. Mục tiêu

Các mục tiêu của module bao gồm:

- Triển khai và quản lý nhiều kiến trúc mô hình.
- Tái sử dụng trọng số pretrained từ ImageNet.
- Áp dụng Transfer Learning bằng cách đóng băng feature extractor.
- Hỗ trợ Fine-tuning thông qua việc mở khóa có chọn lọc các module của mô hình.
- Thay thế classification head ban đầu bằng các lớp Dropout và Linear.
- Tính toán tổng số tham số và số tham số có thể huấn luyện.
- Quản lý chế độ training và evaluation, đặc biệt là hoạt động của BatchNorm.
- Lưu và khôi phục checkpoint của mô hình.
- Xuất thông tin mô hình vào file `metrics.json`.

## 3. Kiến trúc mô hình

### 3.1. Các mô hình được hỗ trợ

| Model | Kiến trúc | Classification Head |
|---|---|---|
| ResNet18 | Mạng Residual | `fc` |
| DenseNet121 | Mạng kết nối dày đặc | `classifier` |

Module định nghĩa các hằng số sau:

- `NUM_CLASSES = 10`: số lượng lớp cần phân loại.
- `HEADS`: ánh xạ mỗi mô hình được hỗ trợ với classification head tương ứng.
- `SUPPORTED_MODELS`: chứa danh sách tên các mô hình được hỗ trợ.

### 3.2. Classification Head

Classification head ban đầu của ImageNet được thay thế bằng:

```python
nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, NUM_CLASSES)
)
```

Classification head mới bao gồm:

- Dropout với xác suất 0.2 nhằm giảm hiện tượng overfitting.
- Lớp Linear ánh xạ đặc trưng được trích xuất thành 10 giá trị đầu ra.

## 4. Transfer Learning và Fine-tuning

### 4.1. Transfer Learning

Theo mặc định, toàn bộ feature extractor được đóng băng bằng:

```python
model.requires_grad_(False)
```

Sau đó, classification head mới được gắn vào mô hình và giữ trạng thái có thể huấn luyện.

Trong quá trình Transfer Learning, chỉ các tham số của classification head được cập nhật.

### 4.2. Fine-tuning

Hàm `build_model()` hỗ trợ mở khóa có chọn lọc thông qua tham số `unfreeze`.

Ví dụ:

```python
model = build_model(
    name="resnet18",
    dropout=0.2,
    unfreeze=["layer4"],
    pretrained=True
)
```

Trong cấu hình này:

- Classification head có thể được huấn luyện.
- Module `layer4` được mở khóa.
- Tất cả các module còn lại của feature extractor tiếp tục bị đóng băng.

## 5. Các hàm chính

| Hàm | Mô tả |
|---|---|
| `head_of(model)` | Trả về classification head của mô hình. |
| `build_model()` | Xây dựng mô hình và cấu hình Transfer Learning hoặc Fine-tuning. |
| `trainable_parameters()` | Trả về danh sách các tham số có thể huấn luyện. |
| `count_parameters()` | Tính tổng số tham số và số tham số có thể huấn luyện. |
| `set_train_mode()` | Cấu hình chế độ training cho các module có tham số trainable. |
| `save_checkpoint()` | Lưu trọng số mô hình và metadata huấn luyện. |
| `load_checkpoint()` | Khôi phục mô hình từ checkpoint. |
| `main()` | Xây dựng các cấu hình được hỗ trợ và xuất báo cáo mô hình. |

### 5.1. Quản lý BatchNorm

Hàm `set_train_mode()` trước tiên đưa toàn bộ mô hình về chế độ evaluation.

Sau đó, hàm bật chế độ training cho classification head và các module trực tiếp sở hữu tham số có thể huấn luyện.

Điều này giúp các lớp BatchNorm thuộc phần bị đóng băng không cập nhật running statistics trong quá trình huấn luyện.

### 5.2. Quản lý Checkpoint

Hàm `save_checkpoint()` lưu các thông tin:

- `state_dict`: các tham số của mô hình.
- `name`: tên kiến trúc mô hình.
- `dropout`: xác suất Dropout.
- `unfreeze`: danh sách các module được mở khóa.
- `epoch`: epoch tại thời điểm lưu checkpoint.
- `val_accuracy`: độ chính xác trên tập validation.
- `val_loss`: validation loss.

Checkpoint được ghi vào một file tạm trước, sau đó chuyển đến vị trí đích bằng `os.replace()` nhằm giảm nguy cơ tạo ra file không hoàn chỉnh.

Hàm `load_checkpoint()` tái tạo mô hình với `pretrained=False`, nạp state dictionary đã lưu, chuyển mô hình đến thiết bị được yêu cầu và trả về mô hình đã khôi phục cùng metadata.

## 6. Kết quả thống kê tham số

Kết quả chạy thực tế `model.py`:

| Model | Tổng số tham số | Số tham số có thể huấn luyện |
|---|---:|---:|
| ResNet18 | 11,181,642 | 5,130 |
| DenseNet121 | 6,964,106 | 10,250 |
| ResNet18 + layer4 | 11,181,642 | 8,398,858 |

Nhận xét:

- ResNet18 chỉ huấn luyện classification head trong cấu hình mặc định.
- DenseNet121 có số tham số trainable ở classification head lớn hơn ResNet18.
- Việc mở khóa `layer4` làm tăng đáng kể số tham số có thể huấn luyện của ResNet18.
- Tổng số tham số không thay đổi khi chuyển từ Transfer Learning sang Fine-tuning.

Cả ba kết quả thống kê tham số đều khớp với các giá trị kỳ vọng trong đặc tả Model.

## 7. Báo cáo mô hình

Sau khi thực thi `main()`, thông tin mô hình được lưu tại:

```text
Lab 2/
└── reports/
    └── metrics.json
```

Báo cáo bao gồm:

- Tên kiến trúc mô hình.
- Xác suất Dropout.
- Các module được mở khóa.
- Tổng số tham số.
- Số tham số có thể huấn luyện.

Thông tin mô hình được lưu dưới khóa `model`. Các phần báo cáo hiện có được giữ nguyên khi file JSON được cập nhật.

## 8. Cách chạy

Chạy lệnh sau từ thư mục gốc của repository:

```powershell
python ".\Lab 2\model.py"
```

Để kiểm tra cú pháp Python:

```powershell
python -m py_compile ".\Lab 2\model.py"
```

Trong lần chạy đầu tiên với `pretrained=True`, Torchvision có thể tải trọng số pretrained từ ImageNet.

## 9. Kết quả kiểm chứng

Bảng sau tổng hợp các kết quả kiểm chứng đã thực hiện:

| Nội dung kiểm chứng | Trạng thái | Minh chứng |
|---|---|---|
| Kiểm tra cú pháp Python | PASS | `py_compile` hoàn thành không có lỗi. |
| Khởi tạo ResNet18 | PASS | Số lượng tham số đúng kỳ vọng. |
| Khởi tạo DenseNet121 | PASS | Số lượng tham số đúng kỳ vọng. |
| Fine-tuning ResNet18 layer4 | PASS | Số lượng tham số đúng kỳ vọng. |
| Tạo báo cáo mô hình | PASS | File `metrics.json` được tạo thành công. |
| Unit tests | PASS | Pytest: 14/14 tests passed, 0 failed trong 7.48 giây. |

## 10. Kết luận

Module Model triển khai hai kiến trúc ResNet18 và DenseNet121 với trọng số pretrained từ ImageNet, đồng thời hỗ trợ cả Transfer Learning và Fine-tuning.

Module cũng cung cấp các chức năng thống kê tham số, quản lý chế độ training có xét đến BatchNorm và lưu/khôi phục checkpoint.

Kết quả chạy thực tế xác nhận cả ba cấu hình mô hình đều tạo ra số lượng tham số đúng kỳ vọng và báo cáo mô hình được tạo thành công.

Cần thực hiện đầy đủ unit tests trước khi kết luận module đã được kiểm chứng hoàn toàn.