# Cấu Hình Huấn Luyện (Training Specification) — Lab 2

Mã nguồn: `src/lab_2/training.py`. Tham số: `configs/lab_2.yaml`.

## 1. Siêu Tham Số Chung

| Tham số | Giá trị | Ghi chú |
| :--- | :---: | :--- |
| Hàm mất mát | `nn.CrossEntropyLoss()` | |
| Optimizer | `torch.optim.AdamW` | Chỉ nhận các tham số huấn luyện được |
| Weight decay | $10^{-4}$ | |
| LR scheduler | `ReduceLROnPlateau(mode="min", patience=2)` | Theo dõi validation loss |
| Số epoch tối đa | 6 | Có thể dừng sớm từ epoch 4 |
| Batch size khi đánh giá | 128 | |
| Seed | 42 | |
| Ghi train loss | Mỗi 50 batch | |

## 2. Các Lần Chạy

Mỗi lần chạy chỉ đổi một yếu tố so với mô hình gốc, nên có thể biết kết quả thay đổi do yếu tố nào.

| Lần chạy | Mục đích | Model | Learning rate | Batch size |
| :--- | :--- | :---: | :---: | :---: |
| `resnet18_baseline` | Mô hình gốc | ResNet18 | $10^{-3}$ | 64 |
| `densenet121` | Exercise 1, đổi model | DenseNet121 | $10^{-3}$ | 64 |
| `resnet18_lr1e-2` | Exercise 2, đổi learning rate | ResNet18 | $10^{-2}$ | 64 |
| `resnet18_bs128` | Exercise 2, đổi batch size | ResNet18 | $10^{-3}$ | 128 |

## 3. Quy Trình Huấn Luyện

Mỗi lần chạy là một cell trong notebook: `python -m lab_2.training --run <tên lần chạy>`. Trong mỗi epoch:

1. Đưa phần được mở khóa về chế độ `train`.
2. Với từng batch: xóa gradient, tính đầu ra và loss, lan truyền ngược, cập nhật trọng số.
3. Đánh giá trên tập validation.
4. Cập nhật checkpoint, log TensorBoard và scheduler.
5. Kiểm tra điều kiện dừng sớm.

Kết thúc lần chạy, nhật ký được ghi vào `training.runs.<tên lần chạy>` trong `reports/metrics.json`, giữ nguyên kết quả của các lần khác. Tập test không được dùng trong quá trình này.

## 4. Smoke Test
- **Cách chạy**: `python -m pytest tests/test_smoke.py -s`, trước khi train thật.
- **Quy mô**: 2 batch, mỗi batch 64 ảnh $(64, 3, 224, 224)$, dùng dữ liệu và trọng số thật.
- **Trường hợp kiểm tra**: ResNet18 đóng băng, DenseNet121 đóng băng, ResNet18 mở khóa `layer4`.
- **Tiêu chí đạt**:
  - Đầu ra có shape đúng $(64, 10)$.
  - Loss là số hữu hạn.
  - Gradient của lớp phân loại tồn tại và khác 0.
  - Tầng đang đóng băng không bị thay đổi.
  - Chạy đủ 2 batch, không hết bộ nhớ.
  - Khi có GPU, VRAM cao nhất không vượt 4,0 GB.

## 5. TensorBoard (Exercise 3)

- **Nơi lưu log**: `reports/runs/<tên lần chạy>/`. Chạy lại một lần thì log cũ của lần đó bị xóa để các đường không chồng lên nhau.
- **Cách mở**: Cell cuối của notebook chạy `%tensorboard --logdir reports/runs`. Trên máy cá nhân dùng `tensorboard --logdir reports/runs`.
- **Bằng chứng**: Biểu đồ TensorBoard không được lưu cùng notebook, nên cần chụp ảnh và lưu tại `reports/figures/tensorboard.png`.

| Chỉ số | Khi ghi | Ý nghĩa |
| :--- | :--- | :--- |
| `Loss/train` | Mỗi 50 batch | Theo dõi quá trình học |
| `Loss/val` | Mỗi epoch | Phát hiện overfitting |
| `Accuracy/val` | Mỗi epoch | So sánh các lần chạy |
| `LearningRate` | Mỗi epoch | Thấy được lúc scheduler giảm learning rate |

## 6. Checkpoint & Early Stopping
- **Checkpoint**: Xem chính sách trong `lab_2_description.md` mục 7. Mỗi file `.pth` lưu trọng số kèm tên model, dropout, epoch, val accuracy và val loss.
- **Early stopping**:
  - Một epoch chỉ được tính là cải thiện nếu val accuracy tăng **hơn** 1 điểm phần trăm so với mức tốt nhất đang tính.
  - Dừng sau 3 epoch liên tiếp không cải thiện.
  - Với 2.500 ảnh validation, 1 điểm phần trăm tương đương 25 ảnh.

## 7. Xử Lý Tình Huống

| Tình huống | Cách xử lý |
| :--- | :--- |
| Colab ngắt phiên | Chạy lại cell của lần bị ngắt, các lần đã xong không bị ảnh hưởng. |
| Không có GPU | In cảnh báo và chạy bằng CPU. KPI tốc độ và VRAM ghi "Không đo được". |
| Một epoch đạt nhiều mốc tốt nhất | Mỗi mốc vẫn lưu trong file riêng, kể cả khi trọng số giống nhau. |
| Lỗi khi đang lưu checkpoint | Ghi vào file tạm rồi mới thay thế, nên mốc cũ không bị hỏng. |
| Tên lần chạy không có trong cấu hình | Dừng và in danh sách tên hợp lệ. |
| Chưa chạy đủ 4 lần | `evaluation.py` dừng và báo còn thiếu lần nào. |
| Chưa có `checkpoints/best.pth` | `demo.py` dừng và báo cần chạy `evaluation.py` trước. |
