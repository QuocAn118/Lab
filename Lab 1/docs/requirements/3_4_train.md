# 3.4. Huấn luyện: forward, loss, backward, optimize

## 1. Mục đích

Huấn luyện mô hình trên tập train, ghi lại cấu hình và diễn biến học để phục vụ đánh giá và so sánh thí nghiệm.

## 2. File thực hiện

- File code dự kiến: `src/train.py`.
- Gọi `create_dataloaders(config)` từ `src/data_pipeline.py` và `build_model(model_config)` từ `src/model.py`.
- Gọi chức năng lưu và kiểm tra tải lại từ `src/model_io.py` ở bước 3.6.
- Sử dụng `CrossEntropyLoss` và optimizer SGD cho cấu hình cơ sở.

## 3. Đầu vào

- Dữ liệu đã chuẩn bị ở bước 3.1; DataLoader và mô hình do bước 3.2–3.3 tạo.
- Tham số dòng lệnh `--run-id` bắt buộc, theo mẫu `run_` và ba chữ số, ví dụ `run_001`.
- Cấu hình cơ sở dự kiến: 10 epoch, batch size 64, learning rate 0.001, SGD với momentum 0.9, seed 42, MLP có hai lớp ẩn `[512, 512]` và ReLU.
- Chọn thiết bị CUDA nếu có, nếu không dùng CPU; ghi lại thiết bị thực tế trong cấu hình.
- Đây là cấu hình khởi đầu của bài làm, không phải cấu hình bắt buộc do đề quy định.

## 4. Các bước xử lý

1. Kiểm tra `run_id` hợp lệ và cả `outputs/<run_id>/` lẫn `checkpoints/<run_id>/` chưa tồn tại. Nếu đã có, dừng và yêu cầu chọn mã mới; không ghi đè thí nghiệm.
2. Tạo hai thư mục của lần chạy; thiết lập seed và ghi cấu hình thực tế trước khi huấn luyện.
3. Khởi tạo DataLoader, mô hình, hàm mất mát và optimizer; chuyển mô hình và các batch sang cùng thiết bị.
4. Đặt mô hình ở chế độ huấn luyện. Với mỗi batch: xóa gradient cũ → forward → tính loss → backward → optimizer cập nhật tham số.
5. Cuối mỗi epoch, tính loss trung bình có trọng số theo số mẫu và accuracy của các dự đoán đã thu trong quá trình huấn luyện epoch đó.
6. Ghi thêm một dòng lịch sử và đóng hoặc flush tệp để giữ kết quả đã hoàn thành.
7. Sau epoch cuối, gọi bước 3.6 để lưu trọng số, tải lại và kiểm tra tính nhất quán.

Không dùng tập test để cập nhật tham số hoặc chọn checkpoint. Quy trình cơ sở lưu mô hình tại epoch cuối; không áp dụng chọn mô hình theo validation hay early stopping.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Đường dẫn | Định dạng và thời điểm ghi |
| --- | --- | --- |
| Cấu hình lần chạy | `outputs/<run_id>/config.json` | JSON UTF-8; ghi trước epoch đầu bằng thư viện JSON. |
| Lịch sử huấn luyện | `outputs/<run_id>/history.csv` | CSV UTF-8 có dòng tiêu đề; tạo trước huấn luyện, thêm một dòng sau mỗi epoch. |
| Trọng số cuối | `checkpoints/<run_id>/final_epoch.pth` | Do chức năng bước 3.6 lưu sau epoch cuối. |
| Kết quả kiểm tra tải lại | `outputs/<run_id>/reload_check.json` | Do chức năng bước 3.6 ghi sau khi kiểm tra. |

Các trường của `config.json`:

- `schema_version`: số nguyên 1; `run_id`: chuỗi mã lần chạy.
- `dataset`: chuỗi `FashionMNIST`; `classes`: danh sách tên lớp theo chỉ số nhãn.
- `model`: đối tượng gồm `name`, `input_shape`, `hidden_dims`, `activation`, `num_classes` như bước 3.3.
- `data`: đối tượng gồm `transforms`, `batch_size`, `num_workers`, `train_shuffle`, `test_shuffle`, `drop_last` theo bước 3.2.
- `training`: đối tượng gồm `epochs`, `seed`, `loss`, `optimizer`, `learning_rate`, `momentum`, `device`.

Các cột của `history.csv`: `epoch` (bắt đầu từ 1), `train_loss`, `train_accuracy`. Accuracy lưu theo tỷ lệ từ 0 đến 1; loss là tổng loss trung bình từng batch nhân số mẫu batch, chia tổng số mẫu của epoch.

Nếu bị gián đoạn, giữ lại cấu hình và các epoch đã ghi; chưa có checkpoint cuối thì lần chạy được xem là chưa hoàn thành. Chạy lại từ đầu phải chọn `run_id` mới; chức năng tiếp tục huấn luyện chưa nằm trong quy trình cơ sở.

## 6. Cách chạy và kiểm tra

Lệnh dự kiến từ thư mục `Lab 1`:

```powershell
python src/train.py --run-id run_001
```

- Có `config.json` ghi đúng các giá trị thực tế đã sử dụng.
- Khi hoàn thành cấu hình cơ sở, `history.csv` có 10 dòng dữ liệu, không có giá trị NaN hoặc vô hạn.
- Accuracy nằm trong `[0, 1]`; loss hữu hạn và không âm. Không coi loss phải giảm ở mọi epoch là điều kiện bắt buộc.
- Có checkpoint cuối và báo cáo kiểm tra tải lại thành công theo bước 3.6.
- Chạy với cùng `run_id` bị từ chối trước khi sửa dữ liệu lần chạy cũ.
