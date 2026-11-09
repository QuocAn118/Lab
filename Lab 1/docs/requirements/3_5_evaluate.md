# 3.5. Đánh giá độ chính xác

## 1. Mục đích

Đánh giá mô hình tại epoch cuối trên toàn bộ tập test và lưu kết quả để báo cáo, phân tích dự đoán.

## 2. File thực hiện

- File code dự kiến: `src/evaluate.py`.
- Đọc cấu hình của lần chạy, gọi `create_dataloaders(config)` từ `src/data_pipeline.py`.
- Gọi `load_model(run_id, device)` từ `src/model_io.py` để khởi tạo mạng và nạp trọng số.

## 3. Đầu vào

- `--run-id`, ví dụ `run_001`, xác định đúng một thí nghiệm đã hoàn thành.
- `outputs/<run_id>/config.json`: cấu hình kiến trúc, transforms và thứ tự lớp nhãn.
- `checkpoints/<run_id>/final_epoch.pth`: trọng số của cùng lần chạy.
- Tập test trong `data/FashionMNIST/raw/`.

## 4. Các bước xử lý

1. Kiểm tra đủ cấu hình, checkpoint và dữ liệu; nếu thiếu thì dừng và báo đường dẫn còn thiếu.
2. Tải mô hình trên thiết bị sẵn có, chuyển sang chế độ đánh giá và tắt tính gradient.
3. Duyệt toàn bộ test_loader với `shuffle=False`, áp dụng đúng transforms đã ghi trong cấu hình.
4. Tính loss bằng CrossEntropyLoss và lấy chỉ số lớp có logit lớn nhất làm nhãn dự đoán.
5. Tính accuracy bằng tổng số mẫu đúng chia tổng số mẫu test; tính test loss trung bình có trọng số theo số mẫu.
6. Lưu chỉ số tổng hợp và kết quả từng mẫu. Không gọi backward hay cập nhật trọng số.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Đường dẫn | Định dạng |
| --- | --- | --- |
| Chỉ số tổng hợp | `outputs/<run_id>/metrics.json` | JSON UTF-8, ghi bằng thư viện JSON sau khi đánh giá thành công. |
| Dự đoán từng ảnh | `outputs/<run_id>/predictions.csv` | CSV UTF-8 có dòng tiêu đề, ghi theo thứ tự của tập test. |

`metrics.json` gồm:

- `run_id`: mã thí nghiệm; `checkpoint`: đường dẫn tương đối của tệp trọng số đã dùng.
- `split`: chuỗi `test`; `num_samples`: số mẫu đánh giá; `num_correct`: số mẫu đúng.
- `test_loss`: loss trung bình trên toàn tập test.
- `test_accuracy`: tỷ lệ đúng trong `[0, 1]`, không lưu dưới dạng chuỗi phần trăm.
- `device`: thiết bị thực tế dùng khi đánh giá.

Các cột của `predictions.csv`: `sample_index`, `true_label`, `predicted_label`, `true_name`, `predicted_name`. `sample_index` bắt đầu từ 0 trong tập test gốc; các trường label là số nguyên từ 0 đến 9; các trường name lấy từ danh sách `classes` đã lưu.

Chỉ công bố hai tệp kết quả cuối khi duyệt thành công toàn bộ tập test; có thể ghi ra tệp tạm rồi thay thế đích. Khi đánh giá lại cùng checkpoint, thay thế hai báo cáo của lần đánh giá trước; không sửa cấu hình, lịch sử hay trọng số.

## 6. Cách chạy và kiểm tra

Lệnh dự kiến từ thư mục `Lab 1`:

```powershell
python src/evaluate.py --run-id run_001
```

- `num_samples` và số dòng dữ liệu trong CSV đều bằng 10.000.
- `sample_index` liên tục từ 0 đến 9.999, không trùng hoặc thiếu.
- Đếm lại số dòng có `true_label` bằng `predicted_label` phải khớp `num_correct` và `test_accuracy` trong JSON.
- Loss hữu hạn, không âm; accuracy nằm trong `[0, 1]`.
- Các tên lớp khớp chỉ số nhãn trong cấu hình; trọng số không thay đổi sau đánh giá.
- Không đặt ngưỡng accuracy bắt buộc khi slide chưa quy định.
