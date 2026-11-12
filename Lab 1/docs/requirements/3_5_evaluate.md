# 3.5. Đánh giá độ chính xác

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | 3.5. Đánh giá độ chính xác |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-07T11:17:41+07:00 ; HoangDang ; d41a9fe |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Trạng thái: đã triển khai và có kiểm thử bằng logits biết trước, loader rỗng, logits không hữu hạn và pipeline giả lập. Bốn normal run đã được đánh giá; kết quả tại evaluation.md và verification.md.

## 1. Mục đích

Đánh giá checkpoint `best.pth` được chọn bằng validation trên tập test và lưu kết quả để báo cáo, phân tích dự đoán. Normal run dùng đủ 10.000 ảnh; smoke run dùng 256 ảnh sau split.

## 2. File thực hiện

Giao diện tích hợp: `evaluate_run(config, run, test_loader, summary)` trong `src/evaluate.py`. Hàm chỉ tạo đầu ra khi `Run` còn `running`, tự nạp `best`, đo test, lưu metrics/predictions/performance, đồ thị loss và gọi demo ảnh mẫu. `summary` từ train cung cấp thời gian train đo trực tiếp, mốc bắt đầu/kết thúc và GPU peak; không suy thời gian từ ngày sửa file.

`performance.json` dùng ba nhóm `accuracy`, `timing`, `resources`. Thời gian test đo toàn bộ vòng duyệt loader, chuyển thiết bị và suy luận; ms/ảnh là trung bình của vòng đánh giá theo batch, không phải benchmark một ảnh độc lập. GPU memory và kích thước checkpoint dùng MB = 1.000.000 byte; CPU ghi GPU memory là `null`. Accuracy theo lớp không có mẫu được ghi `null`.

- File code: `src/evaluate.py`.
- Đọc cấu hình của lần chạy, gọi `create_dataloaders(config)` từ `data/data_pipeline.py`, lấy test loader.
- Khởi tạo mạng từ cấu hình và nạp `best.pth`; dùng `load_model(run_id, device, checkpoint="best")` (xem 3.6).

## 3. Đầu vào

- `run_id`, ví dụ `run_001`, xác định lần chạy đã hoàn tất huấn luyện nhưng vẫn ở trong khối `Run`, chưa chốt manifest. Evaluation dùng cấu hình và checkpoint của chính lần chạy đó.
- `outputs/<run_id>/config.json`: cấu hình kiến trúc, transforms và thứ tự lớp nhãn.
- `checkpoints/<run_id>/best.pth`: trọng số có val accuracy cao nhất của cùng lần chạy; hòa giữ epoch sớm hơn.
- Tập test trong `data/FashionMNIST/raw/`.

## 4. Các bước xử lý

1. Kiểm tra đủ cấu hình, checkpoint và dữ liệu; nếu thiếu thì dừng và báo đường dẫn còn thiếu.
2. Tải mô hình trên thiết bị sẵn có, chuyển sang chế độ đánh giá và tắt tính gradient.
3. Duyệt toàn bộ test_loader với `shuffle=False`, áp dụng đúng transforms đã ghi trong cấu hình.
4. Tính loss bằng CrossEntropyLoss và lấy chỉ số lớp có logit lớn nhất làm nhãn dự đoán.
5. Tính accuracy bằng tổng số mẫu đúng chia tổng số mẫu test; tính test loss trung bình có trọng số theo số mẫu.
6. Lưu chỉ số tổng hợp và kết quả từng mẫu. Không gọi backward hay cập nhật trọng số.
7. Đo và ghi ba nhóm performance theo đặc tả gốc §8; đồng bộ CUDA trước và sau đoạn đo thời gian nếu dùng GPU. Accuracy theo lớp tính từ dự đoán; thông tin train lấy từ lịch sử và metadata checkpoint, không lấy ngày sửa file.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Đường dẫn | Định dạng |
| --- | --- | --- |
| Chỉ số tổng hợp | `outputs/<run_id>/metrics.json` | JSON UTF-8, ghi bằng thư viện JSON sau khi đánh giá thành công. |
| Dự đoán từng ảnh | `outputs/<run_id>/predictions.csv` | CSV UTF-8 có dòng tiêu đề, ghi theo thứ tự của tập test. |
| Performance | `outputs/<run_id>/performance.json` | JSON có ba nhóm accuracy, thời gian, tài nguyên theo đặc tả gốc §8. |

`metrics.json` gồm:

- `run_id`: mã thí nghiệm; `checkpoint`: đường dẫn tương đối của tệp trọng số đã dùng.
- `split`: chuỗi `test`; `num_samples`: số mẫu đánh giá; `num_correct`: số mẫu đúng.
- `test_loss`: loss trung bình trên toàn tập test.
- `test_accuracy`: tỷ lệ đúng trong `[0, 1]`, không lưu dưới dạng chuỗi phần trăm.
- `device`: thiết bị thực tế dùng khi đánh giá.

Các cột của `predictions.csv`: `sample_index`, `true_label`, `predicted_label`, `true_name`, `predicted_name`. `sample_index` bắt đầu từ 0 trong tập test gốc; các trường label là số nguyên từ 0 đến 9; các trường name lấy từ danh sách `classes` đã lưu.

Chỉ công bố kết quả sau khi duyệt thành công toàn bộ test loader; có thể ghi ra tệp tạm rồi thay thế đích. Đánh giá, vẽ đồ thị và kiểm chứng phải diễn ra trong vòng đời `Run` trước khi manifest được chốt. Không thêm hoặc ghi đè báo cáo vào run đã kết thúc; chạy thí nghiệm mới dùng `run_id` mới. `train.py` đã tích hợp các bước này.

## 6. Cách chạy và kiểm tra

Evaluation được gọi qua CLI train, không có bước ghi evaluation vào run đã chốt. Quy trình là train → tải/kiểm tra `best` → đánh giá test → demo/đồ thị → kiểm số liệu, ghi `verify_report.json` → thoát khối `Run` để chốt manifest. Chữ ký hàm tích hợp là `evaluate_run(config, run, test_loader, summary)`.

Sau khi manifest đã chốt, chỉ kiểm hash ở chế độ đọc; không thêm báo cáo vào run cũ. Kiểm tra nguồn gốc và hash bằng `python tools/check_provenance.py --strict` từ `Lab 1`.

- Normal: `num_samples` và số dòng CSV đều bằng 10.000, `sample_index` từ 0 đến 9.999; smoke: 256 mẫu, chỉ số từ 0 đến 255. Không trùng hoặc thiếu.
- Đếm lại số dòng có `true_label` bằng `predicted_label` phải khớp `num_correct` và `test_accuracy` trong JSON.
- Loss hữu hạn, không âm; accuracy nằm trong `[0, 1]`.
- Các tên lớp khớp chỉ số nhãn trong cấu hình; trọng số không thay đổi sau đánh giá.
- Không đặt ngưỡng accuracy bắt buộc khi slide chưa quy định.
