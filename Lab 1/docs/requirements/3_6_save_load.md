# 3.6. Lưu và tải lại mô hình

## 1. Mục đích

Lưu trọng số cuối của từng thí nghiệm, khôi phục đúng mô hình và kiểm tra dự đoán trước/sau khi lưu tải.

## 2. File thực hiện

- File code dự kiến: `src/model_io.py`, chứa các hàm dùng chung:
  - `save_model(model, run_id)`: lưu state_dict tại đường dẫn của lần chạy.
  - `load_model(run_id, device)`: đọc cấu hình, khởi tạo mạng, nạp trọng số và trả về mô hình ở chế độ đánh giá.
  - `verify_reload(model, sample_batch, run_id)`: so sánh mô hình trong bộ nhớ với mô hình tải lại và ghi báo cáo.
- `src/train.py` gọi lưu và kiểm tra sau epoch cuối; `src/evaluate.py` gọi tải lại trước đánh giá.
- `src/model.py` cung cấp `build_model(model_config)` để khởi tạo đúng kiến trúc.

## 3. Đầu vào

- Mô hình đã huấn luyện và `run_id` của bước 3.4.
- `outputs/<run_id>/config.json`, đã được bước 3.4 ghi; đây là bản cấu hình duy nhất cho lần chạy, không tạo bản sao trong thư mục checkpoint.
- Một batch cố định gồm 16 mẫu đầu tiên của tập train, đã áp dụng transforms như cấu hình, dùng để kiểm tra lưu/tải.
- Khi đánh giá, nhận `run_id` và thiết bị cần tải mô hình lên.

## 4. Các bước xử lý

1. Sau epoch cuối, kiểm tra thư mục lần chạy và cấu hình tương ứng tồn tại; không ghi đè checkpoint đã có.
2. Lưu `model.state_dict()` bằng `torch.save` vào tệp trọng số của lần chạy.
3. Đọc `config.json`, sử dụng `build_model` để tạo lại cùng kiến trúc.
4. Đọc checkpoint bằng `torch.load` với `weights_only=True` và `map_location` theo thiết bị đích; nạp bằng `load_state_dict` với kiểm tra khớp đầy đủ các khóa.
5. Chuyển mô hình tải lại sang chế độ đánh giá. Bên gọi thực hiện dự đoán khi tắt tính gradient.
6. Để kiểm tra lưu/tải, so sánh trên CPU: chuyển mô hình gốc sang chế độ đánh giá trên CPU sau khi huấn luyện xong; tải mô hình mới lên CPU; đưa cùng batch 16 ảnh qua cả hai mạng khi tắt tính gradient.
7. So sánh logits và nhãn dự đoán, ghi báo cáo rồi trả trạng thái kiểm tra về `train.py`.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Đường dẫn hoặc nơi sử dụng | Định dạng |
| --- | --- | --- |
| Trọng số epoch cuối | `checkpoints/<run_id>/final_epoch.pth` | Tệp nhị phân do `torch.save` tạo, nội dung là state_dict. |
| Mô hình đã tải | Bộ nhớ của tiến trình gọi `load_model` | Đối tượng mạng đã nạp trọng số và được đặt ở chế độ đánh giá. |
| Kết quả đối chiếu | `outputs/<run_id>/reload_check.json` | JSON UTF-8, ghi sau khi thực hiện so sánh. |

Phần mở rộng `.pth` là quy ước đặt tên. Tệp này chứa các tham số và buffer trong state_dict; không chứa mã định nghĩa kiến trúc, cấu hình tiền xử lý hay trạng thái optimizer. Cần giữ cùng `src/model.py` và `config.json` để tải lại; quy trình cơ sở phục vụ suy luận, chưa hỗ trợ tiếp tục huấn luyện từ checkpoint.

`reload_check.json` gồm `run_id`, `checkpoint`, `device` (chuỗi `cpu`), `checked_samples` (16), `labels_match` (boolean), `max_abs_logit_difference` (số thực), `tolerance` (0.000001) và `passed` (boolean).

Giữ checkpoint và cấu hình để phân tích nếu kiểm tra thất bại; ghi `passed=false` khi đã tính được kết quả so sánh nhưng không đạt. Nếu việc tải hoặc đối chiếu phát sinh lỗi, báo lỗi và không báo lần chạy hoàn thành thành công. Mỗi lần huấn luyện mới phải dùng mã khác để không ghi đè checkpoint cũ.

## 6. Cách gọi và kiểm tra

Module này được gọi trong bước huấn luyện và đánh giá, không cần thêm lệnh CLI riêng:

1. `train.py` gọi `save_model`, sau đó gọi `verify_reload` sau epoch cuối.
2. `verify_reload` gọi `load_model` để kiểm tra chính checkpoint vừa lưu.
3. `evaluate.py` gọi `load_model` trước khi đánh giá tập test.

Điều kiện hoàn thành:

- Tệp `.pth` tồn tại và nạp được vào mô hình mới với kiến trúc từ cấu hình.
- Dự đoán của 16 mẫu giống nhau trước/sau khi tải (`labels_match=true`).
- Chênh lệch tuyệt đối lớn nhất giữa logits của hai mô hình không vượt quá 0.000001.
- `reload_check.json` có `passed=true` khi đồng thời đạt cả điều kiện logits và nhãn.
- Có thể tải checkpoint đã huấn luyện trên GPU về CPU qua `map_location`.

Tham khảo API: [Saving and Loading Models — PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models).
