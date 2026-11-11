# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 3

Exercise 1 chỉ nhận một câu do người dùng nhập, không dùng dataset. Toàn bộ file này áp dụng cho Exercise 2.

Mã nguồn: `src/exercise_2/data.py`. Tham số: mục `data` trong `configs/exercise_2.yaml`.

## 1. Nguồn Gốc Dataset
- **Tên dataset**: `stanfordnlp/imdb` trên Hugging Face Hub.
- **Nhiệm vụ**: Phân loại cảm xúc nhị phân: $0$ = `NEGATIVE`, $1$ = `POSITIVE`.
- **Quy mô gốc**: 25.000 mẫu train và 25.000 mẫu test. Dataset không có sẵn tập validation.

## 2. Phân Chia Dữ Liệu

| Tập | Số mẫu | Lấy từ | Vai trò |
| :--- | :---: | :--- | :--- |
| `train` | 1.000 | Tách từ `train` gốc | Tính gradient và cập nhật trọng số |
| `val` | 200 | Tách từ `train` gốc, ngay sau tập `train` | Theo dõi loss, accuracy; chọn checkpoint; dừng sớm |
| `test` | 200 | Tách từ `test` gốc | Chỉ đánh giá một lần trên mốc `best` |

- **Lý do chọn số mẫu**: Đề bài không quy định số mẫu. Nhóm chọn lượng dữ liệu nhỏ để chạy nhanh và trong giới hạn tài nguyên.
- **Tái lập**: Xáo trộn với `seed = 42` trước khi lấy mẫu.
- **Chống trùng mẫu**: `train` và `val` là hai đoạn không giao nhau sau khi xáo trộn. `test` lấy từ tập gốc khác. Chạy `python -m exercise_2.data` để báo cáo thêm số review có nội dung trùng nhau, vì IMDB gốc có thể có review lặp lại.
- **Cân bằng nhãn**: Không ép cân bằng, chỉ báo cáo số mẫu mỗi nhãn trong từng tập vào mục `data` của `reports/metrics.json`.

## 3. Tiền Xử Lý
- **Làm sạch**: Thay thẻ `<br />` bằng dấu cách, vì thẻ này sẽ bị tách thành các token vô nghĩa và chiếm chỗ trong giới hạn độ dài. Áp dụng cho cả 3 tập và khi đo độ trễ.
- **Tokenizer**: `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.
- **Độ dài tối đa**: `max_length = 256`, `truncation = True` cắt phần vượt quá.
- **Padding**: Dùng `DataCollatorWithPadding` để pad động theo từng batch.
- **Xử lý lỗi**: Nếu số mẫu yêu cầu lớn hơn số mẫu của dataset thì báo `ValueError`.
