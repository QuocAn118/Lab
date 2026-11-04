# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 3

## 1. Nguồn Gốc & Thông Tin Dataset
- **Tên dataset**: `stanfordnlp/imdb` (hoặc Hugging Face dataset `imdb`).
- **Nhiệm vụ**: Phân loại cảm xúc nhị phân (**Binary Sentiment Analysis**: $0$ = Tiêu cực / Negative, $1$ = Tích cực / Positive).
- **Quy mô tập dữ liệu**:
  - Tập huấn luyện (Train): 25,000 đánh giá phim.
  - Tập kiểm thử (Test): 25,000 đánh giá phim.
  - Khảo sát nhanh / Smoke Test: Lấy mẫu rút gọn (ví dụ 2,000 train / 500 val).

## 2. Tiền Xử Lý Dữ Liệu (Preprocessing Pipeline)
- **Bộ mã hóa (Tokenizer)**: `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.
- **Độ dài chuỗi tối đa (`max_length`)**: 256 tokens (hoặc 512 tokens).
- **Chiến lược Padding / Truncation**:
  - `padding="max_length"` hoặc dùng `DataCollatorWithPadding` để pad động theo lô.
  - `truncation=True` cắt bỏ phần text vượt quá giới hạn.
- **Data Split**:
  - `train_dataset`: Dữ liệu dùng để tính gradient và cập nhật trọng số.
  - `val_dataset`: Dữ liệu dùng để đánh giá `best_val_loss`, `best_val_acc` và kiểm tra `best_val_pause (0-2%)`.
  - `test_dataset`: Dữ liệu đánh giá cuối cùng độc lập.

