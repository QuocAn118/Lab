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

## 3. Cấu hình đã chốt

Seed 42; chia phân tầng train gốc thành 20,000 train và 5,000 validation; giữ 25,000 test độc lập. `full` dùng toàn bộ ba tập. `quick` lấy 1,000/200/200, `smoke` lấy 32/16/16; hai chế độ này chỉ kiểm thử rút gọn. Không dùng test chọn epoch, không dùng unsupervised. `max_length=256`, padding động; GPU pad đến bội 8 khi train.

Mỗi run ghi class counts, dataset revision, fingerprint và SHA-256 index từng split; xem [implementation.md](implementation.md).

## Nguồn và phạm vi

- Mục 1–2: đặc tả có sẵn; mục 3 là quyết định triển khai cụ thể hóa lựa chọn cũ.
- [sources.md S-03](sources.md): dataset card IMDB chính thức.
- [AI_Usage.md](AI_Usage.md), [lab_3_description.md](lab_3_description.md): quy tắc và phạm vi.

