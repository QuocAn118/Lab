# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Dữ Liệu Thực Nghiệm (Data Specification) — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

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

Mỗi run ghi class counts, dataset revision, fingerprint và SHA-256 index từng split; xem [08_implementation.md](08_implementation.md).

## Nguồn và phạm vi

- Mục 1–2: đặc tả có sẵn; mục 3 là quyết định triển khai cụ thể hóa lựa chọn cũ.
- [10_sources.md S-03](10_sources.md): dataset card IMDB chính thức.
- [11_AI_Usage.md](11_AI_Usage.md), [01_lab_3_description.md](01_lab_3_description.md): quy tắc và phạm vi.

