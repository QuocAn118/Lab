# Kiến Trúc Mô Hình (Model Specification) — Lab 3

Lab 3 dùng hai model khác nhau cho hai mục đích khác nhau.

| | Exercise 1 | Exercise 2 |
| :--- | :--- | :--- |
| **Model** | `tabularisai/multilingual-sentiment-analysis` | `distilbert-base-uncased` |
| **Cách dùng** | Chỉ suy luận, không huấn luyện | Fine-tune trên IMDB |
| **Số nhãn** | 5 | 2 |
| **Mã nguồn** | `src/exercise_1/sentiment_analysis.py` | `src/exercise_2/model.py` |

## 1. Exercise 1
- **Lý do chọn**: Model hỗ trợ nhiều ngôn ngữ nên phân tích được cả câu tiếng Việt và tiếng Anh.
- **Kiến trúc**: Họ DistilBERT, đã huấn luyện sẵn cho bài toán cảm xúc.
- **Nhãn đầu ra**: `Very Negative`, `Negative`, `Neutral`, `Positive`, `Very Positive`.

## 2. Exercise 2

### Backbone
`distilbert-base-uncased`, bản rút gọn của BERT. Model nhỏ và nhanh hơn BERT nên phù hợp với lượng tài nguyên hạn chế.

### Lớp phân loại
Model được tải qua `AutoModelForSequenceClassification` với `num_labels=2`. Thư viện khởi tạo mới các lớp sau:

| Thứ tự | Lớp | Kích thước |
| :---: | :--- | :--- |
| 1 | `pre_classifier` (Linear) | $768 \to 768$ |
| 2 | ReLU | |
| 3 | `dropout` | $p = 0.2$ |
| 4 | `classifier` (Linear) | $768 \to 2$ |

- **Đầu ra của model**: Logits của 2 lớp. Hàm mất mát Cross-Entropy nhận trực tiếp logits.
- **Softmax**: Chỉ dùng trong `demo.py` để đổi logits thành xác suất hiển thị.
- **Nhãn**: `id2label = {0: "NEGATIVE", 1: "POSITIVE"}`.

### Tham số

| Chỉ số | Giá trị | Nguồn |
| :--- | :---: | :--- |
| Tổng tham số | 66.955.010 | Đo bằng `python -m exercise_2.model` |
| Tham số huấn luyện được | 66.955.010 | Đo bằng `python -m exercise_2.model` |

- **Phạm vi huấn luyện**: Toàn bộ model, không đóng băng tầng nào.
- **Kiểu dữ liệu**: Mixed Precision FP16 khi có GPU, FP32 khi chạy CPU.
