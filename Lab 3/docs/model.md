# Kiến Trúc Mô Hình (Model Specification) — Lab 3

## 1. Mô Hình Lựa Chọn
- **Backbone**: `distilbert-base-uncased` (Phiên bản rút gọn tinh gọn nhẹ hơn 40% của BERT, giữ lại 97% hiệu năng và tốc độ nhanh hơn 60%).
- **Lớp phân loại (Classification Head)**:
  - Tầng đầu ra: `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`.
  - Hàm kích hoạt đầu ra: Softmax (cho xác suất 2 lớp).

## 2. Thông Số Mô Hình (Model Parameters)
- Tổng số tham số (Total Parameters): $\approx 66\text{M}$ tham số.
- Số tham số có thể huấn luyện (Trainable Parameters): Toàn bộ hoặc tầng Classifier.
- Kiểu dữ liệu trọng số: FP32 hoặc Mixed Precision FP16 (tiết kiệm VRAM).

## 3. Kiến Trúc Khảo Sát Bổ Sung (Nghiên Cứu Mở Rộng)
- So sánh hiệu năng và tốc độ giữa Transformer (`DistilBERT`) với các mô hình CNN truyền thống khi xử lý dữ liệu phân loại.

