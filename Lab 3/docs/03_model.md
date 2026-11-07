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

## 4. Triển khai

- Module A giữ model Exercise 1 của notebook cũ: `tabularisai/multilingual-sentiment-analysis`, năm nhãn; trả token, label, score. Không đánh giá model này bằng nhãn IMDB nhị phân.
- Module B: `distilbert/distilbert-base-uncased` (ID đầy đủ của backbone cũ), fine-tune toàn bộ tham số, head 2 lớp, `0=NEGATIVE`, `1=POSITIVE`. Softmax dùng khi suy luận; train nhận logits tính cross entropy.
- Đếm số tham số thực trong `metrics.json`. Tỷ lệ 40%/97%/60% ở mục 1 là thông tin tác giả model, không phải benchmark trên máy này.
- CNN ở mục 3 là mở rộng, không thuộc phần bắt buộc hiện tại.

## Nguồn và phạm vi

- Mục 1–3: đặc tả có sẵn; mục 4 chốt phạm vi từ [01_lab_3_description.md](01_lab_3_description.md) và notebook Exercise 1.
- [10_sources.md S-04–S-05](10_sources.md): model cards; [11_AI_Usage.md](11_AI_Usage.md): khai báo AI.

