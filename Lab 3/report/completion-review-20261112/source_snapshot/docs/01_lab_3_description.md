# Lab 3 — Phân tích cảm xúc với Transformers

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Lab 3 — Phân tích cảm xúc với Transformers |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Bài toán

Bài này dùng mô hình Transformer để xác định cảm xúc của một đoạn văn bản. Có hai phần: dùng model đã huấn luyện sẵn để dự đoán, sau đó fine-tune DistilBERT trên dữ liệu đánh giá phim IMDB.

## 2. Nội dung thực hiện

**Exercise 1 — Dùng model có sẵn**

Nạp model `tabularisai/multilingual-sentiment-analysis`, nhập một câu, xem các token và kết quả dự đoán. Model trả về nhãn cảm xúc cùng score. Phần này chỉ chạy suy luận, không cập nhật trọng số.

**Exercise 2 — Fine-tune DistilBERT**

1. Chuẩn bị thư viện và đặt seed 42.
2. Tải IMDB, chia train thành 20.000 mẫu train và 5.000 mẫu validation; giữ 25.000 mẫu test riêng.
3. Nạp `distilbert-base-uncased` với hai nhãn: `0 = NEGATIVE`, `1 = POSITIVE`.
4. Tokenize văn bản, cắt ở 256 token và padding theo batch.
5. Đặt batch size 16, learning rate 2e-5, tối đa 3 epochs, optimizer AdamW và weight decay 0.01.
6. Huấn luyện, đánh giá validation sau mỗi epoch và lưu model có validation accuracy tốt nhất. Early stopping dùng patience 2 và ngưỡng cải thiện 0.01.
7. Đánh giá tập test bằng accuracy, precision, recall, binary F1, macro F1; xem confusion matrix, loss curve và thử dự đoán câu mới.

Validation dùng để chọn model. Test chỉ dùng để đánh giá cuối, không dùng để chỉnh tham số hoặc chọn checkpoint.

## 3. Kết quả cần lưu

- Model và tokenizer để nạp lại khi dự đoán.
- Kết quả đánh giá và biểu đồ loss/confusion matrix.
- Thời gian bắt đầu, kết thúc và tổng thời lượng của mỗi lần train.
- Thời lượng từng epoch và thông tin thời gian đi kèm checkpoint.

Notebook chỉ gồm các ô code. Nếu checkpoint đã có, mặc định nạp model để đánh giá; đặt `TRAIN=True` để huấn luyện mới. Nạp model hoặc chạy test không được tính là một lần train mới.

## 4. Checkpoint và thời gian

Trong notebook, Trainer lưu checkpoint theo epoch và giữ tối đa hai checkpoint. Model tốt nhất được lưu thêm tại `outputs/simple-training/<run>/best`.

Mỗi run có `training_time.json`. Mỗi checkpoint cũng có file này để tra epoch, bước cập nhật, thời điểm lưu và số giây đã train. File tổng kết chứa thời lượng toàn bộ lượt train và các epoch đã chạy. Cách đo cụ thể nằm trong [04_training.md §6](04_training.md).

Chương trình `lab3.py` có cách lưu chi tiết hơn gồm `final_epoch`, `best_val_loss`, `best_val_acc`, `best_val_pause` và `best`. Phần mở rộng này được tách khỏi code notebook để bài dễ theo dõi.

Với kết quả cũ thiếu log thời gian, chỉ sử dụng số đã ghi trong báo cáo; mốc bắt đầu hoặc kết thúc chưa lưu thì để trống. Không lấy ngày sửa file làm thời điểm huấn luyện.

## Nguồn và phạm vi

- [02_data.md](02_data.md), [03_model.md](03_model.md), [04_training.md](04_training.md), [05_evaluation.md](05_evaluation.md): dữ liệu, mô hình, tham số và cách đánh giá.
- [08_implementation.md](08_implementation.md): cấu trúc một ô Exercise 1 và bảy bước Exercise 2.
- [10_sources.md](10_sources.md): tài liệu chính thức của thư viện, model và dataset.
- [11_AI_Usage.md](11_AI_Usage.md): quy tắc khai nguồn và phạm vi hỗ trợ AI.
