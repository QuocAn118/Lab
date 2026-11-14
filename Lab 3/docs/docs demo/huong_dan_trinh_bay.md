# Hướng dẫn đọc tài liệu và trình bày Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Hướng dẫn đọc tài liệu và trình bày Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-06T22:10:18+07:00 ; HoangDang ; 77cd208 |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Thứ tự đọc

Đọc các file 01–07 để hiểu và trình bày bài. Các file 08–12 dùng để tra cứu cách triển khai, tiến độ và nguồn. Tiền tố số tạo thứ tự hiển thị khi sắp xếp theo tên trong Explorer.

| Thứ tự | File hiện tại | Tên cũ | Nội dung cần nắm |
| --- | --- | --- | --- |
| Mở đầu | [Hướng dẫn trình bày](huong_dan_trinh_bay.md) | File mới | Cách đọc docs và giải thích notebook |
| 01 | [Tổng quan](../01_lab_3_description.md) | `lab_3_description.md` | Bài toán, hai bài thực hành, phạm vi |
| 02 | [Dữ liệu](../02_data.md) | `data.md` | IMDB, nhãn, train/validation/test, token hóa |
| 03 | [Mô hình](../03_model.md) | `model.md` | Model cảm xúc có sẵn và DistilBERT hai lớp |
| 04 | [Huấn luyện](../04_training.md) | `training.md` | Learning rate, batch, epochs, optimizer, early stopping |
| 05 | [Đánh giá](../05_evaluation.md) | `evaluation.md` | Ý nghĩa các chỉ số và file kết quả |
| 06 | [Kết quả kiểm chứng](../06_verification.md) | `verification.md` | Kết quả thực tế và giới hạn |
| 07 | [Demo](../07_demo.md) | `demo.md` | Nạp model đã lưu để dự đoán câu mới |
| 08 | [Thiết kế triển khai](../08_implementation.md) | `implementation.md` | Quyết định kỹ thuật, khác biệt notebook/CLI |
| 09 | [Tiến độ](../09_plan_progress_status.md) | `plan_progress_status.md` | Công việc và nhật ký |
| 10 | [Nguồn kỹ thuật](../10_sources.md) | `sources.md` | Tài liệu API, model và dataset |
| 11 | [Khai báo AI](../11_AI_Usage.md) | `AI_Usage.md` | Quy tắc khai nguồn và phạm vi hỗ trợ AI |
| 12 | [Truy vết](../12_traceability.md) | `traceability.md` | Yêu cầu tương ứng với code và phép kiểm tra |

Tên cũ trong bảng chỉ dùng đối chiếu các bản lưu lịch sử. Artifact và snapshot trong `outputs/` được giữ nguyên, kể cả đường dẫn tài liệu tại thời điểm tạo run.

## 2. Mở đầu phần trình bày

Lab 3 thực hiện phân tích cảm xúc văn bản bằng Transformers. Exercise 1 dùng một model đã huấn luyện sẵn để dự đoán cảm xúc. Exercise 2 fine-tune DistilBERT trên các đánh giá phim tiếng Anh của IMDB để phân loại NEGATIVE hoặc POSITIVE, sau đó đánh giá trên tập test riêng.

Notebook có tám ô code: một ô Exercise 1 và bảy ô cho Exercise 2. Nội dung chính giữ các bước dữ liệu, model, Trainer và đánh giá; phần ghi thời gian/lưu model/vẽ loss được chuyển sang `notebook_helpers.py`. Chương trình `lab3.py` cung cấp thêm CLI, benchmark và quản lý artifact; phạm vi của nó rộng hơn phần trình bày trong notebook.

## 3. Giải thích Exercise 1

Ô đầu nạp pipeline `tabularisai/multilingual-sentiment-analysis`, hiển thị token và trả nhãn cảm xúc cùng score cho câu mẫu. Token là đơn vị văn bản mà tokenizer tạo ra. Score là điểm xác suất model gán cho nhãn dự đoán; một câu mẫu không chứng minh độ chính xác của toàn bộ model. Bài này thực hiện suy luận bằng trọng số có sẵn.

## 4. Giải thích bảy bước Exercise 2

| Bước | Code làm gì? | Giải thích khi trình bày |
| --- | --- | --- |
| 1. Chuẩn bị thư viện | Import thư viện, đặt seed 42 và kiểm tra CUDA | Chuẩn bị công cụ xử lý dữ liệu, model và huấn luyện. Seed giúp tái lập; không bảo đảm mọi phần cứng cho kết quả giống hệt. |
| 2. Tải dữ liệu | Chia 25.000 train gốc thành 20.000 train, 5.000 validation; giữ 25.000 test | Train dùng cập nhật trọng số; validation dùng chọn model và theo dõi huấn luyện; test dùng đánh giá cuối. Chia phân tầng để giữ tỷ lệ nhãn. |
| 3. Tải model/tokenizer | Nạp DistilBERT hai lớp và tokenizer tương ứng; chọn TRAIN hay checkpoint có sẵn | Dùng kiến thức ngôn ngữ đã học trước. Khi TRAIN=True, mô hình được fine-tune cho phân loại cảm xúc; khi TRAIN=False, dùng trọng số đã huấn luyện. |
| 4. Tokenize | Chuyển văn bản thành token IDs, cắt tối đa 256 token và pad động theo batch | Model nhận dữ liệu số. Padding giúp các câu trong cùng batch có chiều dài tương thích. |
| 5. Cấu hình | Đặt learning rate 2e-5, batch 16, tối đa 3 epochs, AdamW, weight decay .01 | Learning rate điều chỉnh mức cập nhật; batch là số mẫu mỗi lượt; epoch là một lượt qua tập train. CUDA dùng FP16. |
| 6. Huấn luyện | Định nghĩa metrics, tạo Trainer, chạy train nếu bật TRAIN, lưu best | Trainer tính loss, gradient và cập nhật trọng số. Validation chọn model theo accuracy. File training_time.json ghi tổng thời lượng và thời gian từng epoch/checkpoint. Early stopping có patience 2 và ngưỡng cải thiện .01, tức 1 điểm phần trăm. |
| 7. Đánh giá | Dự đoán tập test, tính metrics, vẽ confusion matrix/loss và demo | Báo chất lượng trên dữ liệu độc lập. Sau đó nạp checkpoint để dự đoán hai câu mới. |

## 5. Giải thích kết quả

Lần full training đã ghi nhận test accuracy 91.012%: dự đoán đúng 22.753 trên 25.000 mẫu. Macro F1 khoảng .9101 là trung bình F1 của hai lớp. Confusion matrix cho biết số mẫu đúng và số mẫu bị nhầm theo từng nhãn.

Train loss giảm nhưng validation loss tăng từ epoch 1 đến 3 là dấu hiệu overfitting. Model best trong run full được chọn theo validation accuracy, nên có thể khác model đạt validation loss nhỏ nhất. Không dùng test để chọn epoch.

Khi notebook hiển thị chế độ nạp model đã huấn luyện, lần Run All đó chỉ đánh giá lại checkpoint. Biểu đồ loss lấy từ lịch sử full đã lưu; không mô tả lần chạy đó là một lượt full training mới. Cấu hình early stopping không đồng nghĩa đã dừng sớm: run full đã chạy đủ ba epochs.

## 6. Mở bằng chứng khi trình bày

- [Kết quả đã ghi nhận](../06_verification.md): chỉ số, cách đo và giới hạn.
- [Report riêng](../../report/README.md): chọn lượt chạy, mở log, cấu hình, thời gian, metrics và từng file dự đoán; nguồn/hash nằm trong manifest của lượt đó.
- [Thời gian run cũ](../../outputs/full/training_time.json) và [nguồn khôi phục](../../outputs/full/timing_sources.md): 334.46 giây tổng; 113.11, 111.24, 109.95 giây theo epoch. Đây là số từ báo cáo, không phải log timestamp gốc.
- [Notebook](../../Lab_3%20%281%29.ipynb): kết quả test, confusion matrix và demo được tính lại.
- [Cấu hình train](../04_training.md): giải thích tham số và cách ghi thời gian.

Mỗi lượt train mới có thư mục riêng, training_time.json ở run, checkpoint và best. Khi chỉ nạp model để đánh giá, đọc file thời gian sẵn có; không tính thời gian đánh giá thành thời gian train. Nếu thiếu log loss cũ, notebook báo rõ và bỏ qua việc vẽ lại. Không lấy ngày sửa file làm thời điểm huấn luyện.

## 7. Vị trí tài liệu sau sắp xếp

Hướng dẫn này giải thích bộ chính 01–12, dù nằm trong thư mục docs demo. Bảy file data.md, model.md, training.md, evaluation.md, demo.md, lab_3_description.md và plan_progress_status.md cùng thư mục là bản Quốc An tham khảo; xem README của thư mục để phân biệt. Chỉ dùng checkpoint có trọng số tại outputs/imported-thanhbinh-40acced/full khi outputs/full thiếu trọng số. Không tự train khi TRAIN=False và không có checkpoint.

## Nguồn và phạm vi

- [08_implementation.md §6–7](../08_implementation.md): cấu trúc notebook và thứ tự tài liệu đã chốt.
- [02_data.md](../02_data.md), [03_model.md](../03_model.md), [04_training.md](../04_training.md), [05_evaluation.md](../05_evaluation.md): giải thích dữ liệu và phương pháp.
- [06_verification.md](../06_verification.md) và [nguồn thời gian](../../outputs/full/timing_sources.md): kết quả đã ghi nhận, giới hạn và phân biệt train với nạp checkpoint.
- [11_AI_Usage.md](../11_AI_Usage.md): quy tắc khai nguồn. Hướng dẫn này tổng hợp cách trình bày, dựa trên phạm vi và đặc tả hiện có.
