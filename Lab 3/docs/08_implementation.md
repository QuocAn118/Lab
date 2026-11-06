# Thiết kế triển khai và hợp đồng chạy — Lab 3

## 1. Yêu cầu và quyết định đã chốt

| Mã | Hành vi phải có | Căn cứ |
| --- | --- | --- |
| R-01 | Module A: phân tích cảm xúc, trả token/nhãn/xác suất; từ chối chuỗi rỗng | `01_lab_3_description.md` §1; notebook cũ Exercise 1 |
| R-02 | Module B: IMDB, seed 42, chia train/validation riêng; test không chọn model | `02_data.md` §1–2 |
| R-03 | DistilBERT hai nhãn NEGATIVE/POSITIVE; pad động, dài tối đa 256 | `03_model.md` §1; `02_data.md` §2 |
| R-04 | AdamW, LR 2e-5, batch 16, 3 epochs, decay .01, clip 1.0; FP16 khi CUDA | `04_training.md` §1 và quyết định triển khai |
| R-05 | Smoke riêng 2 bước optimizer, loss/gradient hữu hạn; không dùng kết quả smoke làm full | `04_training.md` §2 |
| R-06 | Lưu final_epoch, best_val_loss, best_val_acc, best_val_pause, best; early stop patience 2, ngưỡng .01 điểm accuracy tuyệt đối | `01_lab_3_description.md` §4; `04_training.md` §3 |
| R-07 | Test accuracy, macro/binary F1, precision, recall, confusion matrix; thời gian, GPU/RAM, loss curve | `05_evaluation.md`; `01_lab_3_description.md` §3 |
| R-08 | Nạp lại best từ đĩa để demo và benchmark; không dựa model còn nằm trong RAM | `07_demo.md` |
| R-09 | Mọi file có nguồn MD, bằng chứng không bịa, lưu cấu hình/hash/revision | `11_AI_Usage.md` §2 |

## 2. Chế độ chạy

- `smoke`: 32 train, 16 validation, 16 test; batch 16, `max_steps=2`. Kiểm tra đường đi thật pretrained → gradient → save/load. Không kiểm KPI chất lượng.
- `quick`: 1,000 train, 200 validation, 200 test; tối đa 3 epochs. Dành kiểm tra nhanh, không được mô tả là full benchmark.
- `full`: toàn bộ 25,000 mẫu train được chia phân tầng 80/20 → 20,000 train, 5,000 validation; test đủ 25,000 mẫu. Tối đa 3 epochs và có thể dừng sớm theo R-06. Đây là quyết định triển khai cụ thể hóa đặc tả kế thừa.
- Tất cả split đều phân tầng theo nhãn và seed 42; không đọc split `unsupervised`. Dùng index train/test để ghi fingerprint và SHA-256 index. Không khẳng định loại được mọi trùng lặp ngữ nghĩa của dataset gốc.
- `--output` phải là thư mục mới hoặc rỗng; không ghi đè run có sẵn. Mặc định `Lab 3/outputs/<mode>-<thời gian>`.

## 3. Checkpoint và early stopping

Callback đánh giá theo epoch; nếu accuracy tốt hơn trước bất kỳ mức nào, cập nhật `best_val_acc`; loss nhỏ hơn cập nhật `best_val_loss`. Counter dừng tăng khi accuracy không tăng **hơn 0.01** so với accuracy tốt nhất ở lần đánh giá trước; reset nếu tăng hơn ngưỡng. Dừng sau 2 lần liên tiếp. Đây là ngưỡng tuyệt đối (1 điểm phần trăm), không phải tỷ lệ tăng tương đối.

`best_val_pause` lưu model tại lần đánh giá cuối, kèm cờ `early_stopped`; tồn tại cả khi chạy đủ epoch, không nói dừng sớm nếu không xảy ra. `final_epoch` lưu trọng số cuối trước khi nạp model tốt nhất. `best` là bản sao `best_val_acc`; nếu hòa, giữ checkpoint sớm hơn. Các checkpoint chỉ chứa model/tokenizer để suy luận, không cam kết resume optimizer.

## 4. Phân chia mã nguồn

- `lab3.py`: CLI `infer`, `train`, `demo`, `audit`; cấu hình, dữ liệu, metrics, callback, đo và báo cáo.
- `test.py`: unittest offline, model DistilBERT rất nhỏ khởi tạo ngẫu nhiên **chỉ dùng test**, kiểm gradient, checkpoint và nạp lại. Không thay thế smoke pretrained.
- `Lab_3 (1).ipynb`: bản thực hành rút gọn theo yêu cầu hiện tại, chỉ dùng API thư viện trực tiếp, không import `lab3.py`, không định nghĩa class hoặc kiểm tra AST/hash trong luồng chính. Đọc hợp đồng riêng của notebook tại §6.
- `requirements.txt`: phiên bản môi trường chạy; `requirements-lock.txt` ghi freeze môi trường đã kiểm chứng.
- `outputs/`: run artifacts, không đưa trọng số nhiều GB vào Git; nguồn/hash qua `artifacts.md` và `provenance.json`.

## 5. Đo lường và báo cáo

Đo latency batch 1, 3 lần warm-up + 20 lần, đồng bộ CUDA trước/sau; nêu mean/p50/p95, độ dài token và thiết bị. GPU peak allocation tính trong huấn luyện/eval; RSS host là giá trị cao nhất quan sát ở callback, không cam kết peak tuyệt đối giữa hai lần lấy mẫu. Thời gian epoch có cả validation/save; thời gian train tổng đo wall-clock.

KPI accuracy >= .85 và macro F1 >= .85; GPU <=4 GiB, host <=6 GiB và latency <=25ms. Mốc <=180s/epoch chỉ áp dụng Colab T4: trên GPU khác ghi N/A cho so sánh này. Chất lượng full được đánh giá độc lập; không sửa KPI để khớp số đo.

## 6. Notebook rút gọn

Exercise 1 có một ô code riêng, tự import thư viện để chạy độc lập. Exercise 2 có đúng bảy ô code theo thứ tự bước trong notebook kế thừa tại Git commit `755af75`:

1. Chuẩn bị thư viện và seed; môi trường được cài theo README, không tự chạy pip mỗi lần Run All.
2. Tải IMDB và chia train/validation/test.
3. Tải model và tokenizer.
4. Tokenize dữ liệu và tạo data collator.
5. Cấu hình TrainingArguments.
6. Định nghĩa metrics, tạo Trainer và huấn luyện; nếu đã có checkpoint thì nạp để đánh giá theo công tắc TRAIN.
7. Đánh giá test, confusion matrix, loss curve và demo từ checkpoint.

Đây là cấu trúc trình bày theo notebook kế thừa, chưa xác minh độc lập nguồn gốc tác giả. Giữ tham số và phạm vi dữ liệu đã chốt; không mặc định khôi phục cấu hình lấy mẫu nhỏ của bản cũ. Chỉ giữ một hàm tokenize và một hàm compute_metrics; dùng `EarlyStoppingCallback` có sẵn, không có lớp callback tự viết.

Notebook chỉ giữ các ô code, không có ô Markdown giải thích. Nguồn vẫn được ghi bằng comment `Sources:` ở từng ô và metadata; đầu ra thực nghiệm được giữ nguyên.

- Giữ nguyên train/validation/test độc lập, seed 42, max_length 256, batch 16, LR 2e-5 và tối đa 3 epochs.
- `TRAIN` là công tắc huấn luyện. Nếu đã có checkpoint full, mặc định nạp lại để đánh giá; nếu chưa có checkpoint thì chạy huấn luyện mới. Không gọi việc nạp model là đã train lại.
- Notebook chỉ quản lý model tốt nhất bằng `load_best_model_at_end=True`, lưu tại `outputs/simple-training/<run>/best`. Chính sách 5 checkpoint và benchmark chi tiết ở §1–5 tiếp tục thuộc pipeline CLI đã kiểm chứng, không bắt buộc lặp lại trong notebook rút gọn.
- Dùng `save_strategy=eval_strategy="epoch"`, `save_total_limit=2`, metric accuracy; early stopping patience 2, threshold .01. Trainer dùng AdamW và clipping 1.0.
- Vẽ loss trực tiếp trong notebook. Khi nạp model full đã có, loss history lấy từ run full và ghi rõ nguồn; test metrics được tính lại. Không gán output cũ cho một lượt huấn luyện mới.
- Mỗi ô code giữ một dòng nguồn Markdown. Run huấn luyện mới có `SOURCES.md` áp dụng cho toàn bộ artifact bên dưới thư mục run. Không đưa chức năng audit, manifest, benchmark hệ thống hoặc kiểm thử CLI vào luồng trình bày chính.

## 7. Thứ tự tài liệu và hướng dẫn trình bày

Tài liệu trong `Lab 3/docs` dùng tiền tố số để Explorer hiển thị theo thứ tự đọc: 01 tổng quan, 02 dữ liệu, 03 mô hình, 04 huấn luyện, 05 đánh giá, 06 kết quả kiểm chứng, 07 demo, 08 triển khai, 09 tiến độ, 10 nguồn tham khảo, 11 khai báo AI, 12 truy vết. File `huong_dan_trinh_bay.md` không đánh số. Đây là thứ tự học và trình bày, không phải mức ưu tiên thay thế quy tắc dự án.

Cập nhật đường dẫn trong tài liệu, header nguồn, code hiện hành và metadata notebook khi đổi tên. Giữ nguyên output và snapshot của các run cũ; tên tài liệu trong chúng phản ánh thời điểm chạy, có thể đối chiếu qua bảng tên cũ/tên mới ở `huong_dan_trinh_bay.md`. Chỉ thay tham chiếu tài liệu, không thay tham số hoặc thuật toán huấn luyện.

## Nguồn và phạm vi

- [01_lab_3_description.md](01_lab_3_description.md), [02_data.md](02_data.md), [03_model.md](03_model.md), [04_training.md](04_training.md), [05_evaluation.md](05_evaluation.md), [07_demo.md](07_demo.md): đặc tả đã có.
- [11_AI_Usage.md](11_AI_Usage.md): quy tắc bắt buộc.
- [10_sources.md S-01–S-07](10_sources.md): API chính thức. Các quyết định triển khai bổ sung được ghi nhận tại §1–5 và phân biệt với đặc tả kế thừa.
