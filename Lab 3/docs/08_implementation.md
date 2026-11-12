# Thiết kế triển khai và  chạy — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Thiết kế triển khai và  chạy — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-06T21:05:06+07:00 ; HoangDang ; d108ed4 |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

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
- `Lab_3 (1).ipynb`: giữ các bước dữ liệu, model, Trainer, đánh giá và demo; dùng tiện ích nhẹ từ `notebook_helpers.py` thay cho phần ghi thời gian/lưu model dài trong ô code.
- `notebook_helpers.py`: ghi thời gian, train/lưu model và vẽ loss từ log; không đổi dữ liệu, kiến trúc, tham số hoặc metric.
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

Đây là cấu trúc trình bày theo notebook kế thừa, chưa xác minh độc lập nguồn gốc tác giả. Giữ tham số và phạm vi dữ liệu đã chốt; không mặc định khôi phục cấu hình lấy mẫu nhỏ của bản cũ. Giữ hàm tokenize và compute_metrics trong notebook; dùng `EarlyStoppingCallback` có sẵn. Phần ghi thời gian, lưu model và đọc/vẽ lịch sử loss nằm trong `notebook_helpers.py` để nội dung chính ngắn hơn. Callback vẫn ghi đủ thời gian theo [04_training.md §6](04_training.md).

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

## 8. Hồ sơ model

`03_model.md` được trình bày theo bốn phần: (1) tên, ngày tạo, người tạo, ngày/người cập nhật mới nhất; (2) mục đích; (3) initial model gồm nguồn trọng số, kiến trúc, input/output và file mã nguồn; (4) trained model gồm thay đổi sau train, mô tả hoàn tất và đường dẫn lưu.

Tên người tạo và người cập nhật model fine-tuned là **HoangDang**, theo xác nhận ngày 2026-11-07. Ngày ghi nhận bản fine-tuned lấy từ báo cáo 2026-11-06. Ngày cập nhật hồ sơ không được mô tả như ngày train lại hoặc sửa trọng số. Kiến trúc đối chiếu `config.json`; số tham số đếm từ trọng số đang lưu. Phân biệt model pretrained dùng cho Exercise 1 với model fine-tuned của Exercise 2.

## 9. Thư mục report và log theo lượt chạy

`Lab 3/report/` là nơi đọc báo cáo. Mỗi lượt train, đánh giá hoặc nhập lại dữ liệu cũ có thư mục riêng, không ghi đè lượt trước. Trọng số tiếp tục nằm ở `outputs/`; report chỉ lưu log, dữ liệu đo và tham chiếu tới checkpoint.

- `run.json`: loại thao tác, trạng thái, thời điểm bắt đầu/kết thúc, phạm vi log và nguồn dữ liệu.
- `config.json`, `environment.json`: cấu hình thực, kích thước/fingerprint dataset và phiên bản môi trường.
- `stdout.log`, `stderr.log`: nội dung phát ra trong đúng thao tác được ghi; `python.log` ghi logger của Python/thư viện.
- `events.jsonl`, `trainer_history.json`: sự kiện có timestamp và lịch sử metric thực.
- `training_time.json`, `checkpoints.json`: thời gian và danh mục checkpoint; chỉ có dữ liệu train khi thực sự train.
- `metrics.json`, `predictions.csv`, ảnh confusion matrix/loss, `demo.json`: kết quả đánh giá/dự đoán thực khi thao tác có bước tương ứng.
- `summary.md`, `SOURCES.md`, `source_snapshot/`, `manifest.json`: bản đọc nhanh, nguồn, bản sao code/đặc tả và SHA-256 từng file.

`reporting.py` quản lý ghi file/console và manifest. `notebook_helpers.py` sử dụng nó để notebook ngắn; CLI cũng xuất log vào report. Log train/đánh giá được ghi trực tiếp, flush theo tiến trình; lỗi hoặc ngắt vẫn kết thúc bản ghi với trạng thái rõ ràng. Log không bao gồm bước tải dữ liệu/model diễn ra trước khi vào thao tác được ghi.

Đối với run cũ: sao chép nguyên byte các log còn có, ghi đường dẫn nguồn và hash để đối chiếu. Run full cũ chỉ có một phần file; số khôi phục từ báo cáo phải ghi `recovered_from_documentation`. Không tạo stdout/stderr, timestamp hoặc lịch sử loss giả cho phần đã mất. Manifest mới xác minh các file thực có trong report; manifest gốc được giữ riêng với tên `original_provenance.json`.

Khi đưa lên Git, chỉ stage trọng số của phiên đã kiểm chứng bằng đường dẫn cụ thể và quản lý bằng Git LFS. Optimizer/cache tiếp tục được bỏ qua qua `.gitignore`. Code, cấu hình, report và log được lưu trong repository. `.gitattributes` tắt chuyển đổi xuống dòng cho `report/**` và `outputs/**` để giữ nguyên byte bằng chứng và hash sau clone/checkout.

## 10. Hoàn thiện sau nhập nhánh ngày 2026-11-12

Notebook tiếp tục dùng tám ô code và các tiện ích `notebook_helpers.py` đã có ở commit `eb6bfad`. Exercise 1 dùng câu mẫu có thể sửa trong ô để Run All thực thi được. Exercise 2 khai báo `TRAIN=False` khi đánh giá model đã lưu. Khi cần huấn luyện mới, đặt `TRAIN=True` và dùng thư mục run mới theo tiện ích hiện có.

`available_checkpoint(root)` kiểm tra thư mục `outputs/full/checkpoints/best`, sau đó kiểm tra checkpoint được ánh xạ trong `outputs/imported-thanhbinh-40acced/full/checkpoints/best`. Chỉ nhận thư mục có config và trọng số nhị phân, phát hiện pointer Git LFS chưa tải. Nếu thiếu model dùng cho đánh giá, báo đường dẫn và lệnh lấy LFS. Checkpoint được chọn phải được in trong notebook và ghi vào report đánh giá. Metadata nguồn của bản nhập tiếp tục nằm trong `IMPORT.json`.

Luồng nạp model, lịch sử loss còn có hoặc đã mất, save/reload và thời gian train được kiểm thử qua tiện ích sẵn có. Kiểm chứng toàn notebook dùng đúng 20.000/5.000/25.000 mẫu theo mục 2. Kết quả thực thi và môi trường cập nhật tại `06_verification.md`.

Kiểm tra nguồn bỏ qua `.pytest_cache` do pytest tạo, theo phạm vi cache tại `11_AI_Usage.md` mục 2. Các file code và tài liệu của hai bản triển khai vẫn được kiểm tra header và liên kết nguồn.

Bản `src/exercise_2` kế thừa từ Quốc An được bổ sung kiểm tra trước khi train để từ chối checkpoint hoặc báo cáo training đã có. Báo cáo JSON từ chối giá trị NaN/Infinity trước khi ghi. Đánh giá dùng tên GPU thực tế để đối chiếu hai KPI tốc độ tham chiếu T4, đồng thời lưu các thời gian thô. Các thay đổi tích hợp do HoangDang thực hiện với hỗ trợ OpenAI Codex, nguồn code nhập tiếp tục được ghi trong `12_traceability.md`.

Môi trường Windows được kiểm tra với pandas 2.3.3 và pyarrow 21.0.0 từ PyPI khi bản DLL cài trước bị Windows từ chối nạp. Ghi phiên bản sau kiểm chứng trong requirements và lock, cùng các dependency cần cho bản Exercise 2 nhập từ Quốc An.

## Nguồn và phạm vi

- [01_lab_3_description.md](01_lab_3_description.md), [02_data.md](02_data.md), [03_model.md](03_model.md), [04_training.md](04_training.md), [05_evaluation.md](05_evaluation.md), [07_demo.md](07_demo.md): đặc tả đã có.
- [11_AI_Usage.md](11_AI_Usage.md): quy tắc bắt buộc.
- [10_sources.md S-01–S-07](10_sources.md): API chính thức. Các quyết định triển khai bổ sung được ghi nhận tại §1–5 và phân biệt với đặc tả kế thừa.
