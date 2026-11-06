# Kiểm chứng thực tế — Lab 3

## 1. Kết luận ngày 2026-11-06

Đã triển khai và chạy thực tế cả hai module, full training của pipeline CLI và notebook. Bản notebook hiện tại đã được rút gọn: nạp checkpoint full để đánh giá lại; luồng huấn luyện mới được kiểm tra riêng trên dữ liệu nhỏ. Môi trường: Python 3.12.10, PyTorch 2.9.1+cu128, Transformers 4.57.1, Windows 11, **NVIDIA GeForce RTX 5060 Laptop GPU**. Dependency chính: requirements.txt; toàn bộ phiên bản thực: requirements-lock.txt.

Full run: 20,000 train / 5,000 validation / 25,000 test, mỗi tập cân bằng hai lớp; seed 42, max_length 256, batch 16, 3 epochs, AdamW 2e-5, weight decay .01, FP16. Đã chạy **3,750 optimizer steps**, thời gian train wall **334.46 giây**, không tính tải dữ liệu/model và đánh giá test cuối.

## 2. Số đo trực tiếp từ full run

| Chỉ số | Giá trị | Kết luận theo đặc tả nội bộ |
| --- | ---: | --- |
| Validation accuracy (best_val_acc) | 91.34% | PASS >=85% |
| Test accuracy (25,000 mẫu) | 91.012% | PASS >=85% |
| Test macro F1 | 0.910118 | PASS >=.85 |
| Test binary F1 | 0.909726 | Chỉ số lớp POSITIVE |
| Test precision | 0.913728 | Đã đo |
| Test recall | 0.905760 | Đã đo |
| Test loss | 0.346539 | Đã đo |
| Latency mean / p95 | 4.330 / 4.709 ms | PASS mean <=25ms |
| GPU peak allocated | 1.676 GiB | PASS <=4 GiB |
| Host peak observed RSS | 2.284 GiB | PASS <=6 GiB; giá trị lấy mẫu |
| Tốc độ <=180s/epoch trên Colab T4 | N/A | Máy chạy là RTX 5060 Laptop |

Latency batch 1, review test 256 token, 3 warm-up + 20 lần, gồm tokenize/transfer/forward và đồng bộ CUDA. Không suy rộng cho mọi độ dài/phần cứng. Confusion matrix [[11431,1069],[1178,11322]]: hàng nhãn thật NEG/POS, cột dự đoán NEG/POS; 22,753/25,000 dự đoán đúng.

| Epoch | Val loss | Val accuracy | Epoch wall (s) |
| ---: | ---: | ---: | ---: |
| 1 | 0.238463 | 91.04% | 113.11 |
| 2 | 0.257449 | 91.32% | 111.24 |
| 3 | 0.339619 | 91.34% | 109.95 |

**Nhận xét:** train loss giảm trong khi validation loss tăng từ epoch 1 → 3: dấu hiệu overfitting/tăng tự tin ở dự đoán sai. Validation accuracy chỉ tăng nhẹ. Theo chính sách đã chốt, `best` lấy epoch 3 có validation accuracy cao nhất; `best_val_loss` giữ epoch 1. Không đổi chính sách bằng cách nhìn test, không train lại để chọn test đẹp hơn. Có thể khảo sát giảm epochs/regularization trong thực nghiệm sau với validation, ngoài kết quả hiện tại.

Điều kiện patience=2 kích hoạt đúng bước cuối: `early_stopping_triggered=true`, **`early_stopped=false`** vì đã chạy đủ lịch. Chi tiết sửa cờ và bản gốc được giữ tại [metadata_corrections.md](../outputs/full/metadata_corrections.md).

## 3. Kiểm tra đã chạy

Tất cả lệnh từ repo gốc; dùng `Lab 3/.venv/Scripts/python.exe`:

| Kiểm tra | Lệnh/thao tác thực | Kết quả |
| --- | --- | --- |
| Unit và integration offline | `python "Lab 3/test.py"` | PASS 10 tests; gồm gradient/update/save-load với tiny Trainer, metric biết trước, threshold/tie/NaN, hash và chống ghi đè |
| Module A pretrained thật | `lab3.py infer --text "Bộ phim này rất hay!" --text "The movie was boring." --output "Lab 3/outputs/module-a"` | PASS; Positive .652007 và Negative .764171 |
| Smoke pretrained | `lab3.py train --mode smoke --output "Lab 3/outputs/smoke"` | PASS 2 steps; loss/gradient hữu hạn, save/reload, report; accuracy không dùng nghiệm thu |
| Full | `lab3.py train --mode full --output "Lab 3/outputs/full"` | PASS thực thi 3 epochs và test độc lập |
| 5 checkpoint | Nạp từng thư mục bằng `load_demo` trên câu “I loved this movie.” | PASS cả 5; 2 nhãn và suy luận POSITIVE |
| Demo best | `lab3.py demo --checkpoint "Lab 3/outputs/full/checkpoints/best"` và ô demo notebook | PASS; nạp từ đĩa |
| Nguồn | `lab3.py audit` sau khi sắp lại bảy bước; manifest full đã kiểm ở lần trước | PASS 19 file, 8 ô code; nguồn Markdown |
| Các run khác | `verify_artifacts` cho smoke/module-a | PASS; smoke 41 artifact |
| Notebook rút gọn | NBClient, kernel `lab3`, `allow_errors=False`; chạy 8 ô code | PASS; Exercise 1 và bảy bước Exercise 2, nạp checkpoint full, đánh giá lại đủ 25.000 test, hai biểu đồ và demo |
| Schema/code notebook | `nbformat.validate`, `ast.parse` mọi ô code | PASS |
| Dependency | `pip check` | PASS; no broken requirements |
| Hình | Mở PNG full loss và confusion matrix để kiểm tra | PASS bố cục; chữ trắng trên ô tối |
| Diff | `git diff --check` | PASS; không sửa Lab 1/Lab 2/Learning |

Mô hình ngẫu nhiên nhỏ trong unit test chỉ kiểm cơ chế, không phải mô hình bàn giao và không góp vào số đo IMDB.

## 4. Lịch sử sửa lỗi có bằng chứng

- Smoke đầu: train/save chạy được, evaluator thiếu `eval_dataset` nên Trainer từ chối cấu hình epoch. Đã sửa và chạy lại smoke thành công; run lỗi giữ ở `outputs/smoke-failed-001`, không dùng nghiệm thu. Bản đầu có source đang được chỉnh trong lúc run nên không dùng nó làm chứng cứ revision code cuối.
- Snapshot Markdown trên Windows: `read_text` chuẩn hóa CRLF, làm hash snapshot khác raw source. Đã chuyển sang byte-preserving decode; test kiểm toàn bộ snapshot/hash đã PASS. Smoke/module-a có ghi chú hiệu chỉnh metadata, không thay số đo/trọng số.
- Cờ dừng ở epoch cuối: tách “điều kiện kích hoạt” và “huấn luyện bị rút ngắn”; thêm regression test cho cả hai tình huống. JSON trước hiệu chỉnh và source đã chạy được giữ trong full run. Không cần train lại vì chỉ sửa cờ báo cáo, không thay gradient hay checkpoint.
- Confusion matrix: đổi chữ ô tối sang trắng, render lại từ đúng số liệu cũ; ghi nhận trong metadata_corrections.md.

## 5. Bằng chứng và tái lập

- [Báo cáo full](../outputs/full/report.md), [metric gốc đo](../outputs/full/metrics.json), [nguồn artifact](../outputs/full/artifacts.md).
- `provenance.json`: SHA-256 từng artifact, snapshot các Markdown căn cứ, source code thực đã train và hash code hiệu chỉnh sau đó. Giữ log gốc và JSON trước sửa cờ.
- `run_config.json`: revision chính xác model/dataset, seed, fingerprint, class counts và hash index từng split. `lab3_source.py` là source thực của run, có thể khác bản hiện tại ở phần cờ dừng/hiển thị đã ghi nhận.
- Notebook đã lưu output thực; khi Run All, nó nạp checkpoint full có sẵn để đánh giá, không kiểm hash trong luồng chính. Luồng huấn luyện mới tạo thư mục run theo thời gian.

## 5.1. Notebook rút gọn

Bản hiện tại có **8 ô code: một ô Exercise 1 độc lập và bảy ô Step 1–7 cho Exercise 2**, chỉ định nghĩa hai hàm nhỏ `tokenize` và `compute_metrics`. Thứ tự bước đối chiếu từ notebook kế thừa ở Git commit `755af75`; chưa xác minh độc lập tác giả của bản đó. Không import `lab3.py`, không có class tùy biến, kiểm tra AST/hash hoặc bộ kiểm thử trong luồng chính. Dùng trực tiếp `Trainer`, `EarlyStoppingCallback` và API thư viện.

Đã chạy lại toàn bộ tám ô sau khi sắp xếp bằng NBClient, không có ô lỗi. Model full đã có được nạp và đánh giá lại đủ 25.000 mẫu test: accuracy **91.012%**, macro F1 **0.9101182914**, test loss **0.3465386629**. Hai biểu đồ confusion matrix/loss và demo có output thực. Biểu đồ loss dùng history full đã có; không coi đây là full training mới. Schema, cú pháp tám ô, audit nguồn và 10 kiểm thử CLI đều PASS sau thay đổi.

Luồng `TRAIN=True` được kiểm chứng riêng bằng bản thực thi tạm trên 32 train / 16 validation / 16 test, 1 epoch = 2 optimizer steps. Đã chạy train, validation, lưu best, nạp lại, đánh giá và vẽ hình thành công. Artifact thử nghiệm nằm trong thư mục tạm riêng và được dọn sau kiểm chứng; không dùng metric nhỏ làm kết quả full.

Các kiểm thử 10 cases, 5 checkpoint và benchmark hệ thống ở phần trên thuộc pipeline CLI đã kiểm chứng trước đó. Notebook rút gọn chỉ giữ checkpoint tốt nhất khi chạy mới; nguồn cho artifact mới nằm trong SOURCES.md của run. Không cần chạy CLI để sử dụng notebook.

## 5.2. Kiểm tra kernel trong VS Code

Đã xác nhận extension Python/Jupyter và `ipykernel` được cài. Bộ chọn kernel đang ở màn hình nhập URL server, trong khi môi trường cục bộ nằm tại `Lab 3\.venv\Scripts\python.exe`. Đã chọn **Python Environments → .venv (3.12.10)** và quan sát kernel khởi động thành công. Lần thực thi ô pretrained trước ô import gây `NameError: pipeline is not defined`; chạy **Run All** sau khi chọn đúng môi trường đã hoàn tất đến ô demo. Test accuracy hiển thị trong VS Code là **0.91012**, demo trả POSITIVE/NEGATIVE đúng với hai câu mẫu. Căn cứ kỹ thuật: [10_sources.md S-11](10_sources.md).

## 6. Giới hạn

Lần sắp xếp docs ngày 2026-11-06: 10 kiểm thử offline PASS; audit nguồn PASS cho 20 file và 8 ô notebook; kiểm hash 45 artifact full PASS. Kiểm tra schema, cú pháp và 95 liên kết nội bộ PASS. Đối chiếu với revision trước xác nhận notebook giữ nguyên output và mã tính toán; CLI chỉ đổi tham chiếu tài liệu. Không chạy thêm full training hoặc smoke mới cho lần đổi tên tài liệu.

Phạm vi nghiệm thu dựa trên yêu cầu dự án và đặc tả hiện có. Nguồn gốc tác giả của notebook kế thừa chưa được xác minh độc lập. Không đo trên Colab T4, không triển khai web, không cung cấp resume optimizer. Module B chỉ huấn luyện tiếng Anh; câu trung tính vẫn buộc phân loại hai lớp. RSS host là đỉnh quan sát, không đo đỉnh tuyệt đối. Các ngưỡng chất lượng/phần cứng được kết luận trong đúng điều kiện đo nêu trên.

## Nguồn và phạm vi

- [08_implementation.md](08_implementation.md), [05_evaluation.md](05_evaluation.md), [04_training.md](04_training.md): yêu cầu và cách đo.
- [11_AI_Usage.md](11_AI_Usage.md), [12_traceability.md](12_traceability.md): vai trò AI và truy nguồn.
- [10_sources.md](10_sources.md): tài liệu API chính thức.
- [report.md của full run](../outputs/full/report.md) và các artifact liên kết: bằng chứng số đo, không phải kết quả giả định.
