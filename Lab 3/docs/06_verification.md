# Kiểm chứng thực tế — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Kiểm chứng thực tế — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-06T21:05:06+07:00 ; HoangDang ; d108ed4 |
| Last updated | 2026-11-12T21:38:06+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

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

Điều kiện patience=2 kích hoạt đúng bước cuối: `early_stopping_triggered=true`, **`early_stopped=false`** vì đã chạy đủ lịch. Chi tiết sửa cờ và bản gốc được giữ tại `metadata_corrections.md` (không còn trong bản sao hiện tại).

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

- Báo cáo full `outputs/full/report.md` (không còn trong bản sao hiện tại), [metric gốc đo](../outputs/imported-thanhbinh-40acced/full/metrics.json), [nguồn artifact](../outputs/full/artifacts.md).
- `provenance.json`: SHA-256 từng artifact, snapshot các Markdown căn cứ, source code thực đã train và hash code hiệu chỉnh sau đó. Giữ log gốc và JSON trước sửa cờ.
- `run_config.json`: revision chính xác model/dataset, seed, fingerprint, class counts và hash index từng split. `lab3_source.py` là source thực của run, có thể khác bản hiện tại ở phần cờ dừng/hiển thị đã ghi nhận.
- Notebook đã lưu output thực; khi Run All, nó nạp checkpoint full có sẵn để đánh giá, không kiểm hash trong luồng chính. Luồng huấn luyện mới tạo thư mục run theo thời gian.

## 5.1. Notebook rút gọn

Bản hiện tại có **8 ô code: một ô Exercise 1 độc lập và bảy ô Step 1–7 cho Exercise 2**, giữ các hàm xử lý ngắn và bổ sung callback `TrainingTime` chỉ để ghi thời gian. Thứ tự bước đối chiếu từ notebook kế thừa ở Git commit `755af75`; chưa xác minh độc lập tác giả của bản đó. Không import `lab3.py`, không có kiểm tra AST/hash hoặc bộ kiểm thử trong luồng chính. Callback mới không thay cách train/chọn model. Dùng trực tiếp `Trainer`, `EarlyStoppingCallback` và API thư viện.

Đã chạy lại toàn bộ tám ô sau khi sắp xếp bằng NBClient, không có ô lỗi. Model full đã có được nạp và đánh giá lại đủ 25.000 mẫu test: accuracy **91.012%**, macro F1 **0.9101182914**, test loss **0.3465386629**. Hai biểu đồ confusion matrix/loss và demo có output thực. Biểu đồ loss dùng history full đã có; không coi đây là full training mới. Schema, cú pháp tám ô, audit nguồn và 10 kiểm thử CLI đều PASS sau thay đổi.

Luồng `TRAIN=True` được kiểm chứng riêng bằng bản thực thi tạm trên 32 train / 16 validation / 16 test, 1 epoch = 2 optimizer steps. Đã chạy train, validation, lưu best, nạp lại, đánh giá và vẽ hình thành công. Artifact thử nghiệm nằm trong thư mục tạm riêng và được dọn sau kiểm chứng; không dùng metric nhỏ làm kết quả full.

Các kiểm thử 10 cases, 5 checkpoint và benchmark hệ thống ở phần trên thuộc pipeline CLI đã kiểm chứng trước đó. Notebook rút gọn chỉ giữ checkpoint tốt nhất khi chạy mới; nguồn cho artifact mới nằm trong SOURCES.md của run. Không cần chạy CLI để sử dụng notebook.

## 5.2. Kiểm tra kernel trong VS Code

Đã xác nhận extension Python/Jupyter và `ipykernel` được cài. Bộ chọn kernel đang ở màn hình nhập URL server, trong khi môi trường cục bộ nằm tại `Lab 3\.venv\Scripts\python.exe`. Đã chọn **Python Environments → .venv (3.12.10)** và quan sát kernel khởi động thành công. Lần thực thi ô pretrained trước ô import gây `NameError: pipeline is not defined`; chạy **Run All** sau khi chọn đúng môi trường đã hoàn tất đến ô demo. Test accuracy hiển thị trong VS Code là **0.91012**, demo trả POSITIVE/NEGATIVE đúng với hai câu mẫu. Căn cứ kỹ thuật: [10_sources.md S-11](10_sources.md).

## 5.3. Description và thời gian train — 2026-11-07

Description được viết lại bằng các việc thực hiện trong hai exercise, tách cách lưu checkpoint của notebook khỏi phần CLI. Giữ khai báo hỗ trợ AI tại 11_AI_Usage.md.

- 10 kiểm thử CLI PASS, có kiểm tra thời gian đi kèm checkpoint và giới hạn thời gian train trước lúc xuất model cuối.
- Smoke CLI pretrained PASS với 2 optimizer steps; file tổng kết và cả 5 checkpoint có training_time.json. Manifest của run kiểm thử xác minh 47 artifact PASS.
- Luồng notebook được kiểm tra trên model nhỏ: hai lượt train liên tiếp có thư mục riêng, mỗi lượt ghi 3 epoch và lưu đủ thời gian vào run/best/checkpoint. Khi chủ động gây lỗi giữa lượt train, file tổng kết vẫn tồn tại với status interrupted_or_failed.
- Notebook chạy lại đủ 8 ô trên checkpoint full hiện có; test accuracy .91012, macro F1 .9101182914. Không full train lại. Code không bị lỗi khi lịch sử loss cũ không còn; confusion matrix dùng labels=[0,1] để xử lý cả tập kiểm thử chỉ có một lớp.
- Full run cũ được bổ sung training_time.json từ số đã ghi trong tài liệu này: tổng 334.46 giây, các epoch 113.11 / 111.24 / 109.95 giây. Status recovered_from_documentation; started_at/ended_at/saved_at chưa có bằng chứng nên null. Xem [nguồn khôi phục](../outputs/full/timing_sources.md).
- Bản sao hiện tại không có metrics.json, epoch_history.json, trainer_history.json, report.md và provenance.json của full run cũ. best_val_pause hiện chỉ có config.json. Các dòng PASS lịch sử bên trên mô tả lần kiểm chứng trước, không xác nhận các tệp đó còn đầy đủ ở bản sao hiện tại.

## 5.4. Rút gọn phần trình bày notebook

Notebook giảm từ 171 xuống 98 dòng code, vẫn giữ 8 ô (Exercise 1 và 7 bước Exercise 2). Ô Step 6 giảm từ 72 xuống 19 dòng. Phần ghi thời gian, train/lưu checkpoint và đọc/vẽ loss nằm trong `notebook_helpers.py`; dữ liệu, model, metrics và cấu hình Trainer vẫn ở notebook.

Đã kiểm chứng hai lượt train nhỏ tách biệt có đủ log epoch/checkpoint/best, nạp lại model thành công và không trùng thư mục run. Khi gây lỗi giữa lượt train, tổng kết vẫn được ghi và callback thời gian được gỡ. Kiểm tra đọc loss có/không có log đều đạt. Notebook đã chạy lại đủ 8 ô, test accuracy .91012 và macro F1 .9101182914; không full train lại hoặc thay trọng số model full.

## 6. Giới hạn

Lần sắp xếp docs ngày 2026-11-06: 10 kiểm thử offline PASS; audit nguồn PASS cho 20 file và 8 ô notebook; kiểm hash 45 artifact full PASS. Kiểm tra schema, cú pháp và 95 liên kết nội bộ PASS. Đối chiếu với revision trước xác nhận notebook giữ nguyên output và mã tính toán; CLI chỉ đổi tham chiếu tài liệu. Không chạy thêm full training hoặc smoke mới cho lần đổi tên tài liệu.

Phạm vi nghiệm thu dựa trên yêu cầu dự án và đặc tả hiện có. Nguồn gốc tác giả của notebook kế thừa chưa được xác minh độc lập. Không đo trên Colab T4, không triển khai web, không cung cấp resume optimizer. Module B chỉ huấn luyện tiếng Anh; câu trung tính vẫn buộc phân loại hai lớp. RSS host là đỉnh quan sát, không đo đỉnh tuyệt đối. Các ngưỡng chất lượng/phần cứng được kết luận trong đúng điều kiện đo nêu trên.

## 7. Kiểm chứng bản hiện tại ngày 2026-11-12

HoangDang, thực thi trên Windows/RTX 5060 Laptop, Python 3.12.10, torch 2.9.1+cu128, transformers 4.57.1, datasets 4.3.0, pandas 2.3.3 và pyarrow 21.0.0. Sau điều chỉnh dependency, `pip check` đạt.

- `pytest test.py test_reporting.py tests/test_exercise_2.py` đạt 43 test trong 11,01 giây. Có bốn cảnh báo về mtime checkpoint và backend ảnh trong test. [JUnit](../report/completion-review-20261112/offline-tests.xml).
- Notebook tám cell Run All bằng NBClient đạt, hoàn tất lúc 2026-11-12T20:43:18+07:00. Đánh giá đủ 25.000 test, accuracy 91,012%, macro F1 0,9101182914 và loss 0,3465386629. Exercise 1, hai biểu đồ và demo đều có output mới. [Report đánh giá](../report/evaluate-20261112-204226-805293/summary.md), [report suy luận](../report/infer-20261112-204208-788129/summary.md).
- Smoke CLI pretrained mới dùng 32/16/16 mẫu, hai optimizer step, gradient hữu hạn và save/reload đủ năm checkpoint. Verifier xác nhận 48 artifact khớp hash. [Report smoke](../report/cli-smoke-20261112-204427-088939/summary.md).
- Source audit đạt 40 file và tám cell. Báo cáo đánh giá mới đạt kiểm hash 28 file. Snapshot nhập khớp đủ 176 file, gồm 20 trọng số, với `IMPORT.json`. [Biên bản kiểm tra nguồn](../report/completion-review-20261112/intake_check.json).

Manifest full lịch sử có 18 file khớp byte trực tiếp. 27 file text còn lại khớp hash và kích thước khi khôi phục kiểu xuống dòng đã ghi lúc tạo manifest, gồm CSV có dòng chú thích LF và các dòng dữ liệu CRLF. Verifier byte nghiêm ngặt của manifest lịch sử vì thế báo khác hash trên bản nhập. Biên bản mới ghi từng đường dẫn và phép đối chiếu. [Đối chiếu manifest lịch sử](../report/completion-review-20261112/historical_manifest_check.json).

Phép đánh giá ngày 2026-11-12 dùng trọng số full đã có. Smoke là kiểm tra pipeline rút gọn. Lượt này chưa chạy full training mới hoặc đo benchmark Lab 3 trên T4.

## 7.1. Kiểm lại sau commit `25cc678` — 2026-11-12

HoangDang, dùng `.venv` Lab 3. Lệnh test cần đặt `PYTHONPATH=src` vì `tests/` nhập package `exercise_2` trong `src/`; thiếu biến này pytest dừng ở bước thu thập. Từ thư mục `Lab 3`:

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = 'src'
& .venv/Scripts/python.exe -m pytest test.py test_reporting.py tests/test_exercise_2.py -q
```

- Kết quả: **44 passed**, 4 cảnh báo, 8,09 giây (commit `25cc678` thêm một test chặn ghi đè). `tests/test_smoke.py` chạy riêng cũng đạt.
- `lab3.py audit` PASS, 40 file và tám cell. Notebook chính `Lab_3 (1).ipynb` ở repo gốc có 8/8 cell đã chạy, không có output lỗi; `notebooks/Lab_3 (1).ipynb` là bản nhánh Quốc An chưa chạy, không phải bản nộp chính.
- Không chạy lại notebook, smoke CLI hay đánh giá; số đo accuracy giữ theo mục 7.

## Nguồn và phạm vi

- [08_implementation.md](08_implementation.md), [05_evaluation.md](05_evaluation.md), [04_training.md](04_training.md): yêu cầu và cách đo.
- [11_AI_Usage.md](11_AI_Usage.md), [12_traceability.md](12_traceability.md): vai trò AI và truy nguồn.
- [10_sources.md](10_sources.md): tài liệu API chính thức.
- Báo cáo full gốc `outputs/full/report.md` (không còn trong bản sao hiện tại) và các artifact liên kết: bằng chứng số đo, không phải kết quả giả định.
