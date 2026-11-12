# Rà soát và hoàn thiện ba lab — 2026-11-12

HoangDang thực hiện theo yêu cầu hoàn thiện Lab 1, Lab 2 và Lab 3. Tác giả nguồn và các lần tích hợp trước được đối chiếu trong Git, registry Lab 1 và hồ sơ nhập từng phiên.

| Lab | Kiểm chứng | Kết quả bài thực hành | Phần đã hoàn thiện trong lượt này |
| --- | --- | --- | --- |
| Lab 1 | 17 test, strict provenance, verifier của smoke và bốn normal run đều đạt | Model chọn bằng validation có test accuracy 87,70% trên 10.000 ảnh theo run đã lưu | Sửa trạng thái tài liệu và mô tả khả năng nạp checkpoint best, ghi lần rà trong registry |
| Lab 2 | 85 test và ba smoke thật của code hiện hành, bốn cấu hình đã train và đánh giá trên RTX 5060 | DenseNet121 test 87,68%, macro F1 0,8755 trên 5.000 ảnh | Chụp giao diện TensorBoard mới, kiểm nguồn event/hash và cập nhật liên kết bàn giao |
| Lab 3 | 43 test, source audit, smoke pretrained mới và Run All tám cell đạt | Checkpoint full đã có được đánh giá lại trên 25.000 văn bản test, accuracy 91,012%, macro F1 0,910118 | Sửa chọn checkpoint sau nhập nhánh, nối tiện ích thời gian/report, đọc loss history cùng checkpoint, sửa nguồn và dependency, thêm chặn ghi đè trong bản Quốc An |

## Lab 1

Kiểm thử lại đạt 17/17 trong 22,77 giây. `smoke_001`, `run_001`, `run_002`, `run_003`, `run_004` đều được verifier chỉ đọc xác nhận PASS. `check_provenance.py --strict` có 0 lỗi và 0 cảnh báo. Các số đo, checkpoint và thời điểm train nằm trong manifest ngày chạy tương ứng.

[Kiểm chứng Lab 1](Lab%201/docs/verification.md), [đánh giá bốn cấu hình](Lab%201/docs/evaluation.md), [registry nguồn](Lab%201/docs/provenance.json).

## Lab 2

Đã lưu ảnh giao diện TensorBoard với đủ bốn run, Accuracy/val, Loss/train và Loss/val. Smoothing 0, hiển thị đầy đủ các điểm dữ liệu. Bốn file event khớp hash manifest của phiên `rerun-thanhbinh-20261112-175308`.

[Ảnh TensorBoard](Lab%202/reports/sessions/tensorboard-review-20261112-191218/tensorboard.png), [biên bản bổ sung](Lab%202/reports/sessions/tensorboard-review-20261112-191218/REVIEW.json), [so sánh bốn cấu hình](Lab%202/reports/sessions/rerun-thanhbinh-20261112-175308/comparison.md).

Phiên `colab-t4-20261112-1920` đã được cấp Tesla T4 và bắt đầu chạy notebook từ code `c6f4d54`. Kết quả đầy đủ, notebook có output và checkpoint của phiên này còn chờ tiếp nhận. Hai KPI tốc độ T4 tiếp tục chờ xác minh trên artifact thực tế của phiên T4.

## Lab 3

Notebook dùng câu mẫu có thể sửa trong Exercise 1. Exercise 2 mặc định `TRAIN=False`, tìm checkpoint có trọng số và phát hiện pointer LFS. Model/tokenizer cùng được nạp từ thư mục checkpoint đã chọn. Các tiện ích report và thời gian từ `eb6bfad` được nối lại. Loss history được đọc theo đường dẫn checkpoint nhập.

Bộ kiểm thử đạt 43/43 trong 11,01 giây. Notebook tám cell hoàn tất lúc 2026-11-12T20:43:18+07:00. Đánh giá lại 25.000 test đạt accuracy 0,91012, macro F1 0,9101182914 và loss 0,3465386629. Smoke mới dùng hai optimizer step trên 32/16/16 mẫu, xác minh đủ năm checkpoint và 48 artifact. Source audit đạt 40 file và tám cell.

[Báo cáo đánh giá mới](Lab%203/report/evaluate-20261112-204226-805293/summary.md), [suy luận đa ngôn ngữ](Lab%203/report/infer-20261112-204208-788129/summary.md), [smoke mới](Lab%203/report/cli-smoke-20261112-204427-088939/summary.md), [JUnit](Lab%203/report/completion-review-20261112/offline-tests.xml).

Đã đối chiếu đủ 176 file của snapshot nhập, gồm 20 trọng số, với `IMPORT.json`. Manifest full lịch sử có 18 file khớp byte trực tiếp và 27 file text khớp sau khôi phục kiểu xuống dòng khi tạo manifest. Verifier byte nghiêm ngặt của bản lịch sử báo khác hash tại 27 file đó. Biên bản kiểm tra mới ghi từng phép đối chiếu và hash của bản đang lưu.

[Đối chiếu bản nhập](Lab%203/report/completion-review-20261112/intake_check.json), [đối chiếu manifest lịch sử](Lab%203/report/completion-review-20261112/historical_manifest_check.json).

## Giới hạn kiểm chứng

Lượt này đánh giá lại model full Lab 3 đã có và chạy smoke mới. Chưa chạy full training Lab 3 mới hoặc benchmark Lab 3 trên T4. Số đo GPU RTX được đọc theo môi trường đã lưu. Kết quả T4 của Lab 2 cần lấy từ phiên Colab trước khi kết luận về tốc độ hoặc Run All trên Colab.

## Kiểm lại sau commit `25cc678`

HoangDang, chạy lại lúc 2026-11-12T21:30+07:00. Commit `25cc678` thêm test chặn ghi đè nên số test tăng: Lab 1 **20/20**, Lab 2 **93/93** offline, Lab 3 **44/44** (cần `PYTHONPATH=src`). Strict provenance Lab 1, verifier năm run Lab 1, verifier bàn giao Lab 2 và source audit Lab 3 đều PASS. Các số liệu ở bảng trên là của lượt rà trước, giữ nguyên. Chi tiết và môi trường: [Lab 1](Lab%201/docs/verification.md), [Lab 2](Lab%202/docs/verification.md), [Lab 3](Lab%203/docs/06_verification.md). Kết quả Colab T4 của Lab 2 vẫn chờ.

## Nguồn và phạm vi

- Đặc tả, tiến độ, kiểm chứng và artifact được liên kết cho từng lab.
- [Quy tắc repo](AGENTS.md), [quy tắc Lab 3](Lab%203/AGENTS.md), [truy nguồn Lab 3](Lab%203/docs/12_traceability.md).
- Các sửa đổi và kiểm chứng ngày 2026-11-12 do HoangDang thực hiện. Hồ sơ nguồn của Vân Anh, Thanh Bình, Quốc An, Xuân Bảo và các khai báo AI đã có tiếp tục nằm trong tài liệu và lịch sử tương ứng.
