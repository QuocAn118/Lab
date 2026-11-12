# Kiểm chứng Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Created date | 2026-11-11 |
| Created by | HoangDang, hỗ trợ OpenAI Codex |
| Last updated | 2026-11-11T10:16:25+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Trạng thái | Đạt nghiệm thu kỹ thuật trên môi trường đã đo |

## Kết luận nghiệm thu kỹ thuật

Đã hoàn thành phạm vi Lab 1 trên môi trường được ghi dưới đây: 17 test, smoke thật và bốn normal run đạt. Đã kiểm lại dữ liệu/số liệu/manifest; các run đã chốt không được bổ sung hoặc sửa artifact.

| Giai đoạn | Lệnh/kiểm tra | Kết quả thực tế |
| --- | --- | --- |
| Môi trường | Tensor CUDA, pip check | GPU RTX 5060 Laptop hoạt động; không có dependency hỏng |
| Data | prepare_data, data_pipeline --check, data/tests | 54.000/6.000/10.000; seed 42; train ∩ val = 0; batch [64,1,28,28] |
| Unit/integration | `.venv/Scripts/python.exe -m pytest tests data/tests -q --basetemp <unique-repo-temp>` | **17 passed in 10.61s** |
| Model | model.py --check sau commit code | 669.706 tham số, output [2,10], logits hữu hạn, sidecar/hash đúng |
| Smoke | train.py --run-id smoke_001, rồi verify.py --run-id smoke_001 | Một epoch, 512/256/256, completed, PASS trước và sau manifest |
| Normal | Bốn lệnh theo training.md, verifier chỉ đọc sau từng run | Tất cả completed/PASS, mỗi run 10.000 dòng dự đoán |
| So sánh | compare_runs.py --run-ids run_001 run_002 run_003 run_004 | CSV/đồ thị và sidecar đã tạo từ nguồn thật |
| Demo | demo.py --run-id run_004 --index 0 | Nhãn thật/dự đoán đều Ankle boot; 10 logits hữu hạn |
| Nguồn gốc | tools/check_provenance.py --strict sau cập nhật registry | PASS; kết quả cuối được kiểm lại trước bàn giao |

Không dùng kết quả "8 passed" cũ làm kết quả mới. Lượt test ban đầu bị PermissionError tại thư mục temp Windows; lượt thành công dùng thư mục mới trong .git. README hướng dẫn temp dưới .venv để tránh lỗi môi trường đó.

## Người chạy, thời gian và commit

| Run | Bắt đầu (UTC+07) | Người chạy | Commit nguồn | Test samples | Verify |
| --- | --- | --- | --- | --- | --- |
| run_001 | 2026-11-11T09:41:02+07:00 | HoangDang | 529127e | 10.000 | PASS |
| run_002 | 2026-11-11T09:42:46+07:00 | HoangDang | 7716ca7 | 10.000 | PASS |
| run_003 | 2026-11-11T09:44:15+07:00 | HoangDang | 421caf6 | 10.000 | PASS |
| run_004 | 2026-11-11T09:45:49+07:00 | HoangDang | 41e13a1 | 10.000 | PASS |

Môi trường: Python 3.12.10, torch 2.9.1+cu128, torchvision 0.24.1+cu128, matplotlib 3.10.9, pytest 9.1.0. Xem `requirements-lock.txt` và environment trong manifest. [Nguồn cặp phiên bản PyTorch](https://pytorch.org/get-started/previous-versions/).

## Phép kiểm toàn vẹn

- Trước manifest: history/epoch/giá trị hữu hạn, CSV và thứ tự mẫu, tên lớp, số đúng/accuracy, per-class accuracy, performance, năm checkpoint/index, reload best, định dạng PNG.
- Sau manifest: trạng thái completed, verify_report ban đầu PASS, không thiếu/thừa file và hash/kích thước mọi artifact khớp. CLI không ghi thêm file vào run.
- Test hồi quy sửa CSV phải bị phát hiện, best khác final phải nạp đúng, không ghi đè checkpoint, từ chối loader rỗng/logits NaN và epoch sai; bảng so sánh phải giữ đúng giá trị từng run.
- `.gitattributes` giữ nguyên byte các artifact generated để clone không đổi line ending/kích thước trong manifest. Dataset/trọng số được lưu bằng Git LFS: Git lưu pointer, LFS khôi phục byte gốc tại đúng đường dẫn.

## Sản phẩm và giới hạn bàn giao

Sản phẩm chính là cây repo: source/requirements/tests/docs, raw dataset qua LFS, data reports, model summary, smoke và bốn normal run, 25 checkpoint LFS cùng index, CSV/đồ thị so sánh, ảnh input và JSON demo. Không cần giải nén ZIP; ZIP cũ là snapshot dự phòng của lần bàn giao trước.

Mỗi cấu hình đo một lần, cùng seed/máy; thời gian không phải benchmark nhiều lần, ms/ảnh là trung bình theo batch. Chưa thử full training trên máy khác. Không dùng Drive làm điều kiện lấy sản phẩm: dataset và checkpoint được version bằng Git LFS. Sau clone/pull cần tải LFS trước khi chạy demo/verifier. Tên đầy đủ TV1–TV5/ngày tạo model chưa được chủ thể xác nhận vẫn ghi chưa xác nhận; không tự gán tác giả từ tên nhánh.

## Nguồn và phạm vi

Kiểm tra lại ngày 2026-11-12 bởi HoangDang, hỗ trợ OpenAI Codex, trên bản repo sau tích hợp `c6f4d54`. Bộ test đạt 17/17 trong 22,77 giây. Verifier chỉ đọc xác nhận `smoke_001` và `run_001`–`run_004` đều PASS, kiểm tra provenance strict có 0 lỗi và 0 cảnh báo. Lượt kiểm tra dùng thư mục tạm mới trong `.git/completion-checks` sau khi tạo thư mục cha. Báo cáo và checkpoint của các run tiếp tục là bằng chứng của ngày chạy ghi trong manifest.

- [Đặc tả](lab_1_description.md), [đánh giá](evaluation.md), [truy vết](traceability.md), artifact thật trong outputs/checkpoints. HoangDang thực hiện với hỗ trợ OpenAI Codex ngày 2026-11-11.
