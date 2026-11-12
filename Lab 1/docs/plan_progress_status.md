# Project Roadmap & Progress Status — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Project Roadmap & Progress Status — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T10:16:25+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Trạng thái căn cứ vào code và bằng chứng đang có trong repo. “Có code” không đồng nghĩa đã nghiệm thu pipeline. Các mốc chạy trước đây trong changelog được giữ làm lịch sử, không phải kết quả chạy lại ở lần rà này.

**Nguồn training đã đối chiếu ngày 2026-11-12:** Nguyễn Quốc An (Quoc An) thực hiện `src/train.py` và `tests/test_train.py` tại `773a5a1`/`e2b6402`. [Hồ sơ training](training.md) ghi riêng người cập nhật code nguồn và người hoàn thiện/tích hợp. Người chạy smoke/bốn normal run vẫn là HoangDang theo bằng chứng hiện có. Việc bổ sung tên do HoangDang đối chiếu lịch sử; code, số đo và artifact đã chốt giữ nguyên.

## 1. Tiến độ nghiệm thu kỹ thuật

| Task | Trạng thái | Bằng chứng |
| --- | --- | --- |
| Data/loader | DONE | data/reports, data/tests đã chạy thật |
| Model/sidecar/save-load final/best | DONE | model_summary, 20 tests, reload_check mỗi run |
| Smoke end to end | DONE | outputs/smoke_001, verify PASS trước/sau manifest |
| Bốn normal run | DONE | outputs/run_001–run_004, checkpoints/index, 40.000 dự đoán test tổng cộng |
| Evaluation/performance/đồ thị/demo | DONE | compare_runs.csv, compare_loss.png, ảnh mẫu và demo_example.json |
| Verification/truy vết/tài liệu | DONE | verification.md, traceability.md, registry và manifest/hash |

## 1.1. Kết quả hiện tại ngày 2026-11-11

Model chọn bằng validation là **run_004**, val **88.10%**, test **87.70%**. Cả bốn normal run cùng GPU; nguồn số liệu tại [evaluation.md](evaluation.md). Người hoàn thiện/chạy: HoangDang; không thay tác giả các phần kế thừa trong Git.

Đã chạy test bằng Python .venv Lab 1, không dùng kết quả cũ. Các lỗi merge/nguồn gốc bên dưới là lịch sử, không phải trạng thái hiện tại. Sản phẩm chính đã chuyển sang cây repo, input/checkpoint dùng Git LFS, không yêu cầu ZIP hoặc Drive. Tên TV1–TV5/ngày tạo model chưa xác nhận vẫn giữ nguyên, không tự tạo thông tin đó.

## 1.2. Kiểm chứng lịch sử ngày 2026-11-10

Các gạch đầu dòng dưới ghi trạng thái ở lần rà cũ; trạng thái hiện tại nằm ở mục 1.1.

- Đã đọc code data/model/model I/O/train và kiểm thử hiện có; đối chiếu chữ ký hàm, cấu hình, checkpoint và đầu ra.
- Kiểm tra tĩnh đạt: 17 file Markdown không còn marker conflict, 47 liên kết nội bộ tồn tại, 7 file Python parse được, registry JSON hợp lệ và không trùng entry; chỉ số split đã lưu đủ 54.000/6.000, không giao nhau, phủ đủ 60.000 mẫu.
- Đã giải quyết marker merge conflict trong các file Lab 1 liên quan, giữ cấu trúc `data/` và lịch sử review. Git vẫn đang ở trạng thái merge chưa hoàn tất; chưa stage hay commit thay người dùng.
- `python -m pytest "Lab 1/tests/test_train.py" -q`: dừng ở bước thu thập test vì Python hiện tại thiếu `torch` (`ModuleNotFoundError`). Chưa chạy smoke/normal run hoặc tải lại dataset.
- `check_provenance.py --strict`: còn 3 lỗi có sẵn: `src/model.py`, `src/model_io.py` chưa đăng ký nguồn gốc; `outputs/model_summary.txt` thiếu sidecar. Không sửa tay báo cáo thực nghiệm hoặc tạo bằng chứng chạy thay cho lần chạy gốc.
- Hash nguồn trong sổ đăng ký được cập nhật sau khi rà; đây là xác nhận đối chiếu tài liệu/code, không phải xác nhận runtime đã pass. Phần `best`, evaluation và demo vẫn chưa hoàn thành.

---

## 2. Engineering Changelog
- `2026-11-11` — HoangDang: hoàn thiện pipeline, tạo môi trường/dữ liệu, 17 test PASS, smoke + 4 normal run PASS; tổng hợp số liệu thật và đóng gói source/input/output/trọng số cục bộ. Xem manifest từng run và verification.md; không ghi đè run đã chốt.
- `2026-11-11` — HoangDang: chuẩn hóa bảng thông tin và trạng thái; thống nhất tối thiểu ba normal run, vòng đời evaluation/verification và danh sách đầu ra. Giữ lịch sử ngày 2026-11-10; rà registry qua CLI. Không sửa thuật toán hoặc chạy lại thực nghiệm.
- `2026-11-10` — HoangDang: đồng bộ mô tả chính, README, data/model/training/evaluation/demo, yêu cầu 3.1–3.6 và phân công. Thống nhất `data/*.py`, `data/reports/`, ba loader, MLP `[512,512]`, smoke 512/256/256, tối đa 10 epoch, early stopping 2 epoch/0.01, 5 checkpoint và đường dẫn theo run. Gỡ marker conflict của Lab 1 để tài liệu và công cụ lưu vết đọc được; giữ lịch sử registry. Ghi rõ model I/O mới kiểm tra `final_epoch`, thiếu phần `best`/evaluation/demo, 3 lỗi provenance và giới hạn môi trường thiếu torch. Chờ nhóm review.
- `2026-11-10` — Theo yêu cầu người dùng, gom module 3.1–3.2 vào `data/` (code, `reports/`, `tests/`, dataset gốc). Cập nhật import trong train, đường dẫn báo cáo, Git ignore và công cụ lưu vết; giữ lịch sử nguồn gốc khi chuyển file. Tài liệu chung tiếp tục ở `docs/`.
- `2026-11-10` — HoangDang: triển khai 3.1–3.2, đồng bộ giao diện ba loader theo đặc tả chung. Tải FashionMNIST thật; split 54.000/6.000/10.000, giao train/val bằng 0; batch [64,1,28,28], float32 [0,1]. `python -m pytest "Lab 1/tests" "Lab 1/data/tests" -q`: 8 passed (gồm kiểm tra dữ liệu và train hiện có). Chưa chạy full training. Kiểm tra provenance toàn Lab còn lỗi có sẵn ở model.py, model_io.py và model_summary.txt; phần data đã đăng ký. Chờ review của thành viên nhóm.
- `2026-11-04`: Hoàn thiện khung tài liệu đặc tả hệ thống Module 01 theo chuẩn MLOps.
