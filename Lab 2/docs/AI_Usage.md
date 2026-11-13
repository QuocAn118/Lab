# Khai báo công cụ AI — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Khai báo công cụ AI — Lab 2 |
| Created date | 2026-11-12 |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Last updated | 2026-11-12 |
| Phạm vi | Nơi duy nhất ghi công cụ AI đã dùng trong Lab 2; tài liệu khác chỉ ghi người thực hiện |

## 1. Tổng hợp

| Ngày | Công cụ | Người dùng / tích hợp | Phần hỗ trợ |
| --- | --- | --- | --- |
| 2026-11-10 | Claude Code (Claude Opus 5.5) | Nguyễn Quốc An | Đặc tả, test, evaluation TV4 (chi tiết mục 2) |
| 2026-11-11 | OpenAI Codex | Nguyễn Thanh Bình | Phiên TV3 bàn giao `92930e6`, theo khai báo của thành viên |
| 2026-11-11 – 2026-11-12 | OpenAI Codex | HoangDang | Package common/data/training/demo, ghép model và evaluation, notebook, kiểm tra bàn giao, phiên tái lập, tài liệu |
| 2026-11-12 | Claude Code (Claude Opus 5.5) | HoangDang | Kiểm lại, cập nhật số test, đánh dấu đặc tả cũ, chuyển khai báo công cụ về file này |

Byline công cụ trong các tài liệu khác của Lab 2 đã được bỏ ngày 2026-11-12; lịch sử Git vẫn giữ bản trước đó.

## 2. Khai báo chuyển từ plan_progress_status.md

### Việc dùng AI (mục 5 cũ)

| Công cụ | Mục đích | AI hỗ trợ | Đóng góp của sinh viên |
| :--- | :--- | :--- | :--- |
| Claude Code (Claude Opus 5.5) | Rà soát tài liệu, viết đặc tả và test | Đối chiếu đề bài với đặc tả và notebook; cập nhật `docs/`; viết test; viết một bản code đầy đủ để kiểm tra test và đặc tả, lưu riêng ở nhánh `lab2-ban-day-du` trên máy người điều phối, không có trong nhánh `main` và `QuocAn` | Làm notebook ban đầu; cung cấp đề bài; chọn model, số lần chạy, số epoch, số ảnh, cách chia việc; duyệt các điểm sửa. Phần code của từng thành viên: chưa thực hiện. Nếu dùng hoặc tham khảo nhánh trên, phải ghi rõ tại đây. TV4: `evaluation.py` lấy từ nhánh trên, đã đối chiếu từng hàm với `evaluation.md` mục 8, chạy `tests/test_evaluation.py` (6 passed, dùng tạm các file khác của nhánh trên) và xem lại 3 biểu đồ |


Phần khai báo AI ở bảng trên là thông tin kế thừa do thành viên ghi. Lần chuẩn hóa ngày 2026-11-11 dùng OpenAI Codex để rà tài liệu, không sinh thêm code model/training hoặc số đo mới.

### AI và nguồn của đợt chuẩn bị này

OpenAI Codex hỗ trợ HoangDang triển khai common/data/training/demo, ghép model của Xuân Bảo và evaluation của Quốc An, viết kiểm tra bàn giao, notebook và cập nhật tài liệu. Không truy cập hay lấy code từ nhánh `lab2-ban-day-du` (không có trên origin). Lịch sử khai báo Claude phía trên được giữ nguyên như nguồn kế thừa, không phải công cụ của đợt này. Người train và kết quả thật chưa xác nhận.

## Nguồn và phạm vi

- Chuyển nguyên văn từ `plan_progress_status.md` mục 5 và nhật ký các tài liệu Lab 2 trước ngày 2026-11-12.
