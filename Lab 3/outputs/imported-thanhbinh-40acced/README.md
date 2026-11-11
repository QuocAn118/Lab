# Snapshot lịch sử từ nhánh local ThanhBinh

Nhập ngày 2026-11-11 bởi HoangDang, hỗ trợ OpenAI Codex. Thời gian chính xác và danh sách 176 file nằm trong [IMPORT.json](IMPORT.json).

- Commit gốc: `40accede964160fdafc0bff30741ae6e7b979058`.
- Tác giả và committer gốc: **HoangDang**, thời gian `2026-11-07T07:33:28+07:00`. Tên nhánh không phải bằng chứng tác giả là Thanh Bình.
- Commit chuyển trọng số sang LFS: `ecb71b108d46cb0d96e028075dcde8b53ffa8bd5`; giữ nguyên tác giả, thời gian gốc. Bản raw được giữ ở `backup/ThanhBinh-raw-40acced`.
- 156 file không phải trọng số giữ nguyên blob Git. 20 trọng số chuyển thành pointer LFS; SHA-256 trong pointer xác định nội dung nhị phân gốc. Có 5 trọng số khác nhau về nội dung.
- Các thư mục `full`, `smoke`, `smoke-failed-001` và snapshot khác là bằng chứng kế thừa, không phải kết quả chạy ngày nhập. Tên đường dẫn và hash trong manifest cũ được giữ nguyên; dùng `IMPORT.json` để ánh xạ sang vị trí mới.
- Không thay thế kết quả đang có trong `outputs/` hoặc `report/`. Các artifact lịch sử không được sửa tay để làm khớp trạng thái hiện tại.

## Nguồn và phạm vi

- [Quy tắc truy nguồn](../../docs/11_AI_Usage.md), [thiết kế triển khai](../../docs/08_implementation.md).
- Git commit gốc và commit LFS nêu trên; [nhật ký đồng bộ](../../../BRANCH_SYNC_2026-11-11.md).
