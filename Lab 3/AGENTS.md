# Nguyên tắc làm việc trong Lab 3

1. Đọc `docs/AI_Usage.md`, `docs/implementation.md` và đặc tả liên quan trước khi sửa.
2. Viết/chốt yêu cầu trong Markdown trước code. Mỗi file AI tạo/sửa phải khai nguồn `.md`; không bịa nguồn hoặc trình bày quyết định triển khai như yêu cầu gốc khi chưa có căn cứ.
3. Code/config dạng text dùng header `Sources:`. Markdown có mục `Nguồn và phạm vi`. Notebook có nguồn từng ô code và metadata. Binary/native artifact dùng manifest theo `docs/AI_Usage.md`.
4. Cập nhật `docs/traceability.md` khi thêm yêu cầu hoặc file. Không gọi smoke/quick là full; không điền số đo chưa chạy.
5. Không sửa source khi run đang chạy; manifest lưu snapshot source để kiểm chứng. Giữ thay đổi người dùng, không ghi đè output.
6. Kiểm chứng bằng `test.py`, `lab3.py audit`, smoke và kiểm tra phù hợp; báo trung thực giới hạn. Full dùng test độc lập, không tune theo test.
7. Sử dụng văn phong kỹ thuật trung lập. Tập trung vào yêu cầu, thiết kế, nguồn và bằng chứng; tránh cách xưng hô cá nhân hoặc diễn giải bối cảnh trao đổi trong tài liệu bàn giao.

## Nguồn và phạm vi

- [docs/AI_Usage.md](docs/AI_Usage.md): yêu cầu dự án và quy tắc truy nguồn.
- [docs/implementation.md](docs/implementation.md), [docs/verification.md](docs/verification.md): hợp đồng và kiểm chứng.
- Áp dụng thư mục Lab 3 và phần notebook Lab 3 khi chỉnh sửa liên quan.
