# Quy tắc dùng AI và truy nguồn — Lab 3

## 1. Nguồn gốc quy tắc

Yêu cầu dự án được ghi nhận ngày 2026-11-06: hoàn thiện Lab 3 và bảo đảm mọi file có hỗ trợ AI đều truy được về đặc tả Markdown. Phạm vi triển khai dựa trên các tài liệu hiện có trong `docs/`. Nguồn gốc tác giả của các tài liệu kế thừa chưa được xác minh độc lập.

## 2. Quy tắc bắt buộc

1. Chốt yêu cầu trong Markdown trước khi sinh code. Không lấy code AI làm chứng cứ rằng yêu cầu đã đạt.
2. File code, requirements và ignore có dòng `Sources:` trỏ đến `.md` tồn tại. Notebook có nguồn ở ô đầu, từng ô code và metadata.
3. Tài liệu Markdown có mục **Nguồn và phạm vi**. Phân biệt đặc tả kế thừa, quyết định triển khai bổ sung và tài liệu chính thức tại `10_sources.md`; không trình bày lựa chọn bổ sung như yêu cầu gốc khi chưa có căn cứ.
4. `12_traceability.md` nối mã yêu cầu → mục đặc tả → file/hàm → phép kiểm tra. Không dùng một dòng nguồn chung để che các chức năng ngoài đặc tả.
5. File thực nghiệm phải có nguồn Markdown đi kèm. Pipeline đầy đủ dùng `artifacts.md` và `provenance.json` để ghi đường dẫn, SHA-256, nguồn, lệnh và cấu hình. Notebook rút gọn dùng `SOURCES.md` tại thư mục run để khai phạm vi và nguồn cho toàn bộ artifact thư viện lưu bên dưới. Không chèn comment làm hỏng định dạng model. Các manifest của run đã có vẫn được giữ làm bằng chứng lịch sử.
6. Chỉ ghi số đo thu được từ lần chạy tương ứng; tách smoke/quick/full. Không dùng số mẫu, output notebook cũ hay metric train làm kết quả test.
7. Các thư viện cài trong `.venv` và cache tải từ bên thứ ba không phải file AI sáng tác; không sửa nội dung của chúng. Chúng bị loại khỏi gói bàn giao, phiên bản được ghi lại để tái lập.
8. Không cam kết đạt Accuracy/F1 hoặc tốc độ trước khi đo. Báo cả tiêu chí không đạt và giới hạn phần cứng.
9. Sử dụng văn phong kỹ thuật trung lập trong tài liệu bàn giao: mô tả phạm vi, phương pháp, nguồn và kết quả; tránh cách xưng hô cá nhân và diễn giải bối cảnh trao đổi.
10. Report mới ở `report/<run>/` dùng `SOURCES.md`, `source_snapshot/` và `manifest.json` cho nguồn/hash từng file. Giữ nguyên byte log được nhập từ output cũ; ghi nguồn gốc và phần thiếu. Không tạo log hoặc thời điểm thực nghiệm chưa có bằng chứng. `reporting.py` và tích hợp report được bổ sung với hỗ trợ OpenAI Codex ngày 2026-11-07; xem `08_implementation.md` §9.

## 3. Phạm vi đóng góp được ghi nhận

| Thành phần / vai trò | Công việc | Phạm vi xác nhận |
| --- | --- | --- |
| Chủ dự án | Xác lập phạm vi hoàn thiện và yêu cầu truy nguồn qua Markdown | Yêu cầu dự án ghi nhận ngày 2026-11-06 |
| HoangDang | Người tạo và người cập nhật model fine-tuned IMDB | Xác nhận tên cho cả hai vai trò ngày 2026-11-07; không áp dụng cho pretrained model bên ngoài |
| OpenAI Codex | Đọc đặc tả, đối chiếu tài liệu chính thức, bổ sung code, notebook, kiểm thử, truy nguồn và báo cáo | Lần triển khai ngày 2026-11-06 |
| HoangDang, hỗ trợ OpenAI Codex | Tích hợp lịch sử nhánh QuocAn, xử lý conflict, giữ bản hiện hành và ghi nguồn bản nhập; không sinh lại thuật toán hoặc kết quả thực nghiệm | Ngày 2026-11-11; tác giả và thời điểm commit gốc, kiểm tra và giới hạn trong `../../BRANCH_SYNC_2026-11-11.md` |
| Tác giả notebook/tài liệu cũ | Code và output có trước lần triển khai | Chưa xác nhận tác giả và môi trường chạy |

Người thực hiện chịu trách nhiệm rà soát mã nguồn, kiểm chứng kết quả và công bố phạm vi hỗ trợ AI. Ghi nhận quyền tác giả và đóng góp theo bằng chứng, không quy toàn bộ mã nguồn cho một cá nhân khi chưa được xác nhận.

## Nguồn và phạm vi

- Căn cứ của tài liệu: yêu cầu truy nguồn dự án được ghi nhận tại mục 1; đây là điểm gốc của quy tắc, không sử dụng tham chiếu vòng để chứng minh nguồn gốc.
- Phạm vi dự án: [01_lab_3_description.md](01_lab_3_description.md).
- Tài liệu kỹ thuật ngoài repo: [10_sources.md](10_sources.md).
