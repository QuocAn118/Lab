# Tiến độ và bàn giao — Lab 3

## 1. Kết quả ngày 2026-11-06

| Task | Công việc | Trạng thái | Bằng chứng |
| --- | --- | --- | --- |
| TASK-01 | Chốt đặc tả và nguồn AI | DONE | 08_implementation.md, 11_AI_Usage.md, 10_sources.md, 12_traceability.md |
| TASK-02 | Module A inference | DONE | outputs/module-a/inference.json; câu Việt/Anh chạy thật |
| TASK-03 | IMDB split/tokenize | DONE | 20,000/5,000/25,000; class counts/index hashes trong run_config.json |
| TASK-04 | Smoke pretrained | DONE | outputs/smoke; 2 optimizer steps, save/reload |
| TASK-05 | Full training | DONE | outputs/full; 3 epochs, 3,750 steps, test 25,000 mẫu |
| TASK-06 | 5 checkpoint/early stopping | DONE | Nạp được cả 5; regression test cờ dừng tại epoch cuối |
| TASK-07 | Benchmark và nhận xét | DONE | Test accuracy 91.012%, macro F1 .910118; overfitting được báo rõ |
| TASK-08 | Demo và notebook | DONE | Demo best; 8 ô code (Exercise 1 + 7 bước Exercise 2) đã chạy; 10 tests PASS |
| TASK-09 | Audit nguồn và artifact | DONE | Audit sau sắp xếp: 20 file/8 ô có nguồn; 45 artifact full khớp hash |

Đã hoàn tất phạm vi đặc tả trên máy hiện tại. Đối chiếu tốc độ trên Colab T4: N/A do khác thiết bị đo. Không tự commit/push. Trọng số được giữ trên đĩa trong outputs và bị Git ignore; không nhầm Git không có trọng số với chưa train.

## 2. Nhật ký

- 2026-11-04: bộ khung đặc tả có sẵn; các bước triển khai còn TODO.
- 2026-11-06: ghi nhận yêu cầu truy nguồn của dự án; chốt R-01–R-09 trước code.
- 2026-11-06: triển khai CLI dùng chung notebook, kiểm thử, môi trường riêng và kernel.
- 2026-11-06: sửa lỗi evaluator/snapshot CRLF, chạy lại smoke; full chỉ chạy một lần, không chọn theo test.
- 2026-11-06: hiệu chỉnh cờ dừng đúng epoch cuối và tương phản hình; giữ bản gốc, không thay số đo/model.
- 2026-11-06: cập nhật kết quả, chạy notebook và audit, bàn giao tài liệu và artifact.
- 2026-11-06: sắp notebook thành Exercise 1 và bảy bước Exercise 2; đánh số docs 01–12, thêm hướng dẫn trình bày không đánh số và cập nhật đường dẫn hiện hành. Giữ nguyên artifact lịch sử.

## Nguồn và phạm vi

- [08_implementation.md](08_implementation.md): phạm vi đã chốt; [12_traceability.md](12_traceability.md): liên kết yêu cầu/code/test.
- [06_verification.md](06_verification.md): kết quả thực tế và giới hạn; [11_AI_Usage.md](11_AI_Usage.md): khai báo hỗ trợ AI.

## 3. Notebook rút gọn

Notebook hiện có 8 ô code (Exercise 1 + 7 bước Exercise 2) và hai hàm nhỏ. Dùng thư viện trực tiếp, không class tùy biến hoặc cơ chế audit/AST trong luồng chính. Đã thực thi tất cả ô sau khi sắp bảy bước; kiểm tra trước đó về luồng huấn luyện mới dùng 32/16/16 mẫu. Đánh giá lại checkpoint full trên đủ 25.000 mẫu test đạt accuracy 91.012%, macro F1 .910118; không full train lại. Căn cứ: [08_implementation.md](08_implementation.md) §6 và [06_verification.md](06_verification.md) §5.1. Lần đánh số docs chỉ đổi tham chiếu nguồn trong notebook, giữ nguyên code tính toán và output đã có.
