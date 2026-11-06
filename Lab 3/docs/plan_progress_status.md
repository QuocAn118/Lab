# Tiến độ và bàn giao — Lab 3

## 1. Kết quả ngày 2026-11-06

| Task | Công việc | Trạng thái | Bằng chứng |
| --- | --- | --- | --- |
| TASK-01 | Chốt đặc tả và nguồn AI | DONE | implementation.md, AI_Usage.md, sources.md, traceability.md |
| TASK-02 | Module A inference | DONE | outputs/module-a/inference.json; câu Việt/Anh chạy thật |
| TASK-03 | IMDB split/tokenize | DONE | 20,000/5,000/25,000; class counts/index hashes trong run_config.json |
| TASK-04 | Smoke pretrained | DONE | outputs/smoke; 2 optimizer steps, save/reload |
| TASK-05 | Full training | DONE | outputs/full; 3 epochs, 3,750 steps, test 25,000 mẫu |
| TASK-06 | 5 checkpoint/early stopping | DONE | Nạp được cả 5; regression test cờ dừng tại epoch cuối |
| TASK-07 | Benchmark và nhận xét | DONE | Test accuracy 91.012%, macro F1 .910118; overfitting được báo rõ |
| TASK-08 | Demo và notebook | DONE | Demo best; 11 ô code notebook chạy, 10 tests PASS |
| TASK-09 | Audit nguồn và artifact | DONE | 19 file/11 ô có nguồn; 45 artifact full khớp hash |

Đã hoàn tất phạm vi đặc tả trên máy hiện tại. Đối chiếu tốc độ trên Colab T4: N/A do khác thiết bị đo. Không tự commit/push. Trọng số được giữ trên đĩa trong outputs và bị Git ignore; không nhầm Git không có trọng số với chưa train.

## 2. Nhật ký

- 2026-11-04: bộ khung đặc tả có sẵn; các bước triển khai còn TODO.
- 2026-11-06: ghi nhận yêu cầu truy nguồn của dự án; chốt R-01–R-09 trước code.
- 2026-11-06: triển khai CLI dùng chung notebook, kiểm thử, môi trường riêng và kernel.
- 2026-11-06: sửa lỗi evaluator/snapshot CRLF, chạy lại smoke; full chỉ chạy một lần, không chọn theo test.
- 2026-11-06: hiệu chỉnh cờ dừng đúng epoch cuối và tương phản hình; giữ bản gốc, không thay số đo/model.
- 2026-11-06: cập nhật kết quả, chạy notebook và audit, bàn giao tài liệu và artifact.

## Nguồn và phạm vi

- [implementation.md](implementation.md): phạm vi đã chốt; [traceability.md](traceability.md): liên kết yêu cầu/code/test.
- [verification.md](verification.md): kết quả thực tế và giới hạn; [AI_Usage.md](AI_Usage.md): khai báo hỗ trợ AI.

## 3. Notebook rút gọn

Notebook hiện có 11 ô code, 104 dòng code và hai hàm nhỏ. Dùng thư viện trực tiếp, không class tùy biến hoặc cơ chế audit/AST trong luồng chính. Đã thực thi tất cả ô; kiểm tra riêng luồng huấn luyện mới trên 32/16/16 mẫu và đánh giá lại checkpoint full trên đủ 25.000 mẫu test. Accuracy 91.012%, macro F1 .910118; không full train lại. Căn cứ: [implementation.md](implementation.md) §6 và [verification.md](verification.md) §5.1.
