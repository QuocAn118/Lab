# Tiến độ và bàn giao — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Tiến độ và bàn giao — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## Hiện trạng bàn giao ngày 2026-11-14

Bộ 01–12 là tài liệu chính. Notebook hiện ở `Lab 3/Lab_3 (1).ipynb`, gồm 8 ô code; thư mục `docs/docs demo/` giữ hướng dẫn trình bày bản chính và bảy tài liệu Quốc An tham khảo. Cấu hình full 20.000/5.000/25.000, tối đa 3 epochs không thay đổi.

Lần kiểm sau chuẩn hóa ngày 2026-11-14 đạt **46 test**, source audit **41 file/8 ô code**, **212 liên kết nội bộ** không hỏng và **687 file bằng chứng/notebook** giữ nguyên hash; chi tiết tại [06_verification.md mục 8](06_verification.md). Run All ngày 2026-11-12 đánh giá lại checkpoint full có sẵn, accuracy 91,012%, macro F1 0,910118. Snapshot nhập có đủ năm checkpoint; metadata thời gian khôi phục ở outputs/full là nguồn riêng. Bản chính chưa có benchmark T4 hoặc full training mới trong đợt chuẩn hóa này.

## 1. Kết quả lịch sử ngày 2026-11-06

Trạng thái kiểm lại ngày 2026-11-12 được ghi tại `06_verification.md` mục 7. Notebook tám cell đã chạy trên checkpoint nhập với đủ 25.000 test. Bộ kiểm thử tại thời điểm 2026-11-12 đạt 44 test (kiểm lại sau `25cc678`, cần `PYTHONPATH=src`), smoke pretrained mới và manifest artifact đạt. HoangDang thực hiện sửa đường dẫn nạp model, nối lại tiện ích report/thời gian và bổ sung kiểm tra bảo vệ phiên trong bản Quốc An.

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

Theo báo cáo ngày 2026-11-06, phạm vi đặc tả đã hoàn tất trên máy được ghi trong báo cáo; đây không phải kiểm chứng lại môi trường ngày 2026-11-11. Đối chiếu tốc độ trên Colab T4: N/A do khác thiết bị đo. Trọng số của các bản nhập hiện được quản lý bằng Git LFS và snapshot có manifest nguồn riêng; nội dung trong bảng là kết quả lịch sử, không phải số đo mới sau merge.

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

Notebook hiện có 8 ô code (Exercise 1 + 7 bước Exercise 2) và hai hàm nhỏ. Notebook dùng API thư viện trực tiếp; callback thời gian và ghi report nằm trong notebook_helpers.py, không đưa audit/AST vào luồng trình bày. Đã thực thi tất cả ô sau khi sắp bảy bước; kiểm tra trước đó về luồng huấn luyện mới dùng 32/16/16 mẫu. Đánh giá lại checkpoint full trên đủ 25.000 mẫu test đạt accuracy 91.012%, macro F1 .910118; không full train lại. Căn cứ: [08_implementation.md](08_implementation.md) §6 và [06_verification.md](06_verification.md) §5.1. Lần đánh số docs chỉ đổi tham chiếu nguồn trong notebook, giữ nguyên code tính toán và output đã có.
