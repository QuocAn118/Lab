# Kiểm chứng Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Created date | 2026-11-11 |
| Created by | HoangDang, hỗ trợ OpenAI Codex |
| Last updated | 2026-11-11T09:38:18+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Trạng thái | Đang triển khai; chưa ghi nhận nghiệm thu |

## Tiêu chí kiểm tra

1. Unit/integration tests kiểm dữ liệu thật và pipeline bằng tensor giả; có trường hợp checkpoint `best` khác checkpoint cuối, dữ liệu dự đoán bị sửa, loader rỗng và logits không hữu hạn.
2. Smoke thật: 512/256/256 mẫu, một epoch, đủ đầu ra, `best` tải lại khớp, báo cáo số liệu PASS trước khi chốt manifest; sau đó kiểm hash chỉ đọc.
3. Normal: chạy bốn cấu hình gốc trên cùng máy, mỗi run mới bắt đầu từ model mới. Manifest ghi nguồn/code/môi trường/người chạy/thời gian; không sửa file trong run đã chốt.
4. `verify_run(run_id, finalized=False)` đối chiếu history, năm checkpoint/index, nhãn/tên lớp, số dòng CSV, accuracy/per-class accuracy và reload `best`. Bản finalized kiểm thêm trạng thái manifest và toàn bộ hash/file inventory.
5. Dữ liệu, model summary và file so sánh có sidecar. Mọi số liệu bàn giao dẫn tới artifact thực tế; không dùng lịch sử cũ để nhận kết quả mới.

## Nguồn và phạm vi

- [Đặc tả gốc](lab_1_description.md), [đánh giá](requirements/3_5_evaluate.md), [toàn vẹn](team_plan_integrity.md).
- Tài liệu sẽ được cập nhật bằng lệnh và số đo sau khi triển khai/chạy; không cam kết accuracy trước khi đo.

## Implementation checks (2026-11-11)

- Python 3.12.10; torch 2.9.1+cu128; torchvision 0.24.1+cu128; GPU NVIDIA GeForce RTX 5060 Laptop available and tensor operation succeeded.
- `pip check`: no broken requirements. Exact installed versions: `requirements-lock.txt`.
- FashionMNIST downloaded and checked; 54,000/6,000/10,000 samples, disjoint train/validation; batch [64,1,28,28], float32 [0,1], int64 labels.
- `.venv/Scripts/python.exe -m pytest tests data/tests -q --basetemp <unique-repo-temp>`: 17 passed in 10.61s. The default Windows pytest temp directory was inaccessible; the successful run used a new directory under repository .git.
- These tests include synthetic-data integration; real smoke/normal runs follow after the implementation commit.
