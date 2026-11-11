# Lab 1 — FashionMNIST MLP bằng PyTorch

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Lab 1 — FashionMNIST MLP bằng PyTorch |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-11T09:38:18+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Phân loại ảnh xám 28 × 28 thành 10 lớp trang phục bằng MLP 784 → 512 → 512 → 10. Cấu hình cơ sở: `ToTensor`, batch 64, SGD (lr 0.001, momentum 0.9), tối đa 10 epoch; split train/validation/test là 54.000/6.000/10.000, seed 42.


Đã triển khai pipeline đầy đủ: data → MLP → train/checkpoint → reload best → test/performance → đồ thị/demo → verification/manifest. 17 unit/integration test đã đạt trong môi trường Lab 1. Đang chạy kiểm chứng thật; kết luận cuối tại [verification.md](docs/verification.md).

Phần **3.1–3.2** được gom trong **[data/](data/README.md)**: code chuẩn bị dữ liệu, DataLoader, dữ liệu gốc, báo cáo và kiểm thử. Xem README trong đó để chạy.

Các lệnh sau chạy từ thư mục `Lab 1`; cần Python có `torch`, `torchvision`, `matplotlib` và `pytest` (cho kiểm thử). Dataset gốc không lưu trên Git, bước chuẩn bị cần mạng khi chưa có dữ liệu.

```powershell
python data/prepare_data.py
python data/data_pipeline.py --check
python -m pytest data/tests tests -q
```

Sau khi chuẩn bị dữ liệu, lệnh smoke run dự kiến là `python src/train.py --run-id smoke_001 --purpose "Kiểm tra pipeline"`. Chọn mã mới nếu đã tồn tại. Xem [training.md](docs/training.md) để biết đầu ra hiện có và điều kiện trước normal run. Kiểm tra nguồn gốc file từ đây bằng `python tools/check_provenance.py --strict`.

## 📁 Cấu Trúc Tài Liệu Đặc Tả (`docs/`)
1. **[`docs/lab_1_description.md`](./docs/lab_1_description.md)**: Đặc tả bài toán, mục tiêu hệ thống, chỉ số KPIs & chính sách quản lý Checkpoints.
2. **[`docs/data.md`](./docs/data.md)**:  dữ liệu FashionMNIST, Transforms, DataLoaders.
3. **[`docs/model.md`](./docs/model.md)**: Kiến trúc MLP kế thừa `torch.nn.Module`; hồ sơ model trước và sau huấn luyện.
4. **[`docs/training.md`](./docs/training.md)**: Cấu hình giải thuật tối ưu SGD/Adam, Loss function, quy trình smoke test.
5. **[`docs/evaluation.md`](./docs/evaluation.md)**: Báo cáo định lượng hệ thống (Accuracy, Loss, thời gian & tài nguyên).
6. **[`docs/demo.md`](./docs/demo.md)**: Kịch bản và mã nguồn suy luận nhanh với ảnh mẫu.
7. **[`docs/plan_progress_status.md`](./docs/plan_progress_status.md)**: Lộ trình triển khai, bảng phân rã nhiệm vụ (TODO / DONE).
