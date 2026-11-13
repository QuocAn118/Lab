# Kiểm chứng dữ liệu Lab 1 — Phần 3.1 và 3.2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Created date / Created by | Chưa xác nhận người tạo nội dung ban đầu |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00; HoangDang; aee3dd8 |
| Last updated | 2026-11-11 (UTC+07:00) |
| Updated by — nguồn tài liệu | Nguyễn Thanh Bình (Thanh Binh trong Git), `542b395` |
| Người/ngày tiếp nhận | HoangDang; 2026-11-12 |
| Phạm vi | 3.1 — Chuẩn bị FashionMNIST; 3.2 — Transforms và DataLoader |
| Trạng thái | Hoàn thành triển khai và kiểm thử phần data |

## Kết luận nghiệm thu kỹ thuật

Đã triển khai tải và kiểm tra FashionMNIST, chia train/validation cố định, lưu ảnh khảo sát và tạo ba DataLoader. Dữ liệu sử dụng gồm **54.000 train, 6.000 validation và 10.000 test**. Kiểm thử riêng data chạy lại ngày 2026-11-11 đạt **2 passed in 33.92s**.

| Giai đoạn | Lệnh/kiểm tra | Kết quả thực tế |
| --- | --- | --- |
| Tải dữ liệu — 3.1 | `data/prepare_data.py`, `FashionMNIST(download=True)` | 60.000 train gốc, 10.000 test; ảnh xám 28×28; 10 lớp |
| Chia tập | `split_indices()`, `torch.randperm`, seed 42 | 54.000/6.000; train ∩ validation = 0; giữ test độc lập |
| Khảo sát | 16 ảnh train gốc đầu tiên | PNG 4×4, mỗi ảnh có nhãn thật |
| Tiền xử lý — 3.2 | `ToTensor()` trên cả ba tập | Ảnh `float32 [1,28,28]`, giá trị [0,1]; không Normalize |
| DataLoader | `create_dataloaders(config)` | Trả về train/validation/test; batch cơ sở 64; chỉ train shuffle; `drop_last=False`, `num_workers=0` |
| Batch check | `data/data_pipeline.py --check` | Ảnh `[64,1,28,28]`, nhãn `[64]`; dtype `float32`/`int64`; pixel min 0.0, max 1.0 |
| Kiểm thử riêng data | `python -m pytest "Lab 1/data/tests/test_data.py" -q` từ gốc repo | **2 passed in 33.92s** |
| Tích hợp chung của nhóm | Theo [verification.md](verification.md) | 20 test toàn pipeline, smoke và bốn normal run đạt; không phải 20 test riêng data |

## Người chạy, thời gian và commit

| Mốc | Thời gian | Người/công cụ và nguồn | Phạm vi |
| --- | --- | --- | --- |
| Triển khai data ban đầu | 2026-11-10 | Theo yêu cầu người dùng; commit `eb86ab8` | Code, báo cáo và kiểm thử 3.1–3.2 |
| Kiểm chứng tích hợp | 2026-11-11 | HoangDang; [verification.md](verification.md) | Pipeline chung, không ghi nhận thành công việc riêng phần data |
| Kiểm tra lại phần data | 2026-11-11 | HoangDang chạy trên workspace | Hai test dữ liệu thật đạt; không chạy lại training |

Tác giả nội dung ban đầu chưa xác nhận được giữ nguyên trong lịch sử. Không suy người tạo code từ tên nhánh hoặc người tích hợp.

## Phép kiểm toàn vẹn

- Kiểm tra số ảnh, shape và ánh xạ nhãn; giữ đúng thứ tự 10 tên lớp từ FashionMNIST.
- Tính SHA-256 trên byte nguyên bản của bốn tệp IDX đã giải nén; lưu trong `data_summary.json`.
- Lưu chỉ số chia tập trong `split_indices.json`; kiểm tra train/validation không giao nhau và phủ đủ 60.000 mẫu train gốc.
- Kiểm tra duyệt đủ số mẫu, shape/dtype, miền pixel [0,1] và nhãn 0–9; giữ nguyên thứ tự validation/test.
- Kiểm tra tái lập batch đầu với cùng seed, smoke limits và từ chối batch size/transforms/limits không hợp lệ.
- Mỗi báo cáo có `.prov.json` lưu nguồn gốc, hash và lịch sử tạo lại. Không sửa artifact thực nghiệm đã chốt.

## Sản phẩm và giới hạn bàn giao

| Sản phẩm | Vị trí |
| --- | --- |
| Code chuẩn bị dữ liệu | [data/prepare_data.py](../data/prepare_data.py) |
| Code chia tập và DataLoader | [data/data_pipeline.py](../data/data_pipeline.py) |
| Dataset gốc qua Git LFS | `data/FashionMNIST/raw/` |
| Thống kê và hash | [data_summary.json](../data/reports/data_summary.json) |
| Chỉ số train/validation | [split_indices.json](../data/reports/split_indices.json) |
| Ảnh khảo sát | [data_samples.png](../data/reports/data_samples.png) |
| Báo cáo batch | [batch_check.json](../data/reports/batch_check.json) |
| Kiểm thử | [data/tests/test_data.py](../data/tests/test_data.py) |

Từ thư mục `Lab 1`, dùng Python đã cài các thư viện trong `requirements.txt`:

```powershell
python data/prepare_data.py
python data/data_pipeline.py --check
python -m pytest data/tests/test_data.py -q
```

Sau clone/pull, dùng `git lfs pull` từ gốc repo để lấy dữ liệu nhị phân. Chỉ chạy chuẩn bị khi cần tải hoặc khảo sát lại vì lệnh sẽ tái tạo báo cáo và cập nhật sidecar. Hàm tạo loader không tự tải dữ liệu, không lưu toàn bộ tensor batch thành file; đường dẫn được xác định từ vị trí script.

Split ngẫu nhiên theo chỉ số, chưa chia phân tầng. Cấu hình cơ sở chỉ hỗ trợ `ToTensor()`; thay đổi transforms cần đồng bộ với huấn luyện và suy luận. Kiểm thử chưa mô phỏng mọi tình huống mất mạng hoặc hỏng tệp. Kết quả accuracy và checkpoint thuộc báo cáo chung của nhóm.

## Nguồn và phạm vi

- [Yêu cầu 3.1](requirements/3_1_prepare_data.md), [yêu cầu 3.2](requirements/3_2_transforms_dataloader.md), [mô tả Lab 1](lab_1_description.md).
- Code và báo cáo thật trong `data/`; [kiểm chứng chung](verification.md), [tiến độ](plan_progress_status.md).
