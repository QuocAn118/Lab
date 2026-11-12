# Thử Nghiệm & Trực Quan Hóa (Demo Specification) — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Thử Nghiệm & Trực Quan Hóa (Demo Specification) — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## Input/output thật

[Ảnh input](../outputs/demo_input.png): test index 0, ảnh xám 28×28. [JSON output](../outputs/demo_example.json) được lấy trực tiếp từ CLI đọc best của **run_004**, run được chọn bằng validation.

```json
{
  "run_id": "run_004",
  "checkpoint": "best.pth",
  "sample_index": 0,
  "input_shape": [
    1,
    28,
    28
  ],
  "true_label": 9,
  "true_name": "Ankle boot",
  "predicted_label": 9,
  "predicted_name": "Ankle boot",
  "logits": [
    -6.360266208648682,
    -6.564720630645752,
    -7.691073417663574,
    -7.286394119262695,
    -8.563128471374512,
    0.12268572300672531,
    -5.720861911773682,
    1.0425623655319214,
    -6.852451324462891,
    4.723461151123047
  ]
}
```

Lệnh từ `Lab 1`:

```powershell
& .venv/Scripts/python.exe src/demo.py --run-id run_004 --index 0
```

CLI chỉ đọc checkpoint/config/dataset và in JSON, không ghi vào run đã kết thúc. Đổi index trong khoảng 0–9.999 cho normal run; smoke dùng 0–255. Từ chối index hoặc run/checkpoint không hợp lệ.

## Ảnh mẫu và model

[prediction_samples.png](../outputs/run_004/prediction_samples.png) gồm 16 ảnh test cố định seed 42, nhãn thật/dự đoán; nhãn sai tô đỏ. File được tạo trước khi manifest chốt. Checkpoint ở `checkpoints/run_004/best.pth`; kiến trúc/transforms/classes từ [config.json](../outputs/run_004/config.json).

Logits là điểm số thô, không phải xác suất; argmax chọn nhãn. Model eval và inference_mode, ToTensor theo cấu hình. Demo không dùng để chọn checkpoint bằng test.

## Nguồn và phạm vi

- [Đặc tả gốc](lab_1_description.md), [3.6](requirements/3_6_save_load.md), artifact thật của run_004. Đây là kết quả suy luận thực tế ngày 2026-11-11, không phải số minh họa.
