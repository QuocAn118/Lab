# Thử Nghiệm & Trực Quan Hóa (Demo Specification) — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Thử Nghiệm & Trực Quan Hóa (Demo Specification) — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T09:38:18+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Trạng thái: đã có `src/demo.py`; đang tạo checkpoint và bằng chứng chạy thật.

## 1. Đầu vào

- `outputs/<run_id>/config.json`: kiến trúc, transforms và thứ tự tên lớp.
- `checkpoints/<run_id>/best.pth`: bản sao checkpoint có val accuracy cao nhất; hòa giữ epoch sớm hơn.
- Tập test FashionMNIST đã chuẩn bị, giữ nguyên thứ tự; dùng `ToTensor` theo cấu hình.

## 2. Kịch bản đã triển khai

1. Kiểm tra cấu hình và checkpoint cùng `run_id`; thiếu thì báo lỗi rõ đường dẫn.
2. Dùng `build_model(config["model"])`, tải `best.pth` với `weights_only=True`, `map_location` theo thiết bị; đặt `model.eval()`.
3. Chọn 16 ảnh test cố định bằng seed 42, dự đoán khi tắt gradient. `argmax` trên logits cho nhãn dự đoán; tra tên theo `config["classes"]`.
4. Hiển thị ảnh, nhãn thật và nhãn dự đoán, tô đỏ trường hợp sai; lưu `outputs/<run_id>/prediction_samples.png` trước khi chốt manifest.

Demo dùng cùng best/transforms với đánh giá; không dùng test để chọn checkpoint. CLI chỉ đọc: `python src/demo.py --run-id run_001 --index 0`; in shape ảnh, 10 logits, nhãn thật và dự đoán, không thêm file vào run đã chốt.

