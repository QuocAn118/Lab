# Thử Nghiệm & Trực Quan Hóa (Demo Specification) — Lab 1

Last updated: 2026-11-10 — Codex (AI hỗ trợ, chờ nhóm review).

Trạng thái: đặc tả demo, chưa có script và checkpoint đã huấn luyện để chạy demo.

## 1. Đầu vào

- `outputs/<run_id>/config.json`: kiến trúc, transforms và thứ tự tên lớp.
- `checkpoints/<run_id>/best.pth`: bản sao checkpoint có val accuracy cao nhất; hòa giữ epoch sớm hơn.
- Tập test FashionMNIST đã chuẩn bị, giữ nguyên thứ tự; dùng `ToTensor` theo cấu hình.

## 2. Kịch bản cần triển khai

1. Kiểm tra cấu hình và checkpoint cùng `run_id`; thiếu thì báo lỗi rõ đường dẫn.
2. Dùng `build_model(config["model"])`, tải `best.pth` với `weights_only=True`, `map_location` theo thiết bị; đặt `model.eval()`.
3. Chọn 16 ảnh test cố định bằng seed 42, dự đoán khi tắt gradient. `argmax` trên logits cho nhãn dự đoán; tra tên theo `config["classes"]`.
4. Hiển thị ảnh, nhãn thật và nhãn dự đoán, tô đỏ trường hợp sai; lưu `outputs/<run_id>/prediction_samples.png` trước khi chốt manifest.

Demo dùng cùng checkpoint và transforms với đánh giá; không dùng test để chọn checkpoint. `load_model(run_id, device)` hiện chỉ nạp `final_epoch.pth`, nên chưa đáp ứng bước 2. Cần hoàn thiện phần này theo [3.6](requirements/3_6_save_load.md) trước khi ghi nhận demo hoàn thành.

