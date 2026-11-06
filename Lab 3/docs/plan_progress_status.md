# Tiến Độ Thực Hiện (Project Status) — Lab 3

## 1. Bảng Công Việc

`DONE` chỉ được ghi khi đã có bằng chứng tương ứng. Mã nghiệm thu `AC` xem tại `lab_3_description.md` mục 8.

| Mã | Công việc | Bước trong đề | Trạng thái | Owner | Bằng chứng |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **TASK-01** | Hoàn thiện đặc tả `docs/` | Toàn bài | **DONE** | Team | 7 file trong `docs/` |
| **TASK-02** | Exercise 1: phân tích cảm xúc | Ex 1, bước 1–4 | **DONE** | Team | Đầu ra Colab trong notebook, 18/18 test (AC-01, AC-02) |
| **TASK-03** | Dữ liệu IMDB và tiền xử lý | Ex 2, bước 1, 2, 4 | **DONE** | Team | Mục `data` trong `reports/metrics.json` (AC-03) |
| **TASK-04** | Model và tokenizer | Ex 2, bước 3 | **DONE** | Team | Mục `model` trong `reports/metrics.json`: 66.955.010 tham số |
| **TASK-05** | Smoke test 2 batch | Quy trình `docs/` | **DONE** | Team | Đầu ra cell test trong notebook: VRAM 2,26 GB (AC-04) |
| **TASK-06** | Huấn luyện đủ 1.000 mẫu | Ex 2, bước 5, 6 | **DONE** | Team | `reports/metrics.json`: 4 epoch trên Tesla T4 (AC-05) |
| **TASK-07** | Checkpoint và early stopping | Ex 2, bước 6 | **DONE** | Team | `checkpoints/`, mục `training.stopped_early` (AC-05, AC-06) |
| **TASK-08** | Đánh giá và đối chiếu KPI | Ex 2, bước 7 | **DONE** | Team | `evaluation.md`: 7/7 KPI đạt (AC-07) |
| **TASK-09** | Demo với câu thực tế | Ex 1, Ex 2 | **DONE** | Team | `demo.md` (AC-01, AC-08) |
| **TASK-10** | Run all notebook trên Colab | Toàn bài | **DONE** | Team | `notebooks/Lab_3_colab.ipynb` đã lưu đầu ra (AC-09) |

---

## 2. Nhật Ký Thực Hiện
- `2026-11-04`: Tạo bộ khung tài liệu đặc tả.
- `2026-11-06`:
  - Đối chiếu notebook cũ với đề bài và đặc tả. Notebook có đủ 2 Exercise, nhưng thứ tự cell sai và cell cuối trùng code nên kết quả không tái tạo được khi chạy Run all.
  - Tách code thành các file trong `src/`, thêm `configs/`, `tests/`, `reports/`. Chuyển notebook cũ vào `notebooks/`.
  - Hoàn thiện đặc tả theo đề bài và thống nhất các điểm mâu thuẫn.
  - Kiểm tra trên máy cá nhân bằng CPU: 18 test đạt, toàn bộ quy trình chạy được với 48/32/32 mẫu và `max_length = 64`. Lần chạy này chỉ để kiểm tra code nên số liệu không dùng để nghiệm thu.
  - Chạy notebook trên Colab với GPU Tesla T4: dừng ở epoch 4/5, Test Accuracy 87,0% (±4,66 điểm %), Macro F1 0,8697, cả 7 KPI đạt. Điền số liệu vào `evaluation.md`, `demo.md` và bảng nghiệm thu.
  - Nạp lại checkpoint đã tải về và đo trên CPU: kết quả khớp `metrics.json`.
  - Lưu notebook có đầu ra từ Colab vào repo. Đối chiếu đầu ra với `metrics.json` và `demo.md`: khớp.

---

## 3. Việc Dùng AI

| Công cụ | Mục đích | AI hỗ trợ | Đóng góp của sinh viên |
| :--- | :--- | :--- | :--- |
| Claude Code (Claude Opus 5.5) | Rà soát, thiết kế, viết code và tài liệu | Đối chiếu đặc tả; đề xuất cấu trúc thư mục; viết `src/`, `tests/`, notebook mới; cập nhật `docs/` | Làm notebook ban đầu; cung cấp đề bài; chọn số mẫu; quyết định phạm vi, cấu trúc và các điểm sửa đặc tả; chạy trên Colab. Phần khác: chưa xác nhận |
