# Tiến Độ Thực Hiện (Project Status) — Lab 2

## 1. Bảng Công Việc

`DONE` chỉ được ghi khi đã có bằng chứng tương ứng. Mã nghiệm thu `AC` xem tại `lab_2_description.md` mục 8.

| Mã | Công việc | Bước trong đề | Trạng thái | Owner | Bằng chứng |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **TASK-01** | Hoàn thiện đặc tả `docs/` | Toàn bài | **IN PROGRESS** | Team | 7 file trong `docs/`, đang chờ duyệt |
| **TASK-02** | Cài đặt môi trường | Bước 1 | **TODO** | Team | `requirements.txt`, cell cài đặt trong notebook |
| **TASK-03** | Tải model, xem kiến trúc, thay lớp cuối | Bước 2, 3, 4 | **TODO** | Team | AC-01, AC-02 |
| **TASK-04** | Dữ liệu CIFAR-10 và `DataLoader` | Bước 5 | **TODO** | Team | AC-03 |
| **TASK-05** | Smoke test 2 batch | Quy trình `docs/` | **TODO** | Team | AC-04 |
| **TASK-06** | Huấn luyện mô hình gốc | Bước 6 | **TODO** | Team | AC-08 |
| **TASK-07** | So sánh ResNet18 và DenseNet121 | Exercise 1 | **TODO** | Team | AC-05 |
| **TASK-08** | So sánh learning rate và batch size | Exercise 2 | **TODO** | Team | AC-06 |
| **TASK-09** | Theo dõi bằng TensorBoard | Exercise 3 | **TODO** | Team | AC-07 |
| **TASK-10** | Đánh giá trên tập test và đối chiếu KPI | Bước 7 | **TODO** | Team | AC-09 |
| **TASK-11** | Demo suy luận | Quy trình `docs/` | **TODO** | Team | AC-10 |
| **TASK-12** | Run all notebook trên Colab | Toàn bài | **TODO** | Team | AC-11 |

---

## 2. Nhật Ký Thực Hiện
- `2026-10-31`: Chạy notebook cũ trên Colab với SGD, 2 epoch, ảnh $32 \times 32$. Kết quả xem tại `evaluation.md` mục 7.
- `2026-11-04`: Tạo bộ khung tài liệu đặc tả.
- `2026-11-07`:
  - Đối chiếu đặc tả và notebook cũ với đề bài. Notebook có đủ 7 bước và 3 Exercise, nhưng khác đặc tả về optimizer, kích thước ảnh, số epoch và checkpoint. Đặc tả thiếu 3 Exercise, bước xem kiến trúc và đánh giá trên tập test.
  - Thống nhất làm lại theo đặc tả. Hoàn thiện đặc tả theo đề bài: 2 model, 4 lần chạy, tối đa 6 epoch, 22.500/2.500/5.000 ảnh.

---

## 3. Việc Dùng AI

| Công cụ | Mục đích | AI hỗ trợ | Đóng góp của sinh viên |
| :--- | :--- | :--- | :--- |
| Claude Code (Claude Opus 5.5) | Rà soát và viết tài liệu | Đối chiếu đề bài với đặc tả và notebook; đề xuất các điểm sửa; cập nhật `docs/` | Làm notebook ban đầu; cung cấp đề bài; chọn model, số lần chạy, số epoch, số ảnh; duyệt các điểm sửa. Phần khác: chưa xác nhận |
