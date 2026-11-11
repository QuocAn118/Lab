# <Tên dự án / Lab> — <Một câu mô tả bài toán>

> Mẫu file mô tả gốc cho bài lab hoặc đồ án ML. Ví dụ đã điền: `Lab 1/docs/lab_1_description.md`.
> Xóa các dòng `>` hướng dẫn khi điền xong.

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | |
| Created date | YYYY-MM-DD |
| Created by | |
| Last updated | YYYY-MM-DD — <thay đổi gì> |
| Updated by | |
| Tài liệu tham chiếu | <chương, slide, đề bài> |
| Trạng thái | Đặc tả / Đang triển khai / Đã huấn luyện / Hoàn thành |

## 1. Bài toán

### 1.1. Bối cảnh
### 1.2. Vấn đề cần giải quyết
### 1.3. Đầu vào

| Nội dung | Giá trị |
| --- | --- |
| Dataset | |
| Kiểu dữ liệu / kích thước | |
| Số lớp / nhãn | |
| Phân vùng train / val / test | <số mẫu, cách chia, seed> |

### 1.4. Đầu ra

## 2. Mục tiêu

## 3. Phạm vi

| Trong phạm vi | Ngoài phạm vi |
| --- | --- |
| | |

## 4. Hệ thống tài liệu

| File | Nội dung | Trạng thái |
| --- | --- | --- |
| `<x>_description.md` | Bài toán, phạm vi, quy ước, tiêu chí hoàn thành | |
| `data.md` | Dữ liệu, phân vùng, tiền xử lý | |
| `model.md` | Hồ sơ model (mục 9) | |
| `training.md` | Siêu tham số, smoke/normal run, checkpoint | |
| `evaluation.md` | Accuracy, thời gian, tài nguyên; so sánh run | |
| `demo.md` | Cách nạp model và dự đoán | |
| `plan_progress_status.md` | Công việc, quyết định, nhật ký | |
| `requirements/` | Đặc tả chi tiết từng yêu cầu | |

## 5. Quy trình thực hiện

Description → Discussion → Detail → Implementation → Small prompt → Smoke run → Normal run → Check với file mô tả.

| Bước | Làm gì | Kết quả cần có |
| --- | --- | --- |
| 1. Description | | |
| 2. Discussion | | |
| 3. Detail | | |
| 4. Implementation | | |
| 5. Small prompt | | |
| 6. Smoke run | | |
| 7. Normal run | | |
| 8. Check | | |

### 5.1. Smoke run
> Dữ liệu nhỏ, số epoch/step, mã lần chạy, điều kiện đạt.

### 5.2. Normal run

## 6. Yêu cầu thực hiện

| Yêu cầu | File đặc tả | File code | Đầu ra chính | Kiểm tra |
| --- | --- | --- | --- | --- |
| | | | | |

### Cấu hình cơ sở

| Tham số | Giá trị |
| --- | --- |
| Kiến trúc | |
| Batch size | |
| Loss | |
| Optimizer / learning rate | |
| Số epoch tối đa | |
| Early stopping | patience <n>, ngưỡng <0–2%> |
| Seed | |
| Thiết bị | |

### 6.1. Chính sách checkpoint

| Checkpoint | Khi nào lưu | Ý nghĩa |
| --- | --- | --- |
| `final_epoch` | Sau epoch cuối | Trọng số khi kết thúc |
| `best_val_loss` | Khi val loss giảm | Val loss nhỏ nhất |
| `best_val_acc` | Khi val accuracy tăng | Val accuracy lớn nhất |
| `best_val_pause` | Lần validation cuối, kèm cờ `early_stopped` | Thời điểm dừng (cải thiện 0–2%) |
| `best` | Sau khi train xong | Bản dùng cho test và demo; ghi rõ copy từ checkpoint nào |

## 7. Thí nghiệm và phân tích

| Run | Thay đổi so với cơ sở | Mục đích |
| --- | --- | --- |
| `run_001` | Cơ sở | Mốc so sánh |

## 8. Đánh giá performance

| Nhóm | Chỉ số | Cách đo |
| --- | --- | --- |
| Accuracy | | |
| Thời gian xử lý | Train tổng / mỗi epoch; suy luận ms/mẫu | |
| Tài nguyên | Số tham số, kích thước model, bộ nhớ đỉnh, thiết bị | |

## 9. Hồ sơ model

1. Thông tin: Name, Created date + người tạo, Last updated + người cập nhật.
2. Purpose.
3. Initial model: kiến trúc, input/output, file nguồn.
4. Trained model: run được chọn, mô tả hoàn thành huấn luyện, đường dẫn lưu.

## 10. Quy ước tổ chức và lưu trữ

```text
<project>/
├── docs/
├── src/
├── data/
├── outputs/<run_id>/
└── checkpoints/<run_id>/
```

## 11. Sản phẩm nộp và tiêu chí hoàn thành

- [ ] Smoke run đạt.
- [ ] Normal run có đủ file đầu ra.
- [ ] Performance có đủ accuracy, thời gian, tài nguyên.
- [ ] `model.md`, `evaluation.md`, `plan_progress_status.md` đã cập nhật.
- [ ] Đã đối chiếu kết quả với file mô tả.

## Nguồn và phạm vi
