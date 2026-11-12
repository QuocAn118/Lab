# Performance Evaluation & System Benchmarks — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Performance Evaluation & System Benchmarks — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## Kết quả thực tế ngày 2026-11-11

Bốn normal run trên cùng **NVIDIA GeForce RTX 5060 Laptop GPU**, Python 3.12.10, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128. Mỗi run đánh giá checkpoint best trên đủ 10.000 ảnh test. Số mẫu đúng: run_001 = 8253, run_002 = 8271, run_003 = 8667, run_004 = 8770.

| Run | Epoch | Best val acc (0–1) | Test acc (0–1) | Test loss | Train giây | Test ms/ảnh | Nguồn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run_001 | 6 | 0.831500 | 0.8253 | 0.497000 | 22.082 | 0.06119 | [nguồn](../outputs/run_001/performance.json) |
| run_002 | 8 | 0.835333 | 0.8271 | 0.482226 | 29.316 | 0.05843 | [nguồn](../outputs/run_002/performance.json) |
| run_003 | 4 | 0.873333 | 0.8667 | 0.369785 | 14.654 | 0.05354 | [nguồn](../outputs/run_003/performance.json) |
| run_004 | 4 | 0.881000 | 0.8770 | 0.346382 | 15.644 | 0.04997 | [nguồn](../outputs/run_004/performance.json) |

Nguồn tổng hợp: [compare_runs.csv](../outputs/compare_runs.csv), [compare_loss.png](../outputs/compare_loss.png). Các nguồn metrics/predictions/history/manifest nằm trong thư mục tương ứng của từng run; tất cả đã được verifier đối chiếu.

## Chọn model và nhận xét

Chọn **run_004 bằng validation**, best val accuracy **0.8810**; test của checkpoint đó là **0.8770 (87.70%)**. Không chọn checkpoint hoặc run bằng test accuracy.

- Model nhỏ có 235.146 tham số so với 669.706 của baseline. Lần đo này giảm kích thước checkpoint nhưng chạy 8 epoch thay vì 6, nên tổng thời gian train không nhỏ hơn baseline.
- Learning rate 0.01 cho SGD đạt val/test cao hơn baseline trong 4 epoch. Adam đạt validation cao nhất trong bốn cấu hình của lần đo này.
- Train và val loss đều giảm trong các epoch đã chạy; chưa quan sát đoạn val loss tăng trong khi train loss giảm. Không suy kết luận cho các epoch chưa chạy.
- Tất cả run dừng sớm đúng patience 2, min_delta 0.01. Accuracy vẫn có thể tăng nhẹ mà không reset bộ đếm; đây là quy tắc đã chốt, không phải bắt buộc loss/accuracy ngừng cải thiện hoàn toàn.
- Mỗi cấu hình chạy một lần, seed 42; không báo độ tin cậy thống kê hoặc khẳng định thứ hạng áp dụng cho mọi seed/máy.

## Tài nguyên

| Run | Tham số | Best checkpoint MB | GPU peak allocated MB |
| --- | --- | --- | --- |
| run_001 | 669,706 | 2.681685 | 27.313 |
| run_002 | 235,146 | 0.943445 | 21.246 |
| run_003 | 669,706 | 2.681685 | 27.313 |
| run_004 | 669,706 | 2.681685 | 32.410 |

MB = 1.000.000 byte. GPU peak là PyTorch allocated, không phải toàn bộ bộ nhớ tiến trình/driver trong nvidia-smi. Train seconds chỉ tính bước train, không gồm validation, lưu checkpoint, reload, test hoặc vẽ hình. Test ms/ảnh là trung bình của toàn vòng test theo batch 64, có loader/chuyển thiết bị; không phải benchmark độ trễ một ảnh độc lập.

## Accuracy từng lớp của run_004

| Lớp | Accuracy (0–1) |
| --- | --- |
| T-shirt/top | 0.872 |
| Trouser | 0.973 |
| Pullover | 0.761 |
| Dress | 0.816 |
| Coat | 0.839 |
| Sandal | 0.948 |
| Shirt | 0.681 |
| Sneaker | 0.951 |
| Bag | 0.964 |
| Ankle boot | 0.965 |

Shirt là lớp thấp nhất trong lần đo này; cần đọc ảnh đoán sai và CSV để phân tích thêm, không bịa nguyên nhân từ một con số.

## Truy vết và kiểm chứng

`best.pth` sao chép byte từ best_val_acc; `reload_check.json` kiểm best trên 16 ảnh validation, sai khác logits bằng 0. Năm checkpoint có index và manifest/hash. `verify.py --run-id ...` chỉ đọc; không thêm file vào run đã chốt. Xem [verification.md](verification.md).

## Nguồn và phạm vi

- [Đặc tả gốc](lab_1_description.md), [3.5](requirements/3_5_evaluate.md), artifact thật của bốn run nêu trên. Báo cáo được HoangDang tổng hợp; không dùng kết quả Lab 3 hoặc smoke thay cho normal run.
