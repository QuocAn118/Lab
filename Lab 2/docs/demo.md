# Thử Nghiệm Thực Tế (Demo Specification) — Lab 2

Mục đích: Kiểm chứng model đã huấn luyện bằng cách phân loại từng ảnh và xem xác suất của các lớp. Các lệnh chạy trong thư mục `Lab 2` trên Colab, với `PYTHONPATH=src` đã được notebook thiết lập.

Mã nguồn: `src/lab_2/demo.py`.

## 1. Cách Chạy

```bash
# Mặc định: 3 ảnh đầu của tập test, có nhãn thật để đối chiếu
python -m lab_2.demo

# Dùng ảnh của bạn, có thể truyền nhiều ảnh
python -m lab_2.demo --image "duong_dan/anh_1.jpg" "duong_dan/anh_2.png"
```

Lệnh này nạp `checkpoints/best.pth`, tiền xử lý ảnh theo cột `val`, `test`, demo trong `data.md` mục 3, rồi dùng softmax để đổi logits thành xác suất. Kết quả được ghi vào mục `demo` của `reports/metrics.json`, biểu đồ xác suất lưu tại `reports/figures/demo_probabilities.png`.

Nếu chưa có `checkpoints/best.pth` hoặc không đọc được ảnh, lệnh dừng và báo lỗi rõ ràng.

## 2. Kết Quả Trên Ảnh Tập Test

| Ảnh | Dự đoán | Xác suất | Nhãn thật | Đúng/Sai |
| :--- | :---: | :---: | :---: | :---: |
| Ảnh test số 1 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |
| Ảnh test số 2 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |
| Ảnh test số 3 | Chờ Colab | Chờ Colab | Chờ Colab | Chờ Colab |

## 3. Giới Hạn

- Model chỉ biết 10 lớp của CIFAR-10. Ảnh ngoài 10 lớp này vẫn bị xếp vào một trong 10 lớp, nên xác suất cao không có nghĩa là kết quả đáng tin.
- Model học từ ảnh $32 \times 32$ đã phóng to, nên có thể kém chính xác hơn với ảnh chụp thật có độ phân giải cao.
