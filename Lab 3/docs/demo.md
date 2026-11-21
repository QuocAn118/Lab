# Thử Nghiệm Thực Tế (Demo Specification) — Lab 3

Mục đích: Kiểm chứng model với những câu thực tế, ngoài tập dữ liệu. Các lệnh chạy trong thư mục `Lab 3` trên Colab, với `PYTHONPATH=src` đã được notebook thiết lập.

## 1. Exercise 1: Phân Tích Cảm Xúc

```bash
python -m exercise_1.sentiment_analysis --text "Hôm qua em tuyệt lắm!"
python -m exercise_1.sentiment_analysis --text "The movie was boring and far too long."
```

Câu được truyền qua tham số `--text` thay vì `input()`, để notebook gọi được mà không phải chờ nhập tay. Câu rỗng sẽ báo lỗi.

| Câu thử | Token | Nhãn dự đoán | Điểm | Nguồn |
| :--- | :--- | :---: | :---: | :--- |
| *"Hôm qua em tuyệt lắm!"* | `H`, `##ô`, `##m`, `qua`, `em`, `tuyệt`, `l`, `##ắm`, `!` | Very Positive | 0,6411 | Colab, GPU T4, 2026-11-06 |
| *"The movie was boring and far too long."* | `The`, `movie`, `was`, `bor`, `##ing`, `and`, `far`, `too`, `long`, `.` | Negative | 0,8546 | Colab, GPU T4, 2026-11-06 |

## 2. Exercise 2: Model Đã Fine-Tune

```bash
python -m exercise_2.demo
```

Lệnh này tải mốc `checkpoints/best` và ghi kết quả vào mục `demo` của `reports/metrics.json`. Danh sách câu thử nằm trong `src/exercise_2/demo.py`.

| Câu thử | Dự đoán | Độ tin cậy | Nhãn thật | Đúng/Sai |
| :--- | :---: | :---: | :---: | :---: |
| *"This movie was an absolute masterpiece! The acting and plot were brilliant."* | POSITIVE | 98,03% | POSITIVE | Đúng |
| *"Waste of time. The acting was terrible and the storyline made no sense."* | NEGATIVE | 98,49% | NEGATIVE | Đúng |
| *"It was an average film, neither great nor awful."* | NEGATIVE | 91,48% | Không có | Chỉ quan sát |

Câu thứ ba mang tính trung tính, trong khi model chỉ có 2 lớp, nên câu này không có nhãn đúng và không tính đúng/sai. Model vẫn trả lời NEGATIVE với độ tin cậy 91,48%, cho thấy độ tin cậy cao không có nghĩa là câu trả lời đáng tin khi gặp câu nằm ngoài 2 lớp.
