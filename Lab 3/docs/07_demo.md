# Demo — Lab 3

## 1. Module A: pretrained đa ngôn ngữ

Từ repo gốc, dùng Python trong môi trường đã cài requirements:

```powershell
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' infer --text 'Bộ phim này rất hay!' --text 'The movie was boring.' --output 'Lab 3/outputs/module-a'
```

Trả token, nhãn và score model năm lớp. Mỗi lần lưu cần output mới. Score không đảm bảo chính xác mọi câu tiếng Việt.

## 2. Module B: nạp checkpoint fine-tune

Notebook Exercise 1 có thêm 10 câu theo năm cặp Việt/Anh để khảo sát tích cực, tiêu cực, trung tính, phủ định và cảm xúc trái chiều. Chạy riêng ô đầu; bảng kết quả được sinh từ model thật khi thực thi. Nhãn kỳ vọng chỉ là đối chiếu chủ quan; câu trái chiều để mở. Không tính accuracy từ bộ ví dụ này. Hợp đồng chi tiết: [08_implementation.md](08_implementation.md), phần bổ sung Exercise 1 ngày 2026-11-07.


```powershell
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' demo --checkpoint 'Lab 3/outputs/full/checkpoints/best'
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' demo --checkpoint 'Lab 3/outputs/full/checkpoints/best' --text 'I loved this movie.'
```

Model B huấn luyện IMDB **tiếng Anh**, hai lớp. Câu trung tính vẫn phải chọn một lớp; không dùng nhãn NEUTRAL làm đáp án cho bài nhị phân. Ba câu demo nằm trong `lab3.py::DEMO_TEXTS`; kết quả thật lưu trong `metrics.json` của run.

## 3. Nguyên tắc báo cáo

Đã bỏ bảng 99.2%/98.7% minh họa cũ vì không có bằng chứng đo. Không điền kết quả trước khi chạy. `demo` nạp model/tokenizer từ đĩa, báo lỗi khi thiếu checkpoint hoặc câu rỗng.

## Nguồn và phạm vi

- [01_lab_3_description.md](01_lab_3_description.md): hai module.
- [03_model.md](03_model.md), [08_implementation.md](08_implementation.md): ngôn ngữ, nhãn và R-01/R-08.
- [10_sources.md](10_sources.md): model cards; [11_AI_Usage.md](11_AI_Usage.md): không bịa số đo.
