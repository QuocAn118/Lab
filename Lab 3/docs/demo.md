# Demo — Lab 3

## 1. Module A: pretrained đa ngôn ngữ

Từ repo gốc, dùng Python trong môi trường đã cài requirements:

```powershell
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' infer --text 'Bộ phim này rất hay!' --text 'The movie was boring.' --output 'Lab 3/outputs/module-a'
```

Trả token, nhãn và score model năm lớp. Mỗi lần lưu cần output mới. Score không đảm bảo chính xác mọi câu tiếng Việt.

## 2. Module B: nạp checkpoint fine-tune

```powershell
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' demo --checkpoint 'Lab 3/outputs/full/checkpoints/best'
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' demo --checkpoint 'Lab 3/outputs/full/checkpoints/best' --text 'I loved this movie.'
```

Model B huấn luyện IMDB **tiếng Anh**, hai lớp. Câu trung tính vẫn phải chọn một lớp; không dùng nhãn NEUTRAL làm đáp án cho bài nhị phân. Ba câu demo nằm trong `lab3.py::DEMO_TEXTS`; kết quả thật lưu trong `metrics.json` của run.

## 3. Nguyên tắc báo cáo

Đã bỏ bảng 99.2%/98.7% minh họa cũ vì không có bằng chứng đo. Không điền kết quả trước khi chạy. `demo` nạp model/tokenizer từ đĩa, báo lỗi khi thiếu checkpoint hoặc câu rỗng.

## Nguồn và phạm vi

- [lab_3_description.md](lab_3_description.md): hai module.
- [model.md](model.md), [implementation.md](implementation.md): ngôn ngữ, nhãn và R-01/R-08.
- [sources.md](sources.md): model cards; [AI_Usage.md](AI_Usage.md): không bịa số đo.
