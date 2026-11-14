# Demo — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Demo — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Module A: pretrained đa ngôn ngữ

Từ repo gốc, dùng Python trong môi trường đã cài requirements:

```powershell
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' infer --text 'Bộ phim này rất hay!' --text 'The movie was boring.' --output 'Lab 3/outputs/module-a-20261114-demo'
```

Trả token, nhãn và score model năm lớp. Mỗi lần lưu cần đổi sang output mới; thư mục ví dụ cũng chỉ dùng một lần. Có thể bỏ --output để chỉ ghi report tự sinh. Score không đảm bảo chính xác mọi câu tiếng Việt.

## 2. Module B: nạp checkpoint fine-tune

```powershell
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' demo --checkpoint 'Lab 3/outputs/imported-thanhbinh-40acced/full/checkpoints/best'
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' demo --checkpoint 'Lab 3/outputs/imported-thanhbinh-40acced/full/checkpoints/best' --text 'I loved this movie.'
```

Model B huấn luyện IMDB **tiếng Anh**, hai lớp. Câu trung tính vẫn phải chọn một lớp; không dùng nhãn NEUTRAL làm đáp án cho bài nhị phân. Ba câu demo nằm trong `lab3.py::DEMO_TEXTS`; kết quả CLI demo lưu trong `report/cli-demo-<run>/demo.json`, còn demo trong CLI train nằm ở `outputs/<run>/metrics.json`. Notebook Step 7 ghi `demo.json` trong report đánh giá. Chương trình in đường dẫn report thực tế.

## 3. Nguyên tắc báo cáo

Đã bỏ bảng 99.2%/98.7% minh họa cũ vì không có bằng chứng đo. Không điền kết quả trước khi chạy. `demo` nạp model/tokenizer từ đĩa, báo lỗi khi thiếu checkpoint hoặc câu rỗng.

## Nguồn và phạm vi

- [01_lab_3_description.md](01_lab_3_description.md): hai module.
- [03_model.md](03_model.md), [08_implementation.md](08_implementation.md): ngôn ngữ, nhãn và R-01/R-08.
- [10_sources.md](10_sources.md): model cards; [11_AI_Usage.md](11_AI_Usage.md): không bịa số đo.
