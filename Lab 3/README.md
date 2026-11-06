# Lab 3 — Phân tích cảm xúc với Transformers

Bản thực hành chính: **`../Lab_3 (1).ipynb`**. Notebook viết tuần tự, dùng API thư viện trực tiếp và không phụ thuộc `lab3.py`.

## Nội dung

1. Pretrained sentiment analysis.
2. Chia IMDB thành train/validation/test và tokenize.
3. Fine-tune DistilBERT bằng Trainer và early stopping có sẵn.
4. Đánh giá, confusion matrix, loss curve và demo.

## Cách chạy

Mở notebook, chọn kernel **Python (Lab 3)**, chạy lần lượt hoặc **Run All**. Nếu đã có checkpoint full, mặc định nạp model để đánh giá; đặt `TRAIN=True` tại phần 5 để huấn luyện mới. Mỗi run mới được lưu riêng trong `Lab 3/outputs/simple-training/`.

Môi trường hiện tại đã được chuẩn bị. Khi cài trên Windows mới, chạy từ repo gốc:

```powershell
$env:PYTHONUTF8='1'
python -m venv 'Lab 3/.venv'
& 'Lab 3/.venv/Scripts/python.exe' -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& 'Lab 3/.venv/Scripts/python.exe' -m pip install -r 'Lab 3/requirements.txt'
& 'Lab 3/.venv/Scripts/python.exe' -m ipykernel install --user --name lab3 --display-name 'Python (Lab 3)'
```

Dataset/model được tải từ Hugging Face ở lần đầu. Checkpoint và loss history full hiện có nằm trong `outputs/full/`; giữ thư mục này khi muốn dùng lại kết quả. Nếu chưa có checkpoint, notebook thực hiện huấn luyện mới. CPU có thể chạy chậm hơn GPU.

`lab3.py`, `test.py` và các manifest full là phần hỗ trợ nâng cao đã có; không cần đọc hoặc chạy chúng để theo dõi nội dung notebook rút gọn.

## Nguồn và phạm vi

- [Đặc tả](docs/lab_3_description.md), [thiết kế notebook rút gọn](docs/implementation.md), [training](docs/training.md).
- [Nguồn và khai báo AI](docs/AI_Usage.md), [nguồn kỹ thuật](docs/sources.md), [kiểm chứng](docs/verification.md).
- Mỗi ô code có nguồn Markdown. Artifact mới có SOURCES.md đi kèm; các run full đã có giữ manifest lịch sử.
