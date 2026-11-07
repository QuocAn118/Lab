# Lab 3 — Phân tích cảm xúc với Transformers

Bản thực hành chính: **`../Lab_3 (1).ipynb`**. Notebook chỉ gồm các ô code viết tuần tự, dùng API thư viện trực tiếp và không phụ thuộc `lab3.py`.

Tài liệu trong `docs/` được đánh số theo thứ tự đọc. Bắt đầu từ [Hướng dẫn trình bày](docs/huong_dan_trinh_bay.md), sau đó đọc 01–07 cho nội dung bài; 08–12 là tài liệu triển khai, tiến độ và nguồn để tra cứu thêm.

## Nội dung

**Exercise 1:** pretrained sentiment analysis, trong một ô code chạy độc lập.

**Exercise 2:** đúng 7 bước, mỗi bước một ô code:

1. Chuẩn bị thư viện.
2. Tải và chia dữ liệu IMDB.
3. Tải model và tokenizer.
4. Tokenize dữ liệu.
5. Cấu hình huấn luyện.
6. Tạo Trainer và huấn luyện.
7. Đánh giá, confusion matrix, loss curve và demo.

## Cách chạy

Mở notebook, chọn kernel **Python (Lab 3)**, chạy lần lượt hoặc **Run All**. Nếu đã có checkpoint full, mặc định nạp model để đánh giá; đặt `TRAIN=True` trong ô khai báo model để huấn luyện mới. Mỗi run mới được lưu riêng trong `Lab 3/outputs/simple-training/`.

Trong VS Code, chọn **Select Kernel → Python Environments… → .venv (3.12.10)** có đường dẫn **`Lab 3\.venv\Scripts\python.exe`**. Tên hiển thị có thể khác tên kernel đăng ký. Chạy **Run All**; khi chỉ chạy Exercise 2, bắt đầu từ **Step 1**. Mục **Existing Jupyter Server…** dành cho kết nối server có URL, không cần cho môi trường cục bộ này.

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

- [Đặc tả](docs/01_lab_3_description.md), [thiết kế notebook rút gọn](docs/08_implementation.md), [training](docs/04_training.md).
- [Nguồn và khai báo AI](docs/11_AI_Usage.md), [nguồn kỹ thuật](docs/10_sources.md), [kiểm chứng](docs/06_verification.md).
- Mỗi ô code có nguồn Markdown. Artifact mới có SOURCES.md đi kèm; các run full đã có giữ manifest lịch sử .
