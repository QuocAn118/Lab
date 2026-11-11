# Lab 3 — Phân tích cảm xúc với Transformers

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Lab 3 — Phân tích cảm xúc với Transformers |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Bản thực hành chính: **`../Lab_3 (1).ipynb`**. Notebook giữ các ô code chính viết tuần tự. Tiện ích thời gian/lưu model/vẽ loss được tách vào `notebook_helpers.py`; không dùng pipeline `lab3.py`.

Tài liệu trong `docs/` được đánh số theo thứ tự đọc. Bắt đầu từ [Hướng dẫn trình bày](docs/huong_dan_trinh_bay.md), sau đó đọc 01–07 cho nội dung bài; 08–12 là tài liệu triển khai, tiến độ và nguồn để tra cứu thêm.

## Bản bổ sung từ nhánh QuocAn — 2026-11-11

Đã nhận thêm `src/exercise_1/`, `src/exercise_2/`, `configs/exercise_2.yaml`, `tests/`, notebook trong `notebooks/`, checkpoint trong `checkpoints/` và báo cáo trong `reports/`. Đây là bản triển khai riêng của nhánh QuocAn; không thay thế notebook gốc, pipeline `lab3.py`, báo cáo `report/` hoặc dữ liệu thực nghiệm trong `outputs/`.

Tài liệu của bản này là các file không đánh số trong `docs/`, bắt đầu từ [lab_3_description.md](docs/lab_3_description.md). Danh sách thư viện gốc được giữ ở `requirements_quocan.txt`; môi trường chính tiếp tục dùng `requirements.txt` đã khóa phiên bản. Chưa kiểm chứng chạy chung hai bản. Các checkpoint mới được quản lý bằng Git LFS; xem trạng thái tải trong nhật ký đồng bộ.

Người tích hợp: HoangDang, hỗ trợ OpenAI Codex. Tác giả, thời gian commit gốc, thời gian tích hợp và cách xử lý conflict được ghi trong [nhật ký đồng bộ](../BRANCH_SYNC_2026-11-11.md).

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

Giữ `Lab 3/notebook_helpers.py` đi kèm notebook khi sao chép.

Môi trường được mô tả trong báo cáo cũ không mặc nhiên có trên máy mới. Python đang kiểm tra ngày 2026-11-11 thiếu torch/transformers/datasets. Khối lệnh dưới tái hiện cấu hình PyTorch CUDA 12.8 của máy GPU được ghi trong báo cáo; nếu dùng phần cứng khác, chọn bản PyTorch phù hợp trước khi cài các dependency còn lại. Chạy từ repo gốc:

```powershell
$env:PYTHONUTF8='1'
python -m venv 'Lab 3/.venv'
& 'Lab 3/.venv/Scripts/python.exe' -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& 'Lab 3/.venv/Scripts/python.exe' -m pip install -r 'Lab 3/requirements.txt'
& 'Lab 3/.venv/Scripts/python.exe' -m ipykernel install --user --name lab3 --display-name 'Python (Lab 3)'
```

Dataset/model được tải từ Hugging Face ở lần đầu. Các báo cáo trước đây dùng đường dẫn `outputs/full/`. Snapshot nhập từ nhánh local hiện nằm trong `outputs/imported-thanhbinh-40acced/full/`; xem `IMPORT.json` để ánh xạ. Không mặc định notebook tự tìm snapshot ở vị trí mới: kiểm tra đường dẫn model trước khi Run All; nếu không tìm được checkpoint, code có thể huấn luyện mới. CPU có thể chạy chậm hơn GPU.

`lab3.py`, `test.py` và các manifest full là phần hỗ trợ nâng cao đã có; không cần đọc hoặc chạy chúng để theo dõi nội dung notebook rút gọn.

## Thời gian huấn luyện

Mỗi run mới có `training_time.json`: bắt đầu/kết thúc, tổng số giây, trạng thái và thời lượng từng epoch. Checkpoint lưu thông tin thời gian tại lúc lưu; `best` giữ bản tổng kết. Nạp model cũ chỉ đọc thời gian đã có. Xem [cách ghi và đọc thời gian](docs/04_training.md).

## Nguồn và phạm vi

- [Đặc tả](docs/01_lab_3_description.md), [thiết kế notebook rút gọn](docs/08_implementation.md), [training](docs/04_training.md).
- [Nguồn và khai báo AI](docs/11_AI_Usage.md), [nguồn kỹ thuật](docs/10_sources.md), [kiểm chứng](docs/06_verification.md).
- Mỗi ô code có nguồn Markdown. Artifact mới có SOURCES.md đi kèm; các run full đã có giữ manifest lịch sử.
