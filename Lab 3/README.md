# Lab 3 — Phân tích cảm xúc với Transformers

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Lab 3 — Phân tích cảm xúc với Transformers |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Bản chính và tài liệu bắt đầu

**Bộ docs/01_–docs/12_ là bản chính.** Notebook thực hành là [Lab_3 (1).ipynb](Lab_3%20%281%29.ipynb), nằm ngay trong Lab 3. Cấu hình full: 20.000 train / 5.000 validation / 25.000 test; seed 42, max length 256, batch 16, LR 2e-5, tối đa 3 epochs. Validation chọn model; test chỉ đánh giá cuối.

Đọc [đặc tả chính](docs/01_lab_3_description.md), [hướng dẫn trình bày](docs/docs%20demo/huong_dan_trinh_bay.md), rồi bộ 01–07. Các file 08–12 mô tả triển khai, tiến độ, nguồn và truy vết. Exercise 1 có một ô code độc lập; Exercise 2 có bảy ô theo thứ tự chuẩn bị thư viện → dữ liệu → model → tokenize → tham số → Trainer/train → đánh giá/demo. Các tiện ích thời gian/report nằm trong notebook_helpers.py và reporting.py; notebook không gọi pipeline lab3.py.

## 2. Chuẩn bị môi trường

Môi trường đã kiểm chứng: Python 3.12.10, torch 2.9.1+cu128, transformers 4.57.1, datasets 4.3.0, pandas 2.3.3, pyarrow 21.0.0. Phiên bản được ghi ở requirements.txt và requirements-lock.txt. Ví dụ cài mới từ repo gốc trên Windows:

```powershell
$env:PYTHONUTF8='1'
python -m venv 'Lab 3/.venv'
& 'Lab 3/.venv/Scripts/python.exe' -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
& 'Lab 3/.venv/Scripts/python.exe' -m pip install -r 'Lab 3/requirements.txt'
& 'Lab 3/.venv/Scripts/python.exe' -m ipykernel install --user --name lab3 --display-name 'Python (Lab 3)'
git lfs install
git lfs pull
```

Trong VS Code, mở notebook trong Lab 3, chọn Python Environments → môi trường Lab 3/.venv/Scripts/python.exe hoặc kernel Python (Lab 3). Chạy Run All; nếu chỉ chạy Exercise 2, bắt đầu từ Step 1. Tên hiển thị có thể là .venv (3.12.10). Existing Jupyter Server dành cho server có URL, không phải cách chọn môi trường cục bộ.

Dataset/model cần tải ở lần đầu. Cài môi trường mới là hướng dẫn thiết lập, không phải tuyên bố đã cài lại trong lần chuẩn hóa tài liệu.

## 3. Đánh giá model đã có hoặc huấn luyện mới

Mặc định TRAIN=False. available_checkpoint kiểm tra outputs/full/checkpoints/best, sau đó outputs/imported-thanhbinh-40acced/full/checkpoints/best. Bản sao hiện tại có trọng số đầy đủ ở snapshot nhập; outputs/full chỉ còn một phần metadata. Thiếu trọng số hoặc còn pointer LFS thì báo lỗi và yêu cầu tải LFS, không tự chuyển sang train.

Đặt TRAIN=True trong Step 3 để huấn luyện mới, chạy từ Step 1. Mỗi run có thư mục riêng trong outputs/simple-training/; Trainer giữ tối đa hai checkpoint theo epoch, xuất best và ghi training_time.json. Khi đánh giá checkpoint nhập không có timing, notebook báo thiếu log; thời lượng full khôi phục 334,46 giây nằm riêng tại outputs/full/training_time.json và timing_sources.md. Không coi thời gian đánh giá là thời gian train.

Giữ notebook_helpers.py và reporting.py cùng thư mục Lab 3 khi sao chép. Mỗi thao tác infer/train/evaluate có report riêng; đường dẫn được in lúc chạy. [Danh mục report](report/README.md) mô tả từng file. CLI demo ghi demo.json trong report; CLI train ghi cả metrics và nguồn artifact ở outputs.

## 4. Kiểm chứng và kết quả đã lưu

Ngày 2026-11-12, notebook tám ô đã chạy trên checkpoint full nhập từ commit 40acced, đánh giá lại đủ 25.000 test: accuracy **91,012%**, macro F1 **0,9101182914**, loss **0,3465386629**. Biểu đồ loss dùng lịch sử của checkpoint cũ. Đây là phép đánh giá lại, không phải full training mới. [Metrics](report/evaluate-20261112-204226-805293/metrics.json), [summary](report/evaluate-20261112-204226-805293/summary.md).

Sau chuẩn hóa vị trí file ngày 2026-11-14: **46 test đạt**, audit **41 file/8 ô code**, **212 liên kết nội bộ** không hỏng và **687 file bằng chứng/notebook** giữ nguyên hash. Smoke CLI ngày 2026-11-12 chạy hai optimizer steps và có 48 artifact được verifier đối chiếu. Chi tiết và giới hạn tại [06_verification.md mục 8](docs/06_verification.md). Không dùng score của bản Quốc An thay số liệu bản chính; chưa có full training hoặc benchmark T4 mới của bản chính trong đợt này.

Các lệnh kiểm tra từ repo gốc:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONPATH=(Resolve-Path 'Lab 3/src').Path
& 'Lab 3/.venv/Scripts/python.exe' -m pytest 'Lab 3/test.py' 'Lab 3/test_reporting.py' 'Lab 3/tests' -q --basetemp 'Lab 3/.venv/pytest-docs-temp'
& 'Lab 3/.venv/Scripts/python.exe' 'Lab 3/lab3.py' audit
& 'Lab 3/.venv/Scripts/python.exe' -m pip check
```

tests/test_smoke.py dùng model/dataset pretrained thật; cần chúng trong cache hoặc có mạng. Model ngẫu nhiên nhỏ trong các test khác chỉ kiểm cơ chế, không phải model bàn giao.

## 5. Bản Quốc An tham khảo

Bảy tài liệu không đánh số được đặt riêng trong [docs/docs demo/](docs/docs%20demo/README.md), mô tả src/exercise_*, configs/exercise_2.yaml, checkpoints/ và reports/. Notebook Lab_3_colab.ipynb điều phối pipeline này; Lab_3_quocan.ipynb là notebook kế thừa để tham khảo, không phải bản nộp chính. Các file requirements_quocan.txt giữ danh sách môi trường nhập; môi trường chính dùng requirements.txt.

Kết quả T4 lịch sử của bản Quốc An dùng 1.000/200/200 và tối đa 5 epochs, test accuracy 87%; không trộn với kết quả 91,012% của bản chính. Các lệnh Quốc An mặc định bị chặn khi checkpoint/report đã có; notebook Colab lịch sử không bảo đảm dừng khi !python lỗi. Cách dùng phiên mới và giới hạn đã biết được ghi trong tài liệu tham khảo.

Nguồn commit, tác giả và người tích hợp được giữ tại [nhật ký đồng bộ](../BRANCH_SYNC_2026-11-11.md) và IMPORT.json của snapshot. Việc sắp xếp tài liệu không phải lần train hoặc nghiệm thu lại các pipeline lịch sử.

## Nguồn và phạm vi

- [Đặc tả](docs/01_lab_3_description.md), [dữ liệu](docs/02_data.md), [huấn luyện](docs/04_training.md), [đánh giá](docs/05_evaluation.md).
- [Triển khai §11](docs/08_implementation.md), [kiểm chứng](docs/06_verification.md), [truy vết](docs/12_traceability.md), [khai báo AI](docs/11_AI_Usage.md).
- Ngày 2026-11-14 chuẩn hóa tài liệu và đường dẫn theo vị trí file đã sắp xếp; giữ tác giả nguồn, output notebook và artifact lịch sử.
