# Kế hoạch bàn giao train — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Kế hoạch bàn giao train — Lab 2 |
| Created date | 2026-11-11T11:06:40+07:00 |
| Created by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Last updated | 2026-11-11T11:13:30+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Chuẩn bị code và bàn giao; chưa huấn luyện thật hoặc nghiệm thu KPI |

## 1. Phạm vi và người làm

Người chuẩn bị: HoangDang, hỗ trợ OpenAI Codex. Người train: **chưa chỉ định**; bạn nhận việc điền `OPERATOR` trong cell đầu notebook. Không suy tên thật từ TV1–TV5 hoặc tên nhánh.

Đã chuẩn bị code, notebook và kiểm tra offline. Người nhận chạy thực nghiệm theo đúng config, lưu bằng chứng và bàn giao kết quả. Không phải viết thêm pipeline. Chưa có cam kết accuracy hoặc thời gian chạy; không cần giải nén ZIP.

## 2. Nhận đúng phiên bản

Lấy toàn bộ thư mục Lab 2 từ nhánh `hoangdang` mới nhất, mở `notebooks/Lab_2_colab.ipynb` trên Colab. Bản chuẩn bị dựa trên `main` tại `a7b7ef8`; chưa coi nó đã có trên main cho đến khi PR mới được merge.

Đặt `Lab 2` trong Drive; chỉ cần code Lab 2, không cần tải checkpoint Lab 1/3. Chọn **Runtime → Change runtime type → T4 GPU**. Điền `PROJECT_DIR`, `OPERATOR`, `SESSION_ID` trong cell đầu; giữ cùng mã phiên cho cả 4 run. Chừa **ít nhất 2 GB** Drive cho một phiên: khoảng 540 MB trọng số model, khoảng 350 MB CIFAR-10 sau giải nén cùng archive và phần còn lại cho cache/log/file tạm. Đây là dự trù, chưa đo dung lượng phiên thật.

Notebook dùng config phiên để không sửa config gốc và không ghi đè báo cáo lịch sử. Mỗi run ghi tên người chạy, thời gian có múi giờ, phiên bản thư viện, config và commit Git nếu thư mục còn metadata Git. Nếu upload riêng Lab 2 làm mất metadata Git, ghi commit nguồn vào biên bản bàn giao.

## 3. Thứ tự thực hiện và điều kiện chuyển bước

| Bước | Người làm | Việc cần chạy | Chỉ chuyển bước khi |
| --- | --- | --- | --- |
| 0 | Người train | Điền 3 biến đầu notebook, chọn T4; chạy cell cài thư viện/config phiên | Import được torch; xác nhận GPU và đường dẫn phiên đúng |
| 1 | Người train | Bộ test offline `tests test.py --ignore=tests/test_smoke.py` | Tất cả test pass; lưu output notebook |
| 2 | Người train | `lab_2.data`, `lab_2.model` qua notebook | Báo cáo đúng 22.500/2.500/5.000 ảnh, 10 lớp, input 3×224×224; model head 10 lớp |
| 3 | Người train | `pytest tests/test_smoke.py -q -s` | Cả 3 smoke pass với dữ liệu/trọng số thật; không OOM, gradient hữu hạn, phần đóng băng giữ nguyên |
| 4 | Người train | 4 cell train theo bảng dưới | Mỗi run đủ 3 checkpoint, log và summary; không có lỗi |
| 5 | Người train | `lab_2.evaluation` **một lần** | Đủ 4 run, config khớp; chọn best bằng validation; tạo metrics/KPI/3 biểu đồ |
| 6 | Người train | `lab_2.demo`, mở TensorBoard | Có xác suất/demo; thấy 4 run trong TensorBoard, lưu ảnh chụp |
| 7 | Người train → người tích hợp | Giữ nguyên toàn bộ thư mục phiên, config phiên và notebook đã lưu output | Đối chiếu checklist sản phẩm, không chỉ gửi ảnh accuracy |

### Bốn run bắt buộc, không đổi thứ tự hoặc sửa tham số giữa phiên

| Thứ tự | Run | Model | LR | Batch | Epoch tối đa |
| --- | --- | --- | --- | --- | --- |
| 1 | resnet18_baseline | ResNet18 | 0,001 | 64 | 6 |
| 2 | densenet121 | DenseNet121 | 0,001 | 64 | 6 |
| 3 | resnet18_lr1e-2 | ResNet18 | 0,01 | 64 | 6 |
| 4 | resnet18_bs128 | ResNet18 | 0,001 | 128 | 6 |

Mỗi run dùng AdamW, chỉ train head. Early stopping: tăng **hơn** 1 điểm phần trăm mới tính cải thiện, dừng sau 3 epoch không cải thiện. `best_val_pause` là quy tắc dừng, không có file riêng; ghi `stop_epoch`. Fine-tuning layer4 được kiểm tra bằng smoke, không thêm run thứ 5 vào bảng so sánh này.

## 4. Chạy từng lệnh khi cần xử lý phiên ngắt

Notebook gọi subprocess bằng `check=True`; lệnh lỗi sẽ dừng, không chạy tiếp evaluation với dữ liệu thiếu. Từ `Lab 2/` trên Colab/Linux, thay `<SESSION_ID>` bằng mã phiên đã khai báo:

```bash
export PYTHONPATH=src
export LAB2_OPERATOR="Ho ten nguoi train"
python -m lab_2.training --config "configs/sessions/<SESSION_ID>.yaml" --run resnet18_baseline
python -m lab_2.training --config "configs/sessions/<SESSION_ID>.yaml" --run densenet121
python -m lab_2.training --config "configs/sessions/<SESSION_ID>.yaml" --run resnet18_lr1e-2
python -m lab_2.training --config "configs/sessions/<SESSION_ID>.yaml" --run resnet18_bs128
python -m lab_2.evaluation --config "configs/sessions/<SESSION_ID>.yaml"
python -m lab_2.demo --config "configs/sessions/<SESSION_ID>.yaml"
```

Run đã hoàn thành không chạy lại. Nếu mất kết nối nhưng tiến trình đã xong, kiểm tra JSON/log trước khi chạy cell khác. Nếu run dở có checkpoint/log nhưng chưa có summary, giữ phiên đó và tạo mã phiên mới cho bộ 4 run hoàn chỉnh; code không resume giữa epoch. Không xóa log hoặc đổi config để vượt chặn. Nếu không đạt accuracy, giữ kết quả và ghi Không đạt; không chọn theo tập test hoặc hạ KPI sau khi đo.

## 5. Sản phẩm phải bàn giao

Với `S = reports/sessions/<SESSION_ID>`:

- `configs/sessions/<SESSION_ID>.yaml` và notebook đã lưu output, tên người chạy, commit nguồn, ngày giờ bắt đầu/kết thúc.
- `S/checkpoints/<run>/final_epoch.pth`, `best_val_loss.pth`, `best_val_acc.pth` cho cả 4 run: tổng 12 file; thêm `S/checkpoints/best.pth`: **13 checkpoint**.
- `S/reports/metrics.json`: đủ data, model, training.runs (4 run), evaluation, demo; mỗi run có config/phiên bản/GPU/người chạy/thời gian.
- `S/reports/runs/<run>/events.out.tfevents*` của cả 4 run.
- `S/reports/figures/val_accuracy_by_run.png`, `loss_curve_best.png`, `confusion_matrix.png`, `demo_probabilities.png`, **tensorboard.png** (người train chụp ở cell cuối).
- Nhận xét ngắn Exercise 1/2, lý do run tốt nhất theo validation; ghi từng KPI đạt/không đạt/không đo được và GPU thực tế.

Người tích hợp cập nhật `evaluation.md`, `demo.md` và bảng nghiệm thu từ JSON thật sau khi nhận. Bản notebook không thể tự chụp UI TensorBoard; người train lưu ảnh đó thủ công. Không ghi kết quả giả từ tests vào báo cáo bài nộp.

## 6. Đưa kết quả về codebase

Giữ nguyên thư mục phiên trên Drive; khi bàn giao, chép đúng cấu trúc phiên và config về repo, không dùng ZIP làm sản phẩm chính. Các thư mục phiên được ignore mặc định để tránh stage nhầm phiên dở. Người tích hợp chỉ stage phiên đã review bằng đường dẫn cụ thể; trọng số `.pth` dùng Git LFS theo `.gitattributes` Lab 2, báo cáo/ảnh/notebook dùng Git thường. Không commit data/cache/venv. Push nhánh cá nhân, tạo PR để review vào main; không force-push các nhánh của thành viên.

Trước khi công bố hoàn thành: chạy lại kiểm tra offline trên code bàn giao; đối chiếu 4 run, 13 checkpoint, config và số mẫu test; xác nhận best chọn bằng validation. Giữ các kết quả cũ làm lịch sử, không thay bằng kết quả phiên mới.
