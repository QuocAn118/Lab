# Cấu Hình Huấn Luyện (Training Specification) — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Cấu Hình Huấn Luyện (Training Specification) — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by — nguồn code training/test | Nguyễn Quốc An (Quoc An), `773a5a1`, `e2b6402` |
| Thời gian cập nhật nguồn code | 2026-11-10T09:42:35+07:00 (AuthorDate của `e2b6402`) |
| Updated by — tài liệu/tích hợp | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Cấu hình cơ sở

**Ghi nhận phần triển khai:** Nguyễn Quốc An đưa `src/train.py` và `tests/test_train.py` vào repo tại `773a5a1`, sau đó chỉnh theo đặc tả tại `e2b6402`. Registry nguồn tại commit này ghi code training có hỗ trợ Claude Code (Claude Opus 5.5). Các commit lấy thư mục Lab 1 từ HoangDang hoặc lấy model từ Vân Anh là thao tác tích hợp, không chuyển tác giả các phần đó sang Quốc An. Cập nhật dữ liệu tại `eb86ab8` (Thanh Binh theo Git) và hoàn thiện pipeline tại `e36cc00` (HoangDang) được ghi riêng; người chạy các thí nghiệm ngày 2026-11-11 là HoangDang, hỗ trợ OpenAI Codex. Bản ghi nhận bổ sung ngày 2026-11-12 do HoangDang/OpenAI Codex đối chiếu lịch sử, không phải lần train mới.

Theo [đặc tả gốc §6](lab_1_description.md), `src/train.py` dùng MLP `[512, 512]`, `ToTensor`, batch 64, `CrossEntropyLoss`, SGD (lr 0.001, momentum 0.9), seed 42; chọn CUDA nếu có, nếu không CPU.

- Normal run: tối đa 10 epoch trên 54.000 train / 6.000 validation / 10.000 test.
- Smoke run: 1 epoch trên 512 train / 256 validation / 256 test, giới hạn sau split. Khởi tạo trọng số mới cho mỗi run.
- Sau mỗi epoch: ghi train/val loss và accuracy. Loss tính trung bình có trọng số theo số mẫu; accuracy lưu tỷ lệ 0–1.
- Early stopping: so val accuracy với mức tốt nhất trước đó; tăng ít nhất 0.01 thì reset bộ đếm, ngược lại tăng bộ đếm; dừng sau 2 epoch liên tiếp không đủ cải thiện. Mọi cải thiện dương vẫn cập nhật `best_val_acc`.

## 2. Chạy và kiểm tra

Chạy từ `Lab 1`, sau khi chuẩn bị dữ liệu theo [data/README.md](../data/README.md):

```powershell
python -m pytest tests/test_train.py -q
python src/train.py --run-id smoke_001 --purpose "Kiểm tra pipeline"
```

Kiểm thử train dùng DataLoader tensor giả; không thay cho smoke run tích hợp với FashionMNIST thật. `--run-id` và `--purpose` bắt buộc. Chọn mã mới nếu `outputs/<run_id>/` hoặc `checkpoints/<run_id>/` đã tồn tại.

Normal run chỉ thực hiện sau khi smoke đạt đầy đủ §5.1 của đặc tả gốc và các thay đổi trong Lab 1 đã commit. Các lệnh dự kiến:

```powershell
python src/train.py --run-id run_001 --purpose "Cấu hình cơ sở"
python src/train.py --run-id run_002 --purpose "Lớp ẩn [256,128]" --hidden-dims 256 128
python src/train.py --run-id run_003 --purpose "Learning rate 0.01" --lr 0.01
python src/train.py --run-id run_004 --purpose "Adam lr 0.001" --optimizer Adam
```

Các lệnh này mô tả cách chạy, chưa phải bằng chứng đã chạy thành công.

## 3. Checkpoint và đầu ra

`checkpoints/<run_id>/` có 5 tệp `state_dict`: `final_epoch.pth`, `best_val_loss.pth`, `best_val_acc.pth`, `best_val_pause.pth`, `best.pth`, cùng `checkpoint_index.json`. Chính sách lưu theo [§6.1](lab_1_description.md); `best` là bản sao checkpoint có val accuracy cao nhất, hòa giữ epoch sớm hơn. `best_val_pause` ghi trạng thái tại lần validation cuối và cờ `early_stopped`.

`outputs/<run_id>/` có config, history, reload_check, metrics, predictions, performance, loss_curve, prediction_samples, verify_report và manifest. Cột lịch sử: `epoch, train_loss, train_accuracy, val_loss, val_accuracy, epoch_seconds`. `epoch_seconds` và `elapsed_train_seconds` chỉ tính train, loại validation và kiểm tra tải lại.

## 4. Trạng thái triển khai

Smoke_001 và run_001–run_004 đã hoàn tất, đủ manifest/verify_report và kiểm hash PASS. Train kiểm tra final/best trên 16 ảnh validation, đánh giá test, đo performance, tạo đồ thị/demo và verify_report trước khi chốt manifest. Nghiệm thu còn yêu cầu smoke và các normal run thật đạt kiểm hash. Xem [3.4](requirements/3_4_train.md), [3.6](requirements/3_6_save_load.md) và [tiến độ](plan_progress_status.md).


## 5. Thực thi đã hoàn tất

Đã chạy đủ bốn lệnh normal ở mục 2, lần lượt 6/8/4/4 epoch, đều dừng sớm theo patience 2 và min_delta 0.01. Không chỉnh epoch/learning rate sau khi xem test. Xem [evaluation.md](evaluation.md) cho số đo và [verification.md](verification.md) cho lệnh/nguồn kiểm chứng. Mã run đã tồn tại không dùng lại; chọn mã mới cho lần tái lập.
