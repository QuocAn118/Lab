# 3.4. Huấn luyện: forward, loss, backward, optimize

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | 3.4. Huấn luyện: forward, loss, backward, optimize |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-07T11:17:41+07:00 ; HoangDang ; d41a9fe |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Trạng thái: `src/train.py` đã có; smoke_001 và bốn normal run đã hoàn tất, kiểm hash PASS. Code kiểm tra final/best, gọi evaluation/performance/đồ thị/demo và verifier trước khi chốt manifest.

## 1. Mục đích

Huấn luyện mô hình trên tập train, ghi lại cấu hình và diễn biến học để phục vụ đánh giá và so sánh thí nghiệm.

## 2. File thực hiện

- File code: `src/train.py`.
- Gọi `create_dataloaders(config)` từ `data/data_pipeline.py` (ba loader train/val/test) và `build_model(model_config)` từ `src/model.py`.
- Gọi chức năng lưu và kiểm tra tải lại từ `src/model_io.py` ở bước 3.6.
- Sử dụng `CrossEntropyLoss` và optimizer SGD cho cấu hình cơ sở.

## 3. Đầu vào

- Dữ liệu đã chuẩn bị ở bước 3.1; DataLoader và mô hình do bước 3.2–3.3 tạo.
- Tham số dòng lệnh `--run-id` và `--purpose` bắt buộc. Mã gồm `run_` hoặc `smoke_` và ba chữ số; `purpose` mô tả mục đích trong manifest.
- Cấu hình cơ sở dự kiến: 10 epoch, batch size 64, learning rate 0.001, SGD với momentum 0.9, seed 42, MLP có hai lớp ẩn `[512, 512]` và ReLU.
- Chọn thiết bị CUDA nếu có, nếu không dùng CPU; ghi lại thiết bị thực tế trong cấu hình.
- Đây là cấu hình khởi đầu của bài làm, không phải cấu hình bắt buộc do đề quy định.
- Smoke: 1 epoch, 512 train / 256 validation / 256 test sau split; normal: 54.000 / 6.000 / 10.000. Test không tham gia train hoặc chọn checkpoint.

## 4. Các bước xử lý

1. Kiểm tra `run_id` hợp lệ và cả `outputs/<run_id>/` lẫn `checkpoints/<run_id>/` chưa tồn tại. Nếu đã có, dừng và yêu cầu chọn mã mới; không ghi đè thí nghiệm.
2. Tạo hai thư mục của lần chạy; thiết lập seed và ghi cấu hình thực tế trước khi huấn luyện.
3. Khởi tạo DataLoader, mô hình, hàm mất mát và optimizer; chuyển mô hình và các batch sang cùng thiết bị.
4. Đặt mô hình ở chế độ huấn luyện. Với mỗi batch: xóa gradient cũ → forward → tính loss → backward → optimizer cập nhật tham số.
5. Cuối mỗi epoch, tính loss trung bình có trọng số theo số mẫu và accuracy của các dự đoán đã thu trong quá trình huấn luyện epoch đó.
6. Cuối mỗi epoch, chạy validation ở chế độ eval, tắt gradient; ghi train/val loss và accuracy rồi flush lịch sử. Chỉ cập nhật `best_val_loss` khi loss thấp hơn và `best_val_acc` khi accuracy cao hơn; hòa giữ checkpoint sớm hơn.
7. Tăng bộ đếm nếu val accuracy cải thiện dưới 0.01 so với mức tốt nhất trước epoch; tăng ít nhất 0.01 thì reset. Dừng khi bộ đếm đạt 2 hoặc đủ số epoch tối đa.
8. Lưu `best_val_pause` tại lần validation cuối cùng với cờ `early_stopped`; sao chép `best_val_acc` thành `best`; lưu `final_epoch`. Ghi metadata trong `checkpoint_index.json`.
9. Kiểm tra tải lại final/best trên 16 ảnh validation; best được so với bản sao trọng số tốt nhất giữ trong bộ nhớ.
10. Gọi `evaluate_run(config, run, test_loader, summary)`, tạo đầy đủ metrics, performance, dự đoán, đồ thị và demo; kiểm số liệu và ghi verify_report trước khi thoát Run. Mọi thao tác được chốt manifest kể cả khi lỗi hoặc Ctrl+C.

Không dùng tập test để cập nhật tham số hoặc chọn checkpoint. Pipeline CLI hiện tích hợp các bước này; chỉ chốt `completed` khi báo cáo số liệu trước manifest đạt. Sau đó chạy verifier chỉ đọc để kiểm lại hash toàn bộ run.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Đường dẫn | Định dạng và thời điểm ghi |
| --- | --- | --- |
| Cấu hình lần chạy | `outputs/<run_id>/config.json` | JSON UTF-8; ghi trước epoch đầu bằng thư viện JSON. |
| Lịch sử huấn luyện | `outputs/<run_id>/history.csv` | CSV UTF-8 có dòng tiêu đề; tạo trước huấn luyện, thêm một dòng sau mỗi epoch. |
| Trọng số cuối | `checkpoints/<run_id>/final_epoch.pth` | Do chức năng bước 3.6 lưu sau epoch cuối. |
| Các checkpoint còn lại | `checkpoints/<run_id>/{best_val_loss,best_val_acc,best_val_pause,best}.pth` | `state_dict`, theo chính sách §6.1 của đặc tả gốc. |
| Metadata checkpoint | `checkpoints/<run_id>/checkpoint_index.json` | Mỗi checkpoint có `epoch`, `val_loss`, `val_accuracy`, `saved_at`, `elapsed_train_seconds`; `best_val_pause` thêm `early_stopped`. |
| Kết quả kiểm tra tải lại | `outputs/<run_id>/reload_check.json` | Do chức năng bước 3.6 ghi sau khi kiểm tra. |
| Hồ sơ lần chạy | `outputs/<run_id>/manifest.json` | Do `Run` ghi người chạy, cấu hình, nguồn, môi trường, trạng thái và hash đầu ra. |

Các trường của `config.json`:

- `schema_version`: số nguyên 1; `run_id`: chuỗi mã lần chạy; `created_by`, `started_at` do `Run` thêm.
- `dataset`: chuỗi `FashionMNIST`; `classes`: danh sách tên lớp theo chỉ số nhãn.
- `model`: đối tượng gồm `name`, `input_shape`, `hidden_dims`, `activation`, `num_classes` như bước 3.3.
- `data`: đối tượng gồm `transforms`, `batch_size`, `num_workers`, `train_shuffle`, `test_shuffle`, `drop_last`, `limits` theo bước 3.2; `limits=null` cho normal, `{train:512,val:256,test:256}` cho smoke. Validation luôn không shuffle.
- `training`: đối tượng gồm `epochs`, `seed`, `loss`, `optimizer`, `learning_rate`, `momentum`, `device`.

Các cột của `history.csv`: `epoch` (bắt đầu từ 1), `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `epoch_seconds`. Accuracy lưu theo tỷ lệ từ 0 đến 1; loss là tổng loss trung bình từng batch nhân số mẫu batch, chia tổng số mẫu. `epoch_seconds` và `elapsed_train_seconds` chỉ tính train, không tính validation hoặc kiểm tra tải lại.

Nếu bị gián đoạn, giữ lại cấu hình và các epoch đã ghi; chưa có checkpoint cuối thì lần chạy được xem là chưa hoàn thành. Chạy lại từ đầu phải chọn `run_id` mới; chức năng tiếp tục huấn luyện chưa nằm trong quy trình cơ sở.

## 6. Cách chạy và kiểm tra

Lệnh smoke từ thư mục `Lab 1` sau khi chuẩn bị dữ liệu; đã chạy smoke_001 đạt toàn pipeline; chọn run_id mới nếu tái lập:

```powershell
python src/train.py --run-id smoke_001 --purpose "Kiểm tra pipeline"
```

- Có `config.json` ghi đúng các giá trị thực tế đã sử dụng.
- Normal run có 1–10 dòng dữ liệu tùy early stopping; smoke mặc định có 1 dòng; không có giá trị NaN hoặc vô hạn.
- Accuracy nằm trong `[0, 1]`; loss hữu hạn và không âm. Không coi loss phải giảm ở mọi epoch là điều kiện bắt buộc.
- Có đủ 5 checkpoint và index; `best` trùng byte với `best_val_acc`; báo cáo kiểm tra tải lại thành công theo bước 3.6. Để nghiệm thu đầy đủ còn phải kiểm tra `best` theo đặc tả gốc.
- Chạy với cùng `run_id` bị từ chối trước khi sửa dữ liệu lần chạy cũ.
- `python -m pytest tests/test_train.py -q` dùng tensor giả để kiểm tra train, early stopping, 5 checkpoint và từ chối mã trùng. Kiểm thử này không thay thế smoke run dữ liệu thật.
- Normal run yêu cầu các thay đổi trong Lab 1 đã commit và smoke toàn pipeline đạt; ví dụ `python src/train.py --run-id run_001 --purpose "Cấu hình cơ sở"`.
