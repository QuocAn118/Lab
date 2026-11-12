# TV3 — Hướng dẫn huấn luyện Lab 2

Cập nhật: 2026-11-11. OpenAI Codex hỗ trợ theo yêu cầu người dùng; cần thành viên phụ trách review trước khi bàn giao.

**Cập nhật 2026-11-12:** Các mục mô tả adapter và xóa log bên dưới là thiết kế lịch sử trước ghép. Code hiện tại dùng module main; từ chối ghi đè log/run cũ. Checkpoint đã đưa vào Git LFS. Dùng [biên bản review](TV3_REVIEW_RESOLUTION.md) và [kế hoạch phiên mới](training_handoff.md) khi bàn giao; không chạy lại run cũ.

## Cấu trúc

| Vị trí | Nội dung |
| --- | --- |
| `src/lab_2/training.py` | Code chính TV3: train, scheduler, early stopping, checkpoint, TensorBoard |
| `src/lab_2/common.py`, `data.py` | Phụ thuộc tối thiểu bổ sung theo đặc tả để ghép TV3 |
| `src/lab_2/model.py` | Adapter dùng nguyên `Lab 2/model.py` của Xuân Bảo |
| `src/lab_2/evaluation.py` | Code TV4 kế thừa; TV3 chỉ dùng `evaluate_model` trên validation |
| `configs/lab_2.yaml` | Cấu hình bốn run, không đổi số mẫu/siêu tham số |
| `tests/test_training.py` | Sáu test gốc, giữ nguyên |
| `tests/test_training_standalone.py` | Năm test bổ sung cho vòng điều phối |
| `tests/test_smoke.py` | Ba smoke test thật với pretrained và CIFAR-10 |
| `tools/run_tv3.py` | Chạy kiểm chứng/train, lưu log và thông tin nguồn |
| `tools/export_tv3.py` | Xuất so sánh, kiểm tra event/checkpoint, tạo báo cáo TV3 |
| `tools/capture_tensorboard.py` | Chụp giao diện TensorBoard thật bằng Edge headless |
| `reports/tv3/` | Log, manifest và bảng so sánh |
| `reports/runs/<run>/` | TensorBoard event |
| `checkpoints/<run>/` | Ba checkpoint của mỗi run, giữ trên máy |

## Môi trường và cách chạy

Từ `Lab 2`, PowerShell:

```powershell
python -m venv --system-site-packages .venv
& .venv/Scripts/python.exe -m pip install -r requirements-tv3-cuda.txt
$env:PYTHONPATH = 'src'
& .venv/Scripts/python.exe tools/run_tv3.py --stage verify
# Chỉ chạy train khi verify đạt và chưa có run cùng tên.
& .venv/Scripts/python.exe tools/run_tv3.py --stage train
& .venv/Scripts/python.exe tools/export_tv3.py
```

Cặp wheel đã chạy: torch 2.6.0+cu124 / torchvision 0.21.0+cu124, GPU RTX 4060 Laptop trên Windows. Đây là môi trường cục bộ, không phải Colab T4. Máy CPU dùng `requirements.txt`; không dùng kết quả CPU để nghiệm thu KPI GPU.

`data/` chứa CIFAR-10 gốc và được Git bỏ qua. Bản tải thực tế dùng URL mirror đã có trong notebook cũ; MD5 archive đã xác nhận `c58f30108f718f92721af3b95e74349a`. Loader vẫn dùng torchvision để kiểm tra và mở dataset. Split 22.500/2.500/5.000, seed 42; không dùng test chọn checkpoint.

## Luồng và hàm chính

| Thành phần | Chức năng |
| --- | --- |
| `is_better`, `select_best_run` | Ưu tiên validation accuracy cao, hòa thì loss thấp |
| `EarlyStopping` | Reset khi accuracy tăng hơn 0.01 so với mốc theo dõi; dừng sau ba epoch không đủ cải thiện |
| `find_run` | Kiểm tra tên run và báo các lựa chọn hợp lệ |
| `train_one_epoch` | Train mode theo TV2 → forward → loss → backward → optimizer; log mỗi 50 batch |
| `train_run` | Train → validation → checkpoint → TensorBoard → scheduler → early stopping → summary |

AdamW chỉ nhận tham số trainable. `set_train_mode` giữ BatchNorm đóng băng ở eval. Train loss là trung bình theo batch đúng đặc tả. GPU được synchronize trước/sau train để đo thời gian; thời gian validation không cộng vào train_seconds.

Checkpoint mỗi run: `final_epoch.pth`, `best_val_loss.pth`, `best_val_acc.pth`. TV2 cung cấp hàm lưu nguyên tử. Checkpoint accuracy và early stopping dùng hai tiêu chí riêng; một cải thiện nhỏ có thể cập nhật checkpoint nhưng không reset patience.

TensorBoard ghi `Loss/train`, `Loss/val`, `Accuracy/val`, `LearningRate`. Chạy lại qua `train_run` xóa log cũ đúng run sau khi kiểm tra đường dẫn; công cụ `run_tv3.py --stage train` chủ động từ chối ghi đè run có sẵn để bảo toàn bằng chứng. Report giữ nguyên các mục và run khác.

## Bằng chứng kiểm tra

- CPU: `reports/tv3/20261111-122708/` — 41 test đạt, ba smoke test đạt.
- GPU: `reports/tv3/20261111-123359/` — 41 test đạt, ba smoke test đạt; VRAM lần lượt 0.48/0.68/0.57 GB.
- Train thật: `reports/tv3/20261111-123519/` — đọc `manifest.json` để biết trạng thái; summary chỉ ghi khi run hoàn tất.
- Năm test bổ sung dùng model nhỏ và phụ thuộc thay thế; ba smoke test dùng dataset và pretrained thật. Không đánh đồng hai loại bằng chứng.

Sau khi đủ bốn run, `export_tv3.py` tạo [báo cáo TV3](training.md), `comparison.csv`, kiểm tra checkpoint khớp epoch/metrics và kiểm tra đủ event theo epoch. Các file này lấy số liệu từ run thật.

## Xem TensorBoard và bàn giao

```powershell
& .venv/Scripts/python.exe -m tensorboard.main --logdir reports/runs
# Tùy chọn chụp tự động, cần Microsoft Edge và Playwright:
& .venv/Scripts/python.exe -m pip install playwright
& .venv/Scripts/python.exe tools/capture_tensorboard.py
```

Ảnh chụp lưu tại `reports/figures/tensorboard.png`. Không tạo `best.pth` hoặc chạy đánh giá test trong phần TV3; TV4 tiếp nhận bốn summary và checkpoint để chọn model bằng validation. Notebook Colab và demo vẫn thuộc TV5. Giữ khai báo AI và tác giả model gốc khi bàn giao.
