# Truy vết Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Created date | 2026-11-11 |
| Created by | HoangDang |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |

| Yêu cầu | Code | Phép kiểm | Bằng chứng đích |
| --- | --- | --- | --- |
| 3.1–3.2: data/split/loader | `data/prepare_data.py`, `data/data_pipeline.py` | `data/tests/test_data.py`, hash/split | `data/reports/`, dataset raw |
| 3.3: MLP | `src/model.py` | Shape, tham số, logits hữu hạn, sidecar | `outputs/model_summary.txt` |
| 3.4: train/checkpoint | `src/train.py` | `tests/test_train.py` | history, config, 5 checkpoint/index |
| 3.5: test/performance | `src/evaluate.py` | Giá trị biết trước, loader rỗng, logits sai, đối chiếu CSV | metrics, predictions, performance |
| 3.6: save/load | `src/model_io.py` | Nạp đúng final/best, không ghi đè, so logits | reload_check |
| Demo/đồ thị/so sánh | `src/demo.py`, `src/evaluate.py`, `src/compare_runs.py` | Pipeline smoke, ảnh cố định seed 42 | loss_curve, prediction_samples, compare_runs/compare_loss |
| Toàn vẹn | `src/verify.py`, `src/run_tracking.py`, `tools/check_provenance.py` | Sửa CSV phải bị phát hiện; kiểm manifest/hash sau run | verify_report, manifest, sidecar |

Các dòng trên đã được kiểm chứng qua 20 test, smoke và bốn normal run; kết quả thật được ghi tại [verification.md](verification.md).

## Nguồn và phạm vi

- [Đặc tả gốc](lab_1_description.md), [toàn vẹn](team_plan_integrity.md), các file trong `requirements/`.

## Artifact thực tế

- [Smoke manifest](../outputs/smoke_001/manifest.json).
- [Run 001](../outputs/run_001/manifest.json), [002](../outputs/run_002/manifest.json), [003](../outputs/run_003/manifest.json), [004](../outputs/run_004/manifest.json).
- [So sánh](../outputs/compare_runs.csv), [ảnh input](../outputs/demo_input.png), [JSON output](../outputs/demo_example.json).
