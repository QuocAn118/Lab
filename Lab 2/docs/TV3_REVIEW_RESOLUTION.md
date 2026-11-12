# Xử lý review TV3 — Lab 2

Ngày: 2026-11-12. Thực hiện: HoangDang. Chỉ thay đổi Lab 2; giữ nguyên dữ liệu thực nghiệm đã hoàn tất.

## 1. Checkpoint trên GitHub

Đưa đủ **12 checkpoint** ở `checkpoints/<run>/` vào Git LFS, theo quy tắc `.gitattributes` đã có trên main. Không thay đổi byte trọng số; kích thước và SHA-256 đều khớp [checkpoint_check.json](../reports/tv3/checkpoint_check.json) đã lập ngày 2026-11-11. Không tạo `best.pth` hoặc kết quả test mới.

Sau pull nhánh này, lấy riêng trọng số Lab 2 từ gốc repo:

```powershell
git lfs pull origin ThanhBinh --include="Lab 2/checkpoints/**"
python "Lab 2/tools/verify_tv3_handoff.py"
```

## 2. Hash training.py khác manifest

Manifest train đã ghi SHA-256 `30c283c0941c369039f3557acffbae3107111b1e4c133de9541c222577bee358`. Sau train, dòng chú thích `Spec: docs/training.md mục 8` đổi thành `Spec: docs/training_specification.md mục 8` khi tách báo cáo và đặc tả. Đây là nguyên nhân khác byte; không phải chỉ khác xuống dòng/BOM.

Khôi phục dòng chú thích cũ và đối chiếu SHA-256 cho kết quả khớp **toàn bộ file**. Bản khớp được lưu nguyên byte tại [training.py đã dùng cho bốn run](../reports/tv3/source_snapshots/20261111-123519/src/lab_2/training.py). Không sửa manifest cũ để hợp thức hóa code mới.

Đã khôi phục đủ 7 file nguồn/config cho mỗi lượt CPU, GPU và train: **21 file**, tất cả khớp hash trong ba manifest. [Biên bản đối chiếu](../reports/tv3/source_reconciliation_2026-11-12.json) ghi đường dẫn, SHA-256 và phép khôi phục. Snapshot là bản đối chiếu lịch sử, không phải package đang dùng. Không gán một commit mới thành commit đã dùng cho train cũ.

Manifest trên Git được giữ nguyên blob LF; Windows checkout trước đó có CRLF. Biên bản ghi cả hash blob đã công bố và hash checkout đã quan sát để audit chạy được ở cả hai máy. Điều này tách biệt với source snapshot: từng file source phải khớp chính xác hash runtime, không chuẩn hóa hash source để bỏ qua sai khác.

`tools/run_tv3.py` từ nay lưu cả byte source vào `sources/` cùng hash ngay khi bắt đầu lượt chạy, tránh chỉ có hash mà không còn source.

## 3. Ghép các module với main

Nền đối chiếu: `origin/main` tại `1224a67`. Chỉ lấy file Lab 2, không merge hoặc sửa Lab 1/3 trong lần xử lý này.

- Giữ nguyên `common.py`, data của Vân Anh, model package, evaluation và demo từ main; giữ tác giả và thông tin xác nhận trong tài liệu main.
- Giữ `model.py` entry point tương thích và notebook Colab đã hoàn thiện trên main; bỏ adapter cũ ở package.
- Dùng training trên main: không ghi đè run/log/checkpoint, seed Python/NumPy, lưu config/người chạy/thời gian/commit cho run mới.
- Giữ thêm các kiểm tra TV3: tên và đường dẫn run an toàn, log interval dương, validation metrics hữu hạn.
- Giữ report, CSV, TensorBoard, checkpoint và source snapshot lịch sử của bốn run. Không tuyên bố các run này dùng code tích hợp mới hoặc đủ schema phiên mới; thiếu `config` trong summary cũ không được bổ sung bằng suy đoán. Evaluation mới sẽ yêu cầu config khớp cho phiên mới.

Các kiểm thử bổ sung TV3 được chỉnh theo chính sách **giữ nguyên log cũ**, thay cho hành vi xóa log cũ. Kiểm thử gốc của nhóm được lấy từ main.

## 4. Kiểm chứng

- Audit chỉ đọc: `tools/verify_tv3_handoff.py` xác nhận 21 source khớp, 3 manifest nguyên byte và 12 checkpoint khớp.
- Kiểm thử offline sau ghép: **80 passed in 71.17s**; lệnh `python -m pytest tests test.py --ignore=tests/test_smoke.py -q`, dùng .venv Lab 2 và PYTHONPATH=src. Dữ liệu/model trong test nằm ở thư mục tạm.
- Lượt kiểm chứng có log/source snapshot mới được lưu dưới `reports/tv3/20261112-110855/`; đọc manifest để phân biệt kiểm chứng code mới với train lịch sử.
- Lượt có log này đạt **80 passed in 13.56s** và **3 smoke passed in 39.01s** trên RTX 4060 Laptop; không phải chạy lại bốn normal run.
- Không train lại bốn run, không đánh giá test thật trong lần sửa review này.
