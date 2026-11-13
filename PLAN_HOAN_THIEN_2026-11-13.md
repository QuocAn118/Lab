# Kế hoạch hoàn thiện trước khi nộp — 2026-11-13

Người giao việc: HoangDang. Người thực hiện: một AI agent làm việc trực tiếp trong repo này.
Bản kế hoạch dựa trên đợt rà soát ngày 2026-11-13, ở commit `735e556`, nhánh `hoangdang`.

## 0. Quy tắc bắt buộc — đọc trước khi làm

1. Đọc [AGENTS.md](AGENTS.md), [Lab 3/AGENTS.md](Lab%203/AGENTS.md) và [README.md](README.md) trước khi sửa bất cứ thứ gì.
2. **Không sửa, xóa hay di chuyển artifact thực nghiệm đã có**, gồm:
   - `Lab 1/outputs/`, `Lab 1/checkpoints/`
   - `Lab 2/reports/sessions/`, `Lab 2/reports/tv3/`, `Lab 2/reports/imported-*`
   - `Lab 3/outputs/`, `Lab 3/report/`, `Lab 3/reports/`
   - `BRANCH_SYNC_2026-11-11.*`, `COMPLETION_REVIEW_2026-11-12.md`
3. Không bịa số liệu. Nếu một bước chưa chạy được thì ghi "chưa chạy", kèm lý do và lỗi gặp phải.
4. Mọi số liệu ghi vào tài liệu phải lấy từ output thật của lần chạy đó.
5. Trên máy này pytest có thể báo `PermissionError` ở thư mục `%TEMP%\pytest-of-Legion`. Khi đó luôn thêm `--basetemp <thư mục trong .venv>`. Đây là giới hạn môi trường, không phải lỗi code.
6. Làm trên nhánh `hoangdang`. Mỗi task một commit riêng. Không push, không tạo PR khi chưa được HoangDang đồng ý.
7. Mỗi tài liệu sửa xong phải cập nhật mục 0: dòng `Last updated` (ISO 8601, +07:00) và dòng `Updated by`.
8. Riêng Lab 3: khi thêm hoặc đổi tên file, cập nhật [12_traceability.md](Lab%203/docs/12_traceability.md) (quy tắc 4 trong Lab 3/AGENTS.md).

## 1. Hiện trạng đã kiểm

| Lab | Test | Việc còn lại |
| --- | --- | --- |
| Lab 1 | 20/20 đạt | Không còn việc. Chỉ chạy lại ở bước kiểm cuối |
| Lab 2 | Chưa chạy lại: máy không có `.venv` cho Lab 2 | Dựng môi trường và chạy test. Notebook Colab chưa Run all (TASK-16 vẫn là TODO). `Lab 2.ipynb` nặng 9 MB |
| Lab 3 | 45/45 đạt (cần `PYTHONPATH=src`) | Có hai notebook trùng tên `Lab_3 (1).ipynb`, dễ mở nhầm |

## 2. Các task

### T1 — Lab 2: dựng môi trường và chạy test offline

Từ thư mục `Lab 2/`, chạy trong PowerShell:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'; $env:PYTHONUTF8 = '1'
& .venv/Scripts/python.exe -m pytest tests test.py --ignore=tests/test_smoke.py -q --basetemp .venv/pytest-temp
```

- **Mục tiêu:** 93 passed, đúng như [Lab 2/docs/verification.md](Lab%202/docs/verification.md).
- **Nếu số khác:** không sửa con số trong tài liệu cho khớp. Ghi đúng kết quả thật và báo lại.
- **Nếu có test fail:** dừng task này, chép nguyên output lỗi vào báo cáo. Không tự sửa code khi chưa tìm ra nguyên nhân gốc.
- **Ghi kết quả:** thêm một mục "Kiểm lại 2026-11-13" vào cuối `Lab 2/docs/verification.md`. Ghi số test, thời gian chạy, phiên bản Python và torch.
- **Không làm:** không chạy `tests/test_smoke.py` hay bất kỳ lệnh train nào. Các lệnh đó tải dữ liệu và tạo output mới.

### T2 — Lab 2: giảm dung lượng `notebooks/Lab 2.ipynb`

- **Nguyên nhân:** gần như toàn bộ 9 MB nằm ở `cells[23].metadata.colab`. Cell 23 là ô `%tensorboard --logdir runs`, và phần metadata này là widget Colab bị nhúng vào. Các output thật của notebook chỉ khoảng 24 KB.
- **Việc cần làm:** dùng một script Python nhỏ xóa đúng key `metadata.colab` của cell 23. Giữ nguyên `outputs`, `source` và metadata của mọi cell khác. Ghi file lại bằng `json.dump(..., ensure_ascii=False, indent=1)` và thêm dòng trống cuối file như bản gốc.
- **Kiểm tra sau khi sửa:**
  - File nhỏ hơn 100 KB.
  - `json.load` đọc được file.
  - `nbformat.validate` không báo lỗi, nếu máy có `nbformat`.
  - Danh sách outputs của mọi cell giống hệt trước khi sửa (so sánh bằng `json.dumps`).
- **Ghi chú vào tài liệu:** thêm một dòng vào `Lab 2/docs/plan_progress_status.md`, ghi rằng đã xóa metadata widget Colab của cell TensorBoard và không đổi output.

### T3 — Lab 2: Run all notebook Colab (cần NGƯỜI làm)

AI không tự chạy được Colab T4. Agent chỉ chuẩn bị, rồi xác minh sau khi người dùng gửi kết quả về.

**T3a. Agent viết hướng dẫn ngắn cho người dùng.** Đặt trong báo cáo cuối, không tạo file mới. Hướng dẫn gồm:

1. Upload thư mục `Lab 2` lên Google Drive.
2. Mở `notebooks/Lab_2_colab.ipynb`, chọn GPU T4.
3. Điền ba biến ở ô đầu: đường dẫn project, họ tên người chạy, mã phiên. Mã phiên chưa được trùng với phiên nào có sẵn trong `reports/sessions/`, ví dụ `colab-t4-20261113-<giờ>`.
4. Chạy Run all.
5. Lưu notebook có output, rồi tải về thư mục `reports/sessions/<SESSION_ID>/` và file `configs/sessions/<SESSION_ID>.yaml`.

Có thể dẫn link [Lab 2/docs/training_handoff.md](Lab%202/docs/training_handoff.md), vì file này đã có hướng dẫn chi tiết.

**T3b. Khi người dùng đã đưa kết quả vào repo:**

1. Chạy verifier bàn giao theo hướng dẫn trong `Lab 2/docs/verification.md`.
2. Đọc `reports/sessions/<SESSION_ID>/reports/metrics.json` để lấy KPI tốc độ T4.
3. Cập nhật TASK-13, TASK-15, TASK-16 trong `Lab 2/docs/plan_progress_status.md`, cùng phần trạng thái ở `Lab 2/README.md` và `README.md` gốc. Chỉ ghi "Đạt" cho những gì có bằng chứng trong phiên đó.

**T3c. Nếu người dùng báo không chạy Colab:**

1. Giữ TASK-16 là TODO.
2. Thêm vào đầu phần trạng thái của `Lab 2/README.md` một câu nói rõ: notebook Colab chưa Run all; kết quả nghiệm thu là phiên `rerun-thanhbinh-20261112-175308` chạy trên RTX 5060; KPI tốc độ T4 chưa đo.

### T4 — Lab 3: tách tên hai notebook trùng nhau

Hiện có hai file cùng tên:

- **Bản nộp chính:** `Lab_3 (1).ipynb` ở thư mục gốc. Có 8 cell, đã chạy đủ, không lỗi. `lab3.py`, `reporting.py` và `test.py` đều đọc file này.
- **Bản của nhánh Quốc An:** `Lab 3/notebooks/Lab_3 (1).ipynb`. Có 13 cell, chưa chạy. Không có code nào đọc file này.

**Việc cần làm:**

1. Đổi tên bằng lệnh:
   ```
   git mv "Lab 3/notebooks/Lab_3 (1).ipynb" "Lab 3/notebooks/Lab_3_quocan.ipynb"
   ```
2. Sửa các tham chiếu đang dùng. Chỉ có hai chỗ, đã grep:
   - `Lab 3/docs/evaluation.md:105`
   - `Lab 3/docs/06_verification.md:146`
3. Grep lại toàn bộ chuỗi `notebooks/Lab_3 (1)`. Bỏ qua các thư mục `report/`, `outputs/`, `.kilo/` và file `BRANCH_SYNC_*`, vì đó là bản ghi lịch sử, không sửa.
4. Thêm một dòng vào `Lab 3/docs/12_traceability.md` để ghi lại việc đổi tên.
5. Trong `Lab 3/README.md`, đoạn "Bản bổ sung từ nhánh QuocAn": ghi tên file notebook mới.

**Kiểm tra sau khi làm (chạy từ `Lab 3/`):**

```powershell
$env:PYTHONPATH='src'; & .venv/Scripts/python.exe -m pytest test.py test_reporting.py tests -q --basetemp .venv/pytest-temp
& .venv/Scripts/python.exe lab3.py audit
```

- pytest phải ra 45 passed.
- `lab3.py audit` phải PASS. Lệnh này chỉ đọc, hãy kiểm tra trong code trước khi chạy. Nếu nó ghi file ra `report/` hay `outputs/` thì chỉ chạy pytest, và báo lại điều đó.

### T5 — Lab 3: ghi rõ bộ tài liệu không đánh số là của nhánh Quốc An

Lưu ý: các file `data.md`, `demo.md`, `evaluation.md`, `model.md`, `training.md`, `plan_progress_status.md`, `lab_3_description.md` không phải bản trùng thừa. Đây là bộ tài liệu của nhánh Quốc An, và README đã giải thích. Chỉ cần thêm chú thích để người đọc mở thẳng file không bị nhầm.

**Việc cần làm:**

- Thêm đúng một dòng ngay dưới bảng mục 0 của 7 file trên:
  > Tài liệu của bản triển khai nhánh QuocAn (`src/exercise_*`, `notebooks/Lab_3_quocan.ipynb`). Tài liệu chính của bài là bộ `01_`–`12_`.
- Nếu không làm T4 thì giữ tên notebook cũ trong câu trên.

**Không làm:** không đổi tên, không xóa, không gộp các file này. Lịch sử đồng bộ đang link tới chúng.

### T6 — Kiểm tra cuối

1. Chạy lại test của cả ba lab:
   - Lab 1: dùng `Lab 1/.venv`, lệnh `pytest -q --basetemp ...`. Mục tiêu 20 passed.
   - Lab 2: như T1.
   - Lab 3: như T4.
2. Kiểm tra link tương đối trong mọi file `.md` mà Git đang theo dõi, bỏ qua `report/`, `outputs/`, `reports/`, `source_snapshot*`. Không được có link hỏng mới.

   Lưu ý cách viết regex: tên file có chứa dấu ngoặc, ví dụ `Lab_3%20(1).ipynb`. Regex `\(([^)]+)\)` sẽ cắt link sai và báo lỗi giả.
3. Chạy `git status`. Chỉ được có thay đổi trong các file mà T1–T5 cho phép sửa.
4. Thêm mục "Kiểm lại 2026-11-13" vào cuối `COMPLETION_REVIEW_2026-11-12.md`. Ghi số test của từng lab, những task đã làm, và những task chưa làm kèm lý do. Không sửa các mục cũ trong file.

## 3. Báo cáo trả về cho HoangDang

Báo cáo cuối cùng viết đúng theo khung sau:

```
## Kết quả
| Task | Trạng thái (Xong / Chưa / Bỏ qua) | Bằng chứng (lệnh + kết quả, file thay đổi) |
| T1..T6 | ... | ... |

## Số test
Lab 1: x/20 · Lab 2: x/93 · Lab 3: x/45

## Commit đã tạo
<hash> <message>

## Việc người dùng phải làm
(hướng dẫn chạy Colab ở T3a, nếu chưa có kết quả T4 GPU)

## Rủi ro / chưa kiểm chứng
...
```

## 4. Ngoài phạm vi — không làm

- Train lại hoặc tạo output thực nghiệm mới cho bất kỳ lab nào.
- Sửa các mục "Created date: Chưa xác nhận". Không có nguồn xác nhận ngày tạo thì không được bịa.
- Dọn hoặc viết lại lịch sử Git để giảm dung lượng `.git` (7,2 GB).
- Đóng gói file zip để nộp. Chỉ làm khi HoangDang yêu cầu riêng, và khi đó phải loại `.venv`, `data/`, `.git`.
