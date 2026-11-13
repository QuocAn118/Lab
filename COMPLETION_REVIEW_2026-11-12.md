# Rà soát và hoàn thiện ba lab — 2026-11-12

HoangDang thực hiện theo yêu cầu hoàn thiện Lab 1, Lab 2 và Lab 3. Tác giả nguồn và các lần tích hợp trước được đối chiếu trong Git, registry Lab 1 và hồ sơ nhập từng phiên.

| Lab | Kiểm chứng | Kết quả bài thực hành | Phần đã hoàn thiện trong lượt này |
| --- | --- | --- | --- |
| Lab 1 | 17 test, strict provenance, verifier của smoke và bốn normal run đều đạt | Model chọn bằng validation có test accuracy 87,70% trên 10.000 ảnh theo run đã lưu | Sửa trạng thái tài liệu và mô tả khả năng nạp checkpoint best, ghi lần rà trong registry |
| Lab 2 | 85 test và ba smoke thật của code hiện hành, bốn cấu hình đã train và đánh giá trên RTX 5060 | DenseNet121 test 87,68%, macro F1 0,8755 trên 5.000 ảnh | Chụp giao diện TensorBoard mới, kiểm nguồn event/hash và cập nhật liên kết bàn giao |
| Lab 3 | 43 test, source audit, smoke pretrained mới và Run All tám cell đạt | Checkpoint full đã có được đánh giá lại trên 25.000 văn bản test, accuracy 91,012%, macro F1 0,910118 | Sửa chọn checkpoint sau nhập nhánh, nối tiện ích thời gian/report, đọc loss history cùng checkpoint, sửa nguồn và dependency, thêm chặn ghi đè trong bản Quốc An |

## Lab 1

Kiểm thử lại đạt 17/17 trong 22,77 giây. `smoke_001`, `run_001`, `run_002`, `run_003`, `run_004` đều được verifier chỉ đọc xác nhận PASS. `check_provenance.py --strict` có 0 lỗi và 0 cảnh báo. Các số đo, checkpoint và thời điểm train nằm trong manifest ngày chạy tương ứng.

[Kiểm chứng Lab 1](Lab%201/docs/verification.md), [đánh giá bốn cấu hình](Lab%201/docs/evaluation.md), [registry nguồn](Lab%201/docs/provenance.json).

## Lab 2

Đã lưu ảnh giao diện TensorBoard với đủ bốn run, Accuracy/val, Loss/train và Loss/val. Smoothing 0, hiển thị đầy đủ các điểm dữ liệu. Bốn file event khớp hash manifest của phiên `rerun-thanhbinh-20261112-175308`.

[Ảnh TensorBoard](Lab%202/reports/sessions/tensorboard-review-20261112-191218/tensorboard.png), [biên bản bổ sung](Lab%202/reports/sessions/tensorboard-review-20261112-191218/REVIEW.json), [so sánh bốn cấu hình](Lab%202/reports/sessions/rerun-thanhbinh-20261112-175308/comparison.md).

Phiên `colab-t4-20261112-1920` đã được cấp Tesla T4 và bắt đầu chạy notebook từ code `c6f4d54`. Kết quả đầy đủ, notebook có output và checkpoint của phiên này còn chờ tiếp nhận. Hai KPI tốc độ T4 tiếp tục chờ xác minh trên artifact thực tế của phiên T4.

## Lab 3

Notebook dùng câu mẫu có thể sửa trong Exercise 1. Exercise 2 mặc định `TRAIN=False`, tìm checkpoint có trọng số và phát hiện pointer LFS. Model/tokenizer cùng được nạp từ thư mục checkpoint đã chọn. Các tiện ích report và thời gian từ `eb6bfad` được nối lại. Loss history được đọc theo đường dẫn checkpoint nhập.

Bộ kiểm thử đạt 43/43 trong 11,01 giây. Notebook tám cell hoàn tất lúc 2026-11-12T20:43:18+07:00. Đánh giá lại 25.000 test đạt accuracy 0,91012, macro F1 0,9101182914 và loss 0,3465386629. Smoke mới dùng hai optimizer step trên 32/16/16 mẫu, xác minh đủ năm checkpoint và 48 artifact. Source audit đạt 40 file và tám cell.

[Báo cáo đánh giá mới](Lab%203/report/evaluate-20261112-204226-805293/summary.md), [suy luận đa ngôn ngữ](Lab%203/report/infer-20261112-204208-788129/summary.md), [smoke mới](Lab%203/report/cli-smoke-20261112-204427-088939/summary.md), [JUnit](Lab%203/report/completion-review-20261112/offline-tests.xml).

Đã đối chiếu đủ 176 file của snapshot nhập, gồm 20 trọng số, với `IMPORT.json`. Manifest full lịch sử có 18 file khớp byte trực tiếp và 27 file text khớp sau khôi phục kiểu xuống dòng khi tạo manifest. Verifier byte nghiêm ngặt của bản lịch sử báo khác hash tại 27 file đó. Biên bản kiểm tra mới ghi từng phép đối chiếu và hash của bản đang lưu.

[Đối chiếu bản nhập](Lab%203/report/completion-review-20261112/intake_check.json), [đối chiếu manifest lịch sử](Lab%203/report/completion-review-20261112/historical_manifest_check.json).

## Giới hạn kiểm chứng

Lượt này đánh giá lại model full Lab 3 đã có và chạy smoke mới. Chưa chạy full training Lab 3 mới hoặc benchmark Lab 3 trên T4. Số đo GPU RTX được đọc theo môi trường đã lưu. Kết quả T4 của Lab 2 cần lấy từ phiên Colab trước khi kết luận về tốc độ hoặc Run All trên Colab.

## Kiểm lại sau commit `25cc678`

HoangDang, chạy lại lúc 2026-11-12T21:30+07:00. Commit `25cc678` thêm test chặn ghi đè nên số test tăng: Lab 1 **20/20**, Lab 2 **93/93** offline, Lab 3 **44/44** (cần `PYTHONPATH=src`). Strict provenance Lab 1, verifier năm run Lab 1, verifier bàn giao Lab 2 và source audit Lab 3 đều PASS. Các số liệu ở bảng trên là của lượt rà trước, giữ nguyên. Chi tiết và môi trường: [Lab 1](Lab%201/docs/verification.md), [Lab 2](Lab%202/docs/verification.md), [Lab 3](Lab%203/docs/06_verification.md). Kết quả Colab T4 của Lab 2 vẫn chờ.

## Nguồn và phạm vi

- Đặc tả, tiến độ, kiểm chứng và artifact được liên kết cho từng lab.
- [Quy tắc repo](AGENTS.md), [quy tắc Lab 3](Lab%203/AGENTS.md), [truy nguồn Lab 3](Lab%203/docs/12_traceability.md).
- Các sửa đổi và kiểm chứng ngày 2026-11-12 do HoangDang thực hiện. Hồ sơ nguồn của Vân Anh, Thanh Bình, Quốc An, Xuân Bảo và các khai báo AI đã có tiếp tục nằm trong tài liệu và lịch sử tương ứng.


## Kiểm lại 2026-11-13

Last updated: 2026-11-13T08:59:13+07:00

Updated by: AI agent (thực thi kế hoạch của HoangDang)

Chỉ bổ sung mục này theo T6; toàn bộ nội dung cũ phía trên được giữ nguyên byte. Thực hiện trên nhánh `hoangdang`, nền `735e556`; không push hoặc tạo PR. Tác giả và artifact nguồn giữ nguyên, người thực hiện lượt kiểm này được ghi riêng.

| Task | Trạng thái | Bằng chứng và giới hạn |
| --- | --- | --- |
| T1 | Chưa | Đã tạo `.venv` Lab 2, cài requirements và `pip check` PASS. Python 3.12.10; torch distribution 2.15.0, torchvision 0.29.2. Pytest bị Windows chặn DLL (`WinError 4551`): 9 lỗi thu thập trong 5,26 giây, chưa thực thi test nào. Không sửa code hoặc chính sách Windows. Output lỗi đầy đủ của lần kiểm lại T6 và thông tin T1 được bổ sung cuối `Lab 2/docs/verification.md`; không đổi số 93 trong lịch sử. |
| T2 | Xong | `Lab 2/notebooks/Lab 2.ipynb` còn 42.278 byte. Chỉ xóa `cells[23].metadata.colab`; JSON, `nbformat.validate`, so sánh outputs bằng `json.dumps` và so sánh source đều PASS. Nhật ký tại `Lab 2/docs/plan_progress_status.md`. |
| T3 | Chưa | T3a: hướng dẫn Colab trong báo cáo bàn giao cuối, không tạo file mới. T3b chờ người dùng gửi notebook có output, config và thư mục phiên Colab T4 đã hoàn tất; chưa kiểm verifier/KPI của phiên mới. Người dùng chưa thông báo bỏ Colab nên không áp dụng T3c. TASK-16 và KPI T4 giữ trạng thái chờ. |
| T4 | Xong | `git mv` notebook Quốc An thành `Lab 3/notebooks/Lab_3_quocan.ipynb`, Git nhận rename 100%, blob hash không đổi; notebook chính không đổi. Sửa hai tham chiếu hiện hành, README và traceability. Pytest 45 passed, 4 warnings, 14,94 giây; audit PASS 40 file/8 cell. |
| T5 | Xong | Thêm đúng một dòng chú thích dưới bảng mục 0 của 7 tài liệu không đánh số theo kế hoạch; cập nhật Last updated/Updated by, giữ nội dung và nguồn lịch sử. |
| T6 | Chưa | Đã chạy lại test cả ba lab, audit, kiểm link và phạm vi Git; Lab 2 vẫn bị chặn nên chưa xác nhận toàn bộ test đạt. Kết quả chi tiết bên dưới. |

### Lệnh và kết quả kiểm cuối

Dùng `.venv/Scripts/python.exe` tương ứng, chạy từ thư mục từng lab, `PYTHONUTF8=1`, `PYTHONPATH=src`. Các lệnh được gọi bằng subprocess để giữ nguyên output, exit code và thư mục làm việc; log tạm nằm trong `.venv` Lab 2, không đưa vào Git.

| Lab | Lệnh (sau `.venv/Scripts/python.exe`) | Kết quả thực tế |
| --- | --- | --- |
| Lab 1 | `-m pytest tests data/tests -q --basetemp .venv/pytest-temp` | **20 passed**, 7,99 giây; exit 0 |
| Lab 2 | `-m pytest tests test.py --ignore=tests/test_smoke.py -q --basetemp .venv/pytest-temp` | **Chưa chạy test**: 9 lỗi thu thập, 1,38 giây; exit 2; Windows Application Control chặn DLL của torch |
| Lab 3 | `-m pytest test.py test_reporting.py tests -q --basetemp .venv/pytest-temp` | **45 passed**, 4 warnings, 15,19 giây; exit 0 |

Lab 3: `.venv/Scripts/python.exe lab3.py audit` PASS lần kiểm cuối, 40 file và 8 cell code. Đã kiểm tra code của nhánh audit là chỉ đọc. Bộ pytest Lab 3 theo lệnh được giao bao gồm smoke hai batch của bản Quốc An và các fixture model nhỏ; không chạy lệnh CLI train hoặc Run All notebook, không tạo run thực nghiệm mới. Cảnh báo về mtime checkpoint và backend ảnh chỉ phát sinh trong test.

Kiểm link bằng `& 'Lab 2/.venv/Scripts/python.exe' 'Lab 2/.venv/verify_links.py'` từ repo gốc: quét mọi Markdown được Git theo dõi, bỏ qua `report/`, `reports/`, `outputs/`, `source_snapshot*`; dùng parser Markdown và HTML để xử lý ngoặc cân bằng, URL-encoding và liên kết tham chiếu. Self-check có tên `Lab_3%20(1).ipynb` và dạng `<Lab_3 (1).ipynb>`. Trước và sau thay đổi: **93 file, 440 liên kết tương đối, 0 link hỏng, 0 link hỏng mới**. Phạm vi kiểm là đường dẫn file/thư mục, không kiểm fragment tiêu đề hoặc URL mạng.

Kiểm phạm vi Git: chỉ thay đổi notebook/tài liệu được T1, T2, T4, T5 và phần bổ sung T6 cho phép. `PLAN_HOAN_THIEN_2026-11-13.md` là file chưa theo dõi đã có từ đầu; giữ nguyên, không stage. Không sửa/xóa/di chuyển artifact trong các thư mục được bảo vệ, nhật ký BRANCH_SYNC hoặc phần cũ của báo cáo này. Không đóng gói ZIP, không sửa ngày tạo chưa xác nhận, không đổi lịch sử Git.

### Phần còn lại

- T1 và điều kiện toàn bộ test đạt của T6: chưa hoàn thành do lỗi nạp DLL dưới Windows Application Control; chưa xác định dependency DLL cụ thể bị chặn. Môi trường Lab 2 mới vẫn được giữ để kiểm tra tiếp, không coi `pip check` thành công là torch import thành công.
- T3: người dùng cần chạy `Lab 2/notebooks/Lab_2_colab.ipynb` trên T4, dùng `PROJECT_DIR`, `OPERATOR`, `SESSION_ID` mới; lưu notebook có output và bàn giao `reports/sessions/<SESSION_ID>/` cùng `configs/sessions/<SESSION_ID>.yaml`. Chưa có phép đo T4 mới để nghiệm thu.
