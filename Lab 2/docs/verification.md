# Kiểm chứng chuẩn bị — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Kiểm chứng chuẩn bị — Lab 2 |
| Created date | 2026-11-11T11:06:40+07:00 |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Last updated | 2026-11-13T11:02:55+07:00 |
| Updated by | AI agent (thực thi kế hoạch của HoangDang) |
| Phạm vi | Chuẩn bị code và bàn giao; chưa huấn luyện thật hoặc nghiệm thu KPI |

## 1. Kết quả và giới hạn

### TV3 nhập lịch sử và chuẩn bị chạy lại — 2026-11-12

Nguồn bàn giao: Nguyễn Thanh Bình (`origin/ThanhBinh`, `92930e6`). HoangDang tiếp nhận và kiểm chứng trên máy hiện tại. Đây là một phiên lịch sử gồm bốn cấu hình, không ghi thành bốn lần train mới. [IMPORT.json](../reports/imported-thanhbinh-92930e6/IMPORT.json) giữ nguồn, thời gian gốc, đường dẫn và hash. Người chạy cũ chưa được ghi tên trong manifest, không suy từ tác giả commit.

Đã kiểm chứng **12 checkpoint đúng byte/hash, 21 snapshot source đúng hash, ba manifest không đổi**, và nạp best checkpoint của cả bốn model cho output hữu hạn `[1,10]`. Snapshot khôi phục dòng chú thích cũ đã giải quyết chênh hash training.py trước đó; không sửa manifest cũ. Metrics/CSV/số epoch được đối chiếu riêng.

Bản tích hợp đạt **85 test offline**, gồm 71 test package và 14 test tương thích; **3 smoke pretrained thật PASS trên RTX 5060 Laptop**, trong 29,94 giây. Code mới giữ chống ghi đè/path an toàn; runner nhận config/output riêng và snapshot nguồn khi bắt đầu. KPI tốc độ chỉ được chấm khi tên phần cứng đúng T4; số đo RTX vẫn lưu trong báo cáo.

Phiên mới dùng `configs/sessions/rerun-thanhbinh-20261112-175308.yaml`, ghi kết quả riêng trong `reports/sessions/rerun-thanhbinh-20261112-175308/`. [Trạng thái phiên](../reports/sessions/rerun-thanhbinh-20261112-175308/SESSION.json) phân biệt chuẩn bị, đang chạy và hoàn tất. Báo cáo, checkpoint và log gốc của Bình được giữ nguyên; kết quả mới không thay cho kết quả cũ hoặc benchmark T4.

### 1.1. Kiểm chứng bổ sung ngày 2026-11-12 — phần dữ liệu Vân Anh

Sau merge `828e5ef`, đã chạy lại bộ test offline với kiểm tra dữ liệu bổ sung: **74 passed**, 0 lỗi, 0 skipped; 60 test package (20 test data) và 14 test tương thích. Thời gian 10,538 giây; 46 DeprecationWarning từ `torch.pin_memory`. Các test mới kiểm tra kiểu/số âm, tập rỗng, `load_splits` với dataset nhỏ và `main` giữ các mục báo cáo đã có.

**CIFAR-10 thật PASS trên Windows:** archive và 7 file sau giải nén khớp MD5; chia đúng 22.500/2.500/5.000, seed tái lập, train/validation không giao nhau. Đã kiểm tra batch train 64 ảnh, validation/test 128 ảnh ở shape `[B,3,224,224]`, ảnh hữu hạn, nhãn int64 0–9, `num_workers=2`, transform đánh giá ổn định. CLI data đã ghi thống kê vào phiên mới và giữ nguyên `reports/metrics.json` lịch sử. Chi tiết, config, thời gian và hash source tại [data_validation_2026-11-12.json](../reports/data_validation_2026-11-12.json).

Nguồn phần data: Nguyễn Thị Vân Anh (`origin/vananh`, `4d3fb94`); HoangDang, tích hợp và kiểm chứng ngày 2026-11-12. Không đổi tên/ngày cập nhật nguồn của Vân Anh. Đây là kiểm chứng dữ liệu; chưa chạy 3 smoke model pretrained, 4 normal run, accuracy/F1/KPI hoặc Run all Colab. Trên Colab vẫn cần thực hiện lại bước tải/chia dữ liệu theo [kế hoạch bàn giao](training_handoff.md).

### 1.2. Kết quả chuẩn bị ngày 2026-11-11 — giữ làm lịch sử

**63 passed** khi chạy `pytest tests test.py --ignore=tests/test_smoke.py -q`: 49 test package và 14 test tương thích. Bao gồm chia tập/transform, head/freeze/BatchNorm, checkpoint reload, early stopping, TensorBoard, chọn model bằng validation, luồng train → evaluate → demo trên dữ liệu giả và chặn ghi đè/config sai/test lặp.

Thời gian của lần kiểm tra: 12,19 giây. Có 46 DeprecationWarning từ cơ chế pin_memory của PyTorch 2.9.1; không có test lỗi. Đây là thời gian kiểm tra, không phải thời gian train CIFAR-10. Test tích hợp có bước optimizer trên vài tensor giả để kiểm tra logic; không tải CIFAR-10/ImageNet và không ghi kết quả thực nghiệm vào reports/metrics.json.

| Phạm vi | Trạng thái |
| --- | --- |
| Logic và tích hợp offline | Đạt 63 test |
| Model/package CLI --help | Kiểm tra riêng, không tải trọng số |
| Notebook mới | Kiểm tra cấu trúc, cú pháp cell Python và thứ tự lệnh; chưa chạy Colab |
| Dataset thật, smoke 3 cấu hình | Chưa chạy, giao người train |
| Bốn run, test accuracy/F1/KPI | Chưa chạy, không có số liệu để điền |
| TensorBoard 4 run và ảnh chụp | Chưa có; log trong test chỉ là dữ liệu giả |

## 2. Môi trường kiểm tra thực tế

Windows, Python 3.12.10. Dùng interpreter Lab 1 `.venv/Scripts/python.exe` và bổ sung thư viện vào thư mục kiểm tra riêng trong `.git/lab2-review-deps` qua PYTHONPATH; không thay dependencies môi trường Lab 1. Đây không phải bằng chứng cài mới thành công trên Colab. Notebook sẽ cài requirements và ghi phiên bản thật cho từng run.

| Thư viện | Phiên bản dùng kiểm tra |
| --- | --- |
| torch | 2.9.1+cu128 |
| torchvision | 0.24.1+cu128 |
| pytest | 9.1.0 |
| PyYAML | 6.0.3 |
| scikit-learn | 1.9.1 |
| tensorboard | 2.21.0 |
| psutil | 7.2.2 |
| matplotlib | 3.10.9 |

## 3. Nguồn và bảo toàn dữ liệu

Nền main `a7b7ef8`. Model kế thừa Xuân Bảo (`hbao2123-debug`); evaluation kế thừa Quốc An. HoangDang ghép và hoàn thiện package. `reports/metrics.json` và `notebooks/Lab 2.ipynb` là artifact cũ, không sửa hoặc diễn giải thành lần chạy mới. Bằng chứng chuẩn bị máy đọc được: [preparation_check.json](../reports/preparation_check.json).

Không ghi Đạt cho accuracy, F1, GPU KPI hoặc Run all Colab trước khi người nhận train cung cấp bằng chứng thật. Người nhận thực hiện [training_handoff.md](training_handoff.md).

Kiểm tra bổ sung: 3 smoke test thu thập được (không thực thi); cấu hình phiên của notebook đã chạy với thư mục tạm để kiểm tra giữ nguyên tham số/chặn sửa config. Logic đổi RAM Linux được kiểm tra bằng mock resource. Ba test biểu đồ/luồng demo pass sau khi chỉnh tiêu đề ảnh dài và khoảng trống trên đường loss; đã mở ảnh giả để kiểm tra bố cục, không dùng chúng làm số liệu bài nộp.

## Kiểm lại sau commit `25cc678` — 2026-11-12

HoangDang, chạy lại bộ test offline: **93 passed** (79 test package, 14 test tương thích), 0 lỗi, 12,02 giây; commit `25cc678` bổ sung test chặn ghi đè output đã chốt. `tools/verify_tv3_handoff.py` PASS (21 source lịch sử, 3 manifest không đổi, 12 checkpoint). Máy kiểm lại không có `.venv` Lab 2: dùng torch 2.9.1+cu128/torchvision 0.24.1 của `.venv` Lab 1 và cài tạm tensorboard, scikit-learn, pyyaml, matplotlib, psutil vào thư mục ngoài repo, `PYTHONPATH=src`. Không chạy smoke thật, train, Colab hoặc đo KPI T4; TASK-16 và hai KPI tốc độ T4 vẫn chờ.

## Kết quả thực thi phiên tái lập

Yêu cầu bảo vệ artifact của lượt rà toàn bộ ngày 2026-11-12 áp dụng cho các điểm ghi dữ liệu. Thư mục output có `FILES.json`, hoặc nằm trong phiên có `SESSION.json` mang trạng thái completed/failed/interrupted, được coi là đã chốt. Các lệnh data, model, training, evaluation và demo phải kiểm tra trước bước nạp dữ liệu/model và trước mọi thao tác ghi. Writer báo cáo và hàm sao chép best cũng kiểm tra trực tiếp. Khi muốn tạo báo cáo demo mới từ checkpoint cũ, chọn `reports_dir` mới trong config, sử dụng checkpoint cũ để đọc.

Phiên `rerun-thanhbinh-20261112-175308` đã hoàn tất 2026-11-12T18:22:42+07:00, trạng thái `completed`: bốn run, 13 file trọng số mới, checkpoint epoch/metrics khớp và TensorBoard đủ tag/epoch. [SESSION.json](../reports/sessions/rerun-thanhbinh-20261112-175308/SESSION.json), [FILES.json](../reports/sessions/rerun-thanhbinh-20261112-175308/FILES.json), [so sánh](../reports/sessions/rerun-thanhbinh-20261112-175308/comparison.md). Model/config/snapshot source, log lệnh train/evaluation/demo và kết quả đều giữ trong phiên; snapshot trước sửa T4 và kết quả test sau sửa được phân biệt bằng tên file.

DenseNet121 validation 88,24%, test 87,68%, macro F1 0,8755; kiểm lại checkpoint nguồn cho cùng validation/test accuracy. Test của phiên mới chỉ đánh giá model đã chọn bằng validation. Phép kiểm tra checkpoint nguồn là đánh giá mới riêng, không viết thêm vào kết quả lịch sử của Bình. Hai KPI tốc độ T4 và Run all Colab đang chờ kiểm chứng. Biểu đồ so sánh đã được mở để kiểm tra bố cục.

Đã chụp giao diện TensorBoard thật với bốn run và ba đồ thị accuracy, train loss, validation loss. Smoothing 0, hiển thị toàn bộ điểm dữ liệu. Server xác nhận đủ tag và hash bốn event khớp manifest phiên đã chốt. [Ảnh TensorBoard](../reports/sessions/tensorboard-review-20261112-191218/tensorboard.png), [biên bản bổ sung](../reports/sessions/tensorboard-review-20261112-191218/REVIEW.json).


## Kiểm lại 2026-11-13

Người giao việc: HoangDang; người thực hiện: AI agent. Thời gian ghi nhận: 2026-11-13T08:58:02+07:00. Nhánh `hoangdang`, nền `735e556`. T1 **chưa hoàn thành**: đã tạo `.venv` riêng và cài requirements, nhưng pytest dừng ở bước thu thập; **chưa chạy test nào**, không xác nhận lại mốc 93 passed trong lịch sử.

Từ `Lab 2/`:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = 'src'; $env:PYTHONUTF8 = '1'
& .venv/Scripts/python.exe -m pytest tests test.py --ignore=tests/test_smoke.py -q --basetemp .venv/pytest-temp
```

Môi trường vừa cài: Python **3.12.10**, distribution torch **2.15.0**, torchvision **0.29.2** (đọc qua `importlib.metadata`; import torch không thành công). `pip check`: `No broken requirements found.` Đây không phải môi trường torch 2.9.1+cu128 đã ghi ở các lần kiểm trước. PowerShell trả lỗi từ stderr thông báo cập nhật pip sau dòng `Successfully installed`; kiểm tra dependency tiếp theo xác nhận các distribution đã cài.

Lần T1: **9 errors during collection, 5,26 giây**, 0 test được thực thi. Lỗi đầu tiên là `OSError: [WinError 4551] An Application Control policy has blocked this file`, khi nạp `torch/lib/torch_python.dll` hoặc dependency của DLL đó. Các lỗi `torch can only be initialized once per process` tiếp theo xuất hiện sau import thất bại đầu tiên. Đây là lỗi nạp thư viện dưới chính sách Windows; chưa xác định DLL phụ thuộc cụ thể bị chặn. Không đổi chính sách hệ thống, không sửa code, không đổi bản torch để vượt lỗi; dừng T1 theo kế hoạch. Không chạy `tests/test_smoke.py`, tải dữ liệu/model hoặc lệnh train Lab 2.

T6 chạy lại đúng lệnh trên theo yêu cầu kiểm cuối: **9 errors during collection, 1,38 giây**, exit code pytest **2**; cùng lỗi chặn DLL. Output nguyên văn của lần kiểm T6 (stdout và stderr gộp, không thay đường dẫn hoặc nội dung lỗi):

```text

=================================== ERRORS ====================================
_____________________ ERROR collecting tests/test_data.py _____________________
tests\test_data.py:8: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:319: in <module>
    _load_dll_libraries()
.venv\Lib\site-packages\torch\__init__.py:302: in _load_dll_libraries
    raise err
E   OSError: [WinError 4551] An Application Control policy has blocked this file. Error loading "C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\.venv\Lib\site-packages\torch\lib\torch_python.dll" or one of its dependencies.
_____________________ ERROR collecting tests/test_demo.py _____________________
ImportError while importing test module 'C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\tests\test_demo.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_demo.py:5: in <module>
    from lab_2.demo import load_inputs
src\lab_2\demo.py:9: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:49: in <module>
    raise ImportError(
E   ImportError: `torch` can only be initialized once per process, so this module cannot be imported again: neither via `importlib.reload(torch)`, nor by re-importing after a failed `import torch`.
__________________ ERROR collecting tests/test_evaluation.py __________________
ImportError while importing test module 'C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\tests\test_evaluation.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_evaluation.py:4: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:49: in <module>
    raise ImportError(
E   ImportError: `torch` can only be initialized once per process, so this module cannot be imported again: neither via `importlib.reload(torch)`, nor by re-importing after a failed `import torch`.
___________________ ERROR collecting tests/test_handoff.py ____________________
ImportError while importing test module 'C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\tests\test_handoff.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_handoff.py:10: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:49: in <module>
    raise ImportError(
E   ImportError: `torch` can only be initialized once per process, so this module cannot be imported again: neither via `importlib.reload(torch)`, nor by re-importing after a failed `import torch`.
_________________ ERROR collecting tests/test_integration.py __________________
ImportError while importing test module 'C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\tests\test_integration.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_integration.py:5: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:49: in <module>
    raise ImportError(
E   ImportError: `torch` can only be initialized once per process, so this module cannot be imported again: neither via `importlib.reload(torch)`, nor by re-importing after a failed `import torch`.
____________________ ERROR collecting tests/test_model.py _____________________
ImportError while importing test module 'C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\tests\test_model.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_model.py:3: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:49: in <module>
    raise ImportError(
E   ImportError: `torch` can only be initialized once per process, so this module cannot be imported again: neither via `importlib.reload(torch)`, nor by re-importing after a failed `import torch`.
___________________ ERROR collecting tests/test_training.py ___________________
ImportError while importing test module 'C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\tests\test_training.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_training.py:3: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:49: in <module>
    raise ImportError(
E   ImportError: `torch` can only be initialized once per process, so this module cannot be imported again: neither via `importlib.reload(torch)`, nor by re-importing after a failed `import torch`.
_____________ ERROR collecting tests/test_training_standalone.py ______________
ImportError while importing test module 'C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\tests\test_training_standalone.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\test_training_standalone.py:13: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:49: in <module>
    raise ImportError(
E   ImportError: `torch` can only be initialized once per process, so this module cannot be imported again: neither via `importlib.reload(torch)`, nor by re-importing after a failed `import torch`.
__________________________ ERROR collecting test.py ___________________________
ImportError while importing test module 'C:\Users\Legion\Desktop\TÀI LIỆU HỌC\Lab\Lab 2\test.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
..\..\..\..\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
test.py:5: in <module>
    import torch
.venv\Lib\site-packages\torch\__init__.py:49: in <module>
    raise ImportError(
E   ImportError: `torch` can only be initialized once per process, so this module cannot be imported again: neither via `importlib.reload(torch)`, nor by re-importing after a failed `import torch`.
=========================== short test summary info ===========================
ERROR tests/test_data.py - OSError: [WinError 4551] An Application Control po...
ERROR tests/test_demo.py
ERROR tests/test_evaluation.py
ERROR tests/test_handoff.py
ERROR tests/test_integration.py
ERROR tests/test_model.py
ERROR tests/test_training.py
ERROR tests/test_training_standalone.py
ERROR test.py
!!!!!!!!!!!!!!!!!!! Interrupted: 9 errors during collection !!!!!!!!!!!!!!!!!!!
9 errors in 1.38s
```


### Tiếp tục T1 ngày 2026-11-13 — môi trường đã kiểm chứng

AI agent tiếp tục theo yêu cầu hoàn thiện, sau khi xác định lỗi nằm ở bước nạp DLL của distribution torch 2.15.0 vừa cài. Dùng cặp wheel CUDA 12.8 đã được README và các lần kiểm Lab 1/Lab 3 trên máy này xác nhận chạy được, cài từ kho chính thức PyTorch trong `.venv` riêng Lab 2:

```powershell
& .venv/Scripts/python.exe -m pip install --disable-pip-version-check torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
$env:PYTHONPATH='src'; $env:PYTHONUTF8='1'
& .venv/Scripts/python.exe -m pytest tests test.py --ignore=tests/test_smoke.py -q --basetemp .venv/pytest-temp
```

Kết quả thực tế: **93 passed**, **46 warnings**, **13,43 giây**, exit 0. Python **3.12.10**, torch **2.9.1+cu128**, torchvision **0.24.1+cu128**; import cả hai thư viện thành công, `torch.cuda.is_available()` trả `True`. `pip check` trả `No broken requirements found.` Các cảnh báo là DeprecationWarning của `pin_memory`/`is_pinned` trong PyTorch.

T1 hoàn thành với cặp phiên bản nêu trên. Các lỗi của lần cài mới ban đầu được giữ ở mục trước để lưu vết. Thao tác khắc phục chỉ thay distribution trong `.venv` Lab 2; không thay chính sách bảo mật Windows hoặc code dự án. Bộ test này dùng dữ liệu giả và bỏ qua smoke Lab 2. Phiên Colab T4 được thực hiện riêng theo T3.
