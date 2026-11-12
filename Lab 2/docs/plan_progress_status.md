# Tiến Độ Thực Hiện (Project Status) — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Tiến Độ Thực Hiện (Project Status) — Lab 2 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-12T21:38:06+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Thành Viên

Họ tên giữ theo bảng đang có trong workspace trong lúc chuẩn bị. Vai trò phụ trách không thay thế ghi nhận tác giả code gốc; người thực hiện phiên train mới phải khai báo riêng trong notebook.

| Mã | Họ tên | Phụ trách |
| :---: | :--- | :--- |
| **TV1** | Nguyễn Thị Vân Anh | Dữ liệu: `data.py`, `data.md` |
| **TV2** | Huỳnh Công Xuân Bảo | Model: `model.py`, `model.md` |
| **TV3** | Nguyễn Thanh Bình | Huấn luyện, TensorBoard: `training.py`, `training.md` |
| **TV4** | Nguyễn Quốc An | Đánh giá, KPI, biểu đồ: `evaluation.py`, `evaluation.md` |
| **TV5** | Nguyễn Hoàng Đăng | Tiện ích, demo, notebook, ghép và chạy Colab: `common.py`, `demo.py`, `Lab_2_colab.ipynb`, `demo.md`, file này |

### 1.1. Đóng góp đã đối chiếu của Nguyễn Quốc An — 2026-11-12

Tên nguồn được đối chiếu từ tác giả commit `Quoc An` và khai báo AI của thành viên, không suy từ tên nhánh. Vai trò trong bảng phân công không thay thế hồ sơ đóng góp dưới đây.

| Phần Quốc An thực hiện | File/phạm vi | Commit nguồn |
| --- | --- | --- |
| Cập nhật đặc tả theo đề, thêm cấu hình và yêu cầu môi trường | `README.md`, `configs/lab_2.yaml`, `requirements.txt`, `.gitignore`, `docs/lab_2_description.md`, `docs/data.md`, `docs/model.md`, `docs/training.md`, `docs/evaluation.md`, `docs/demo.md`, `docs/plan_progress_status.md` | `3a86e93` |
| Viết đặc tả từng hàm, giao diện và bộ test nền | Các tài liệu `docs/` trên; `tests/conftest.py`, `test_common.py`, `test_data.py`, `test_demo.py`, `test_evaluation.py`, `test_integration.py`, `test_model.py`, `test_training.py`; smoke test ban đầu ở `tests/test_smoke.py` | `00b8d04`; smoke có từ `3a86e93` |
| Khôi phục các mục tài liệu bị mất | `data.md`, `model.md`, `training.md`, `evaluation.md`, `demo.md` | `fa27469`, `0f71b4f`, `9096223`, `f9f2fd8`, `d0cd42f` |
| Tích hợp, đối chiếu và kiểm thử evaluation TV4 | `src/lab_2/evaluation.py`, khai báo công cụ tại [AI_Usage.md](AI_Usage.md) | `32ca8b1` |
| Đưa notebook cũ vào repo và chuyển vị trí | `notebooks/Lab 2.ipynb` | `186582e`, `00b8d04`; ghi nhận thao tác nhập/chuyển file, không tự kết luận người viết mọi nội dung hoặc người chạy notebook |

Phần đặc tả/test/evaluation của Quốc An có khai báo công cụ tại [AI_Usage.md](AI_Usage.md), được giữ nguyên. Bản data/tài liệu hiện hành của Vân Anh, model của Xuân Bảo, cùng các sửa đổi/tích hợp sau đó được ghi riêng; bộ test hiện hành có các bổ sung sau bộ test nền, không quy toàn bộ cho Quốc An. HoangDang, thực hiện đối chiếu và bổ sung hồ sơ ngày 2026-11-12; không đổi tác giả commit nguồn và không tạo kết quả thực nghiệm mới.

---

## 2. Cách Làm Việc Nhóm

Đã có đủ package `common`, `data`, `model`, `training`, `evaluation`, `demo` và notebook Colab. HoangDang chuẩn bị theo main tại `a7b7ef8`, kế thừa model Xuân Bảo và evaluation Quốc An; phần data của Vân Anh được nhận qua merge `828e5ef`. Sau ghép TV3, 85 test offline và 3 smoke thật đạt trên RTX 5060; đã nhận phiên lịch sử gồm bốn cấu hình/12 checkpoint của Thanh Bình và đã hoàn tất chạy lại bằng phiên riêng do HoangDang thực thi. Phân công TV1–TV5 bên dưới giữ theo bảng hiện có. Hướng dẫn cho phiên tiếp theo: [training_handoff.md](training_handoff.md).

| Thành viên | File phụ trách | Đặc tả từng hàm | Số test phải đạt |
| :---: | :--- | :--- | :---: |
| **TV1** | `src/lab_2/data.py` | `data.md` mục 4 | 20 |
| **TV2** | `src/lab_2/model.py` | `model.md` mục 7 | 12 |
| **TV3** | `src/lab_2/training.py` | `training.md` mục 8 | 6, cộng 3 smoke test |
| **TV4** | `src/lab_2/evaluation.py` | `evaluation.md` mục 8 | 6 |
| **TV5** | `src/lab_2/common.py`, `src/lab_2/demo.py`, `notebooks/Lab_2_colab.ipynb` | `lab_2_description.md` mục 9, `demo.md` mục 4 | 3 + 2, cộng 2 test ghép |

`lab_2_description.md` mục 10 ghi các khóa trong `metrics.json`, mục 11 ghi các trường hợp dễ sai. Không cần tạo `__init__.py`.

**Thứ tự phụ thuộc module (đã triển khai):**

```mermaid
flowchart LR
    TV5A["TV5: common.py"] --> TV1["TV1: data.py"]
    TV5A --> TV2["TV2: model.py"]
    TV1 --> TV3["TV3: training.py"]
    TV2 --> TV3
    TV1 --> TV4["TV4: evaluation.py"]
    TV2 --> TV4
    TV3 --> TV5B["TV5: demo.py, notebook, chạy Colab"]
    TV4 --> TV5B
```

| Đợt ghép theo thiết kế | Thành phần | Phụ thuộc |
| :---: | :--- | :--- |
| 1 | TV5: `common.py` | Mọi file đều import file này, nên làm đầu tiên |
| 2 | TV1: `data.py`, TV2: `model.py` | Chỉ cần `common.py`, hai người làm song song |
| 3 | TV3: `is_better`, `select_best_run`, rồi phần còn lại của `training.py` | Cần `data.py`, `model.py`. `evaluation.py` import `select_best_run` |
| 3 | TV4: `evaluation.py` | Cần `data.py`, `model.py`, `select_best_run`. Làm song song với TV3 |
| 4 | TV5: `demo.py`, notebook, chạy Colab | Cần đủ các phần trên |

Các module trên hiện đã import được và test offline đạt. Nếu người nhận gặp ModuleNotFoundError, kiểm tra đã lấy đủ phiên bản bàn giao, cài requirements và đặt PYTHONPATH=src; không coi lỗi import là trạng thái bình thường của bản bàn giao.

**Cách đẩy code:** làm trên nhánh cá nhân, sau đó tạo pull request vào `main`. Với người tích hợp hiện tại:

```powershell
git switch hoangdang
git pull --ff-only origin hoangdang
# Viết code, chạy kiểm tra đúng phạm vi và stage các file liên quan.
git commit -m "Lab 2: cập nhật phần việc đã kiểm chứng"
git push origin hoangdang
```

**Quy tắc ghép:**

| Việc | Lệnh | Ai làm | Khi nào |
| :--- | :--- | :---: | :--- |
| Kiểm tra phần của mình | `python -m pytest tests/test_<tên>.py -q` | Từng người | Trước khi gộp code |
| Kiểm tra ghép | `python -m pytest tests/test_integration.py -q` | TV5 | Sau khi đủ 5 phần |
| Kiểm tra 2 batch | `python -m pytest tests/test_smoke.py -q -s` | TV3 | Sau khi TV1, TV2 xong |
| Chạy thật | `notebooks/Lab_2_colab.ipynb` trên Colab | TV5 | Sau khi các lệnh trên đạt |

Muốn đổi tên hàm, tham số hoặc khóa trong `metrics.json` thì phải báo người đang dùng chúng và cập nhật đặc tả trước.

---

## 3. Bảng Công Việc

`DONE` chỉ được ghi khi đã có bằng chứng tương ứng. Mã nghiệm thu `AC` xem tại `lab_2_description.md` mục 8.

| Mã | Công việc | Bước trong đề | Trạng thái | Owner | Bằng chứng |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **TASK-01** | Hoàn thiện đặc tả `docs/` | Toàn bài | **DONE** | Team | Bộ đặc tả gốc và tài liệu bàn giao/kiểm chứng đã cập nhật theo code |
| **TASK-02** | Viết đặc tả từng hàm, test và phân công | Toàn bài | **DONE** | Team | Mục "Đặc Tả Mã Nguồn" trong `docs/`, `tests/`, mục 1, 2 của file này |
| **TASK-03** | Cài đặt môi trường; viết `common.py` | Bước 1 | **ĐẠT OFFLINE** | TV5 | `requirements.txt`, 3 test common đạt |
| **TASK-04** | Dữ liệu CIFAR-10 và `DataLoader` | Bước 5 | **ĐẠT DỮ LIỆU THẬT TRÊN WINDOWS** | TV1 | 20 test data; [checksum/split/batch/CLI thật](../reports/data_validation_2026-11-12.json); AC-03 đạt trên máy này, cần kiểm tra lại khi chạy Colab |
| **TASK-05** | Tải model, xem kiến trúc, thay lớp cuối | Bước 2, 3, 4 | **ĐÃ GHÉP, ĐẠT OFFLINE** | TV2 | `tests/test_model.py` đạt, AC-01, AC-02 |
| **TASK-06** | Logic huấn luyện và early stopping | Bước 6 | **ĐẠT OFFLINE** | TV3 | 13 test training/điều phối; giữ chống ghi đè và kiểm tra đường dẫn |
| **TASK-07** | Smoke test 2 batch | Quy trình `docs/` | **ĐẠT TRÊN RTX 5060** | TV3 | 3 smoke pretrained/CIFAR-10 thật, 29,94 giây; AC-04 |
| **TASK-08** | Huấn luyện mô hình gốc | Bước 6 | **ĐẠT TRÊN WINDOWS** | TV3 | Đã nhận nguồn và tái lập đủ 4 cấu hình; 12 checkpoint từng run và bản best mới |
| **TASK-09** | So sánh ResNet18 và DenseNet121 | Exercise 1 | **ĐẠT, MỘT SEED** | TV3 | [So sánh phiên nguồn và phiên mới](../reports/sessions/rerun-thanhbinh-20261112-175308/comparison.md); DenseNet121 hơn baseline 3,48 điểm validation |
| **TASK-10** | So sánh learning rate và batch size | Exercise 2 | **ĐẠT, MỘT SEED** | TV3 | [So sánh phiên nguồn và phiên mới](../reports/sessions/rerun-thanhbinh-20261112-175308/comparison.md); LR 0,01 −1,24 điểm, batch 128 +0,32 điểm so baseline trong phiên mới |
| **TASK-11** | Theo dõi bằng TensorBoard | Exercise 3 | **ĐẠT EVENT/PLOT/ẢNH UI** | TV3 | [Ảnh giao diện bốn run mới](../reports/sessions/tensorboard-review-20261112-191218/tensorboard.png), event và biên bản bổ sung |
| **TASK-12** | Viết code đánh giá và biểu đồ | Bước 7 | **ĐẠT OFFLINE** | TV4 | `tests/test_evaluation.py` đạt |
| **TASK-13** | Đánh giá trên tập test và đối chiếu KPI | Bước 7 | **ĐẠT WINDOWS; T4 CHƯA NGHIỆM THU** | TV4 | DenseNet121 test 87,68%, F1 0,8755; KPI tốc độ T4 chưa đo trên T4 |
| **TASK-14** | Demo suy luận | Quy trình `docs/` | **ĐẠT VỚI MODEL THẬT** | TV5 | Đã tạo xác suất và biểu đồ cho 3 ảnh test trong phiên mới; AC-10 đạt trên Windows |
| **TASK-15** | Ghép code, hoàn thiện notebook | Toàn bài | **ĐẠT PIPELINE WINDOWS, CHỜ COLAB** | TV5 | 93 test offline (kiểm lại sau `25cc678`), 3 smoke, 4 run, evaluation và demo thật; notebook chưa Run all Colab |
| **TASK-16** | Run all notebook trên Colab | Toàn bài | **TODO** | TV5 | AC-11 |

---

## 4. Nhật Ký Thực Hiện
- `2026-11-12` — HoangDang: kiểm lại sau `25cc678`, 93 test offline và verifier bàn giao PASS; môi trường mượn torch của `.venv` Lab 1, chi tiết tại [verification.md](verification.md). TASK-16 và KPI T4 chưa đổi.
- `2026-11-12` — HoangDang đã chụp giao diện TensorBoard thật của phiên chạy lại, đủ bốn run và ba đồ thị accuracy/train loss/validation loss. Biên bản `tensorboard-review-20261112-191218/REVIEW.json` lưu nguồn và hash event. Colab đã cấp Tesla T4 cho notebook `Lab_2_T4_verification_20261112.ipynb`; kết quả đầy đủ đang chờ tiếp nhận và kiểm chứng.
- `2026-11-12` — Phiên mới `rerun-thanhbinh-20261112-175308` hoàn tất trên RTX 5060, HoangDang thực thi từ code `5c03204`. Đủ bốn run, checkpoint/log/manifest riêng; validation chênh nguồn từ −0,04 đến +0,04 điểm. DenseNet121 được chọn: test 87,68%, macro F1 0,8755 trên 5.000 ảnh; checkpoint nguồn được kiểm lại riêng, kết quả nguồn không sửa. [So sánh](../reports/sessions/rerun-thanhbinh-20261112-175308/comparison.md). Tốc độ T4/Run all Colab và ảnh UI TensorBoard mới chưa nghiệm thu.
- `2026-11-12` — Nhận TV3 Nguyễn Thanh Bình tại `92930e6`: một phiên lịch sử, bốn cấu hình, 12 checkpoint LFS. HoangDang đối chiếu 21 snapshot source, ba manifest, hash checkpoint và nạp bốn model; 85 test offline và ba smoke thật trên RTX 5060 đạt. Chuẩn bị phiên mới `rerun-thanhbinh-20261112-175308`, không ghi đè kết quả nguồn; tên người bàn giao và người chạy lại tách riêng. [Biên bản nhập](../reports/imported-thanhbinh-92930e6/IMPORT.json).
- `2026-11-12` — HoangDang: rà cả Lab 1/Lab 2 của nhánh `QuocAn` (`32ca8b1`); xác nhận toàn bộ lịch sử nguồn đã có trong main `bae987d`. Bổ sung tên Nguyễn Quốc An cho training/test training Lab 1, đặc tả/cấu hình/test nền và evaluation Lab 2; ghi rõ các thao tác lấy file của thành viên khác là tích hợp. Giữ tên Vân Anh/Xuân Bảo ở phần của họ, giữ khai báo AI nguồn, code và artifact đã kiểm chứng. Xem mục 1.1 và nhật ký đồng bộ ở gốc repo.
- `2026-11-12` — Nguồn phần dữ liệu/tài liệu: Nguyễn Thị Vân Anh (`origin/vananh`, `4d3fb94`, merge `828e5ef`). HoangDang, hoàn thiện kiểm tra kiểu/kích thước, bổ sung test `load_splits`/`main`, đồng bộ tài liệu; 20 test data và toàn bộ 74 test offline đạt. CIFAR-10 thật đã kiểm tra checksum, chia tập, batch ba tập với worker=2 và CLI ghi báo cáo phiên riêng; giữ nguyên metrics lịch sử. Chưa smoke model thật/train/KPI/Colab. Bằng chứng: `reports/data_validation_2026-11-12.json`. Tên người cập nhật nguồn giữ là Vân Anh; người tích hợp/kiểm chứng ghi riêng.
- `2026-11-11T11:06:40+07:00` — HoangDang (Nguyễn Hoàng Đăng): hoàn thiện package theo main a7b7ef8, wrapper model cũ, notebook Colab, hướng dẫn bàn giao; 63 test offline đạt. Chưa tải dữ liệu/model thật, chưa smoke thật/4 run, không tạo số accuracy. Báo cáo tham số cũ được giữ nguyên.

Các dòng dưới là lịch sử từng giai đoạn:
- `2026-11-11` — HoangDang: nhận model/test từ xbao và evaluation từ QuocAn; cập nhật trạng thái, đường dẫn và workflow nhánh cá nhân → PR main. Chưa chạy lại kiểm thử trên máy hiện tại; chưa đủ module package để chạy tích hợp.
- `2026-10-31`: Chạy notebook cũ trên Colab với SGD, 2 epoch, ảnh $32 \times 32$. Kết quả xem tại `evaluation.md` mục 7.
- `2026-11-04`: Tạo bộ khung tài liệu đặc tả.
- `2026-11-07`:
  - Đối chiếu đặc tả và notebook cũ với đề bài. Notebook có đủ 7 bước và 3 Exercise, nhưng khác đặc tả về optimizer, kích thước ảnh, số epoch và checkpoint. Đặc tả thiếu 3 Exercise, bước xem kiến trúc và đánh giá trên tập test.
  - Thống nhất làm lại theo đặc tả. Hoàn thiện đặc tả theo đề bài: 2 model, 4 lần chạy, tối đa 6 epoch, 22.500/2.500/5.000 ảnh.
  - Chia việc cho 5 thành viên: xóa code, ghi đặc tả từng hàm vào `docs/`, tách test theo từng người. Chuyển notebook cũ vào `notebooks/`.
