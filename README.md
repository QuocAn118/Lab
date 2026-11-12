# Lab — thực hành Deep Learning

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Lab — thực hành Deep Learning |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-12T21:38:06+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Kho bài nhóm gồm ba lab, tài liệu học trong `Learning/` và hướng dẫn làm việc trong `workflow/`. Trạng thái dưới đây căn cứ vào file hiện có; báo cáo lịch sử không thay cho kiểm thử trên máy mới.

## Bắt đầu từ đâu

| Phần | Tài liệu bắt đầu | Trạng thái triển khai |
| --- | --- | --- |
| Lab 1 — FashionMNIST MLP | [README](Lab%201/README.md), [mô tả gốc](Lab%201/docs/lab_1_description.md) | Hoàn thành kỹ thuật: 20 test, smoke + 4 normal run PASS; model chọn bằng validation có test accuracy 87,70%; sản phẩm trực tiếp trong repo, input/model qua Git LFS; xem README Lab 1 |
| Lab 2 — CIFAR-10 transfer learning | [README](Lab%202/README.md), [mô tả gốc](Lab%202/docs/lab_2_description.md) | 93 test offline và 3 smoke thật đạt; đã nhận phiên TV3 Thanh Bình (4 cấu hình, 12 checkpoint LFS), đã tái lập bốn cấu hình, test 87,68% với DenseNet121; chưa nghiệm thu KPI tốc độ T4; xem README Lab 2 |
| Lab 3 — sentiment analysis | [README](Lab%203/README.md), [hướng dẫn trình bày](Lab%203/docs/huong_dan_trinh_bay.md) | 44 test (cần `PYTHONPATH=src`), smoke pretrained và notebook tám cell đạt trên môi trường hiện tại; đánh giá lại 25.000 test đạt accuracy 91,012%; xem hồ sơ kiểm chứng Lab 3 |

Mỗi lab có hướng dẫn môi trường, cách chạy và tiến độ riêng. Các template `01_DEFINE.md`, `05_TOOLS.md`, `06_VERIFICATION.md`, `08_RELEASE.md` nằm trong `workflow/docs/`; không mặc định các file đó tồn tại trong `Lab X/docs/`.

[Rà soát và hoàn thiện ba lab ngày 2026-11-12](COMPLETION_REVIEW_2026-11-12.md) ghi các sửa đổi, phép kiểm chứng và hạng mục còn chờ kết quả Colab T4. Số test trên là lần kiểm lại sau commit `25cc678`, ghi ở cuối file đó.

## Cấu trúc đang dùng

```text
Lab/
├── Learning/                    # Tài liệu lý thuyết
├── workflow/                    # Hướng dẫn và template
├── Lab 1/                       # docs/, data/, src/, tests/, tools/
├── Lab 2/                       # package src/lab_2/, notebook Colab, kế hoạch bàn giao train
├── Lab 3/                       # docs/, notebook_helpers.py, lab3.py, report/, outputs/
├── Lab_3 (1).ipynb              # Notebook chính Lab 3
├── BRANCH_SYNC_2026-11-11.md     # Người sửa, thời gian, cách ghép nhánh
└── BRANCH_SYNC_2026-11-11.json   # Metadata chi tiết
```

## Làm việc với Git

Làm trên nhánh cá nhân và tạo pull request vào `main`. Với nhánh hiện tại:

```powershell
git switch hoangdang
git pull --ff-only origin hoangdang
# Sửa đúng phạm vi, chạy kiểm tra của phần việc, rồi commit các file liên quan.
git push origin hoangdang
```

Không coi merge đủ lịch sử là đã nghiệm thu code. Trước PR, đọc tiến độ của lab, kiểm tra các phần còn thiếu và ghi đúng kết quả đã chạy. Không sửa tay báo cáo thực nghiệm để làm khớp tài liệu.

## Trọng số và dữ liệu

- Lab 1 lưu dataset gốc và checkpoint bằng Git LFS tại `data/FashionMNIST/raw/` và `checkpoints/`. Môi trường Python/cache vẫn bị bỏ qua; Lab 2–3 theo README tương ứng.
- Các trọng số đã nhập vào Git LFS cần `git lfs install` và `git lfs pull` sau clone/pull để có nội dung nhị phân.
- Snapshot local ThanhBinh nằm ở [thư mục nhập lịch sử](Lab%203/outputs/imported-thanhbinh-40acced/README.md); file `IMPORT.json` ánh xạ về đường dẫn và commit gốc. Không dùng ngày nhập làm ngày huấn luyện.

## Lưu vết và cập nhật tài liệu

Tài liệu đang dùng có bảng thông tin mục 0. Người tạo chưa xác nhận được ghi rõ; tác giả commit không mặc nhiên là người sáng tác nội dung. Thời gian cập nhật dùng ISO 8601 với UTC+07:00. Tác giả/committer gốc và người tích hợp được ghi riêng trong [nhật ký đồng bộ](BRANCH_SYNC_2026-11-11.md).

## Nguồn và phạm vi

- README, đặc tả và tiến độ hiện có của Lab 1–3; cấu trúc Git sau đồng bộ ngày 2026-11-11.
- Yêu cầu hoàn thiện cả ba lab của người dùng; HoangDang chịu trách nhiệm tích hợp. Các lần chạy, báo cáo nhập lịch sử và phép kiểm chứng mới có hồ sơ nguồn, người thực hiện và thời gian tương ứng.
