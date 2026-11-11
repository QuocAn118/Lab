# Lab — thực hành Deep Learning

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Lab — thực hành Deep Learning |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T09:26:53+07:00 ; HoangDang ; ca3b03e |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Kho bài nhóm gồm ba lab, tài liệu học trong `Learning/` và hướng dẫn làm việc trong `workflow/`. Trạng thái dưới đây căn cứ vào file hiện có; báo cáo lịch sử không thay cho kiểm thử trên máy mới.

## Bắt đầu từ đâu

| Phần | Tài liệu bắt đầu | Trạng thái triển khai |
| --- | --- | --- |
| Lab 1 — FashionMNIST MLP | [README](Lab%201/README.md), [mô tả gốc](Lab%201/docs/lab_1_description.md) | Hoàn thành kỹ thuật: 17 test, smoke + 4 normal run PASS; model chọn bằng validation có test accuracy 87,70%; sản phẩm/ZIP cục bộ trong README Lab 1 |
| Lab 2 — CIFAR-10 transfer learning | [README](Lab%202/README.md), [mô tả gốc](Lab%202/docs/lab_2_description.md) | Có model độc lập và evaluation; chưa đủ các module package dùng chung để chạy toàn pipeline |
| Lab 3 — sentiment analysis | [README](Lab%203/README.md), [hướng dẫn trình bày](Lab%203/docs/huong_dan_trinh_bay.md) | Có notebook/pipeline và báo cáo lịch sử; bản nhập QuocAn và snapshot ThanhBinh được phân biệt rõ, chưa chạy lại trên môi trường hiện tại |

Mỗi lab có hướng dẫn môi trường, cách chạy và tiến độ riêng. Các template `01_DEFINE.md`, `05_TOOLS.md`, `06_VERIFICATION.md`, `08_RELEASE.md` nằm trong `workflow/docs/`; không mặc định các file đó tồn tại trong `Lab X/docs/`.

## Cấu trúc đang dùng

```text
Lab/
├── Learning/                    # Tài liệu lý thuyết
├── workflow/                    # Hướng dẫn và template
├── Lab 1/                       # docs/, data/, src/, tests/, tools/
├── Lab 2/                       # docs/, configs/, model.py, test.py, src/lab_2/
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

- Dataset gốc và môi trường Python được bỏ qua bởi Git; chuẩn bị theo README từng lab.
- Các trọng số đã nhập vào Git LFS cần `git lfs install` và `git lfs pull` sau clone/pull để có nội dung nhị phân.
- Snapshot local ThanhBinh nằm ở [thư mục nhập lịch sử](Lab%203/outputs/imported-thanhbinh-40acced/README.md); file `IMPORT.json` ánh xạ về đường dẫn và commit gốc. Không dùng ngày nhập làm ngày huấn luyện.

## Lưu vết và cập nhật tài liệu

Tài liệu đang dùng có bảng thông tin mục 0. Người tạo chưa xác nhận được ghi rõ; tác giả commit không mặc nhiên là người sáng tác nội dung. Thời gian cập nhật dùng ISO 8601 với UTC+07:00. Tác giả/committer gốc và người tích hợp được ghi riêng trong [nhật ký đồng bộ](BRANCH_SYNC_2026-11-11.md).

## Nguồn và phạm vi

- README, đặc tả và tiến độ hiện có của Lab 1–3; cấu trúc Git sau đồng bộ ngày 2026-11-11.
- Yêu cầu chuẩn hóa tài liệu của người dùng; HoangDang chịu trách nhiệm tích hợp với hỗ trợ OpenAI Codex. Không sinh thêm code huấn luyện hoặc kết quả thực nghiệm trong lần này.
