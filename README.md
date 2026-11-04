# Hướng Dẫn & Bộ Tài Liệu Quy Trình Thực Hành Lab

Thư mục này áp dụng bộ quy trình chuẩn hóa và kiểm chứng chất lượng dựa trên khung **Vibe Coding Workflow** cho các bài thực hành Lab.

---

## 📁 Cấu Trúc Thư Mục

```text
Lab/
├── Learning/                     # Slide lý thuyết & tài liệu Markdown chuyên sâu
│   ├── README.md                 # Mục lục chi tiết & liên kết bài giảng
│   ├── Chapter 01/               # Introduction to Deep Learning (.pdf & .md)
│   ├── Chapter 02/               # Artificial Neural Networks (.pdf & .md)
│   ├── Chapter 03/               # Training Deep Neural Networks (.pdf & .md)
│   ├── Chapter 04/               # Fine-Tune Models & Training Algorithms (.pdf & .md)
│   ├── Chapter 05.1/             # CNNs Part 1 (.pdf & .md)
│   ├── Chapter 05.2/             # Object Detection Survey (.pdf & .md)
│   └── Chapter 06/               # Recurrent Neural Networks - RNNs (.pdf & .md)
├── workflow/                     # Bộ khung quy trình chuẩn 14 bước & docs mẫu
│   ├── 01-define/ ... 14-definition-of-done/
│   ├── docs/                     # Bộ template tài liệu gốc
│   └── references/
├── Lab 1/
│   ├── docs/                     # Tài liệu theo dõi & kiểm chứng cho Lab 1
│   ├── README.md
│   └── test.py
├── Lab 2/
│   ├── docs/                     # Tài liệu theo dõi & kiểm chứng cho Lab 2
│   ├── README.md
│   └── test.py
├── Lab 3/
│   ├── docs/                     # Tài liệu theo dõi & kiểm chứng cho Lab 3
│   ├── README.md
│   └── test.py
├── Lab_3 (1).ipynb               # Notebook thực hành Lab 3
└── README.md                     # Tài liệu hướng dẫn này
```

---

## 🔄 14 Bước Quy Trình Khi Làm Bài Lab

Mỗi bài Lab (Lab 1, Lab 2, Lab 3,...) được khuyến nghị đi qua chu trình kiểm chứng:

1. **[01-define](workflow/01-define/GUIDE.md)**: Đọc kỹ đề bài, xác định mục tiêu bài lab, input dữ liệu, output mong đợi, yêu cầu ràng buộc (thư viện, phần cứng, GPU/CPU).
2. **[02-design](workflow/02-design/GUIDE.md)**: Thiết kế giải pháp (kiến trúc model, pipeline tiền xử lý, cấu trúc module/hàm).
3. **[03-plan](workflow/03-plan/GUIDE.md)**: Chia nhỏ bài Lab thành các bài tập (Exercise 1, Exercise 2,...) và mốc thời gian.
4. **[04-describe](workflow/04-describe/GUIDE.md)**: Mô tả chi tiết từng exercise / task cụ thể trước khi viết code.
5. **[05-generate](workflow/05-generate/GUIDE.md)**: Triển khai code vào notebook `.ipynb` hoặc file script `.py`.
6. **[06-review](workflow/06-review/GUIDE.md)**: Kiểm tra lại code vừa sinh (xem shape tensor, dtype, luồng xử lý, tránh copy-paste mù quáng).
7. **[07-run](workflow/07-run/GUIDE.md)**: Chạy thực tế cell notebook hoặc script Python trong đúng môi trường virtualenv/conda/Colab.
8. **[08-test](workflow/08-test/GUIDE.md)**: Kiểm chứng kết quả (Accuracy, Loss, F1-score, Confusion Matrix hoặc output test cases).
9. **[09-debug](workflow/09-debug/GUIDE.md)**: Khi xảy ra lỗi (OOM, shape mismatch, loss NaN, cú pháp API thay đổi), tìm nguyên nhân gốc và sửa dứt điểm.
10. **[10-refactor](workflow/10-refactor/GUIDE.md)**: Tối ưu code sạch sẽ, đóng gói hàm dùng lại, bỏ cell debug rác.
11. **[11-commit](workflow/11-commit/GUIDE.md)**: Commit Git với thông điệp rõ ràng theo từng Exercise hoàn thành.
12. **[12-explain-code](workflow/12-explain-code/GUIDE.md)**: Giải thích cơ chế hoạt động, ghi nhận comment hiểu bài để chuẩn bị vấn đáp/bảo vệ.
13. **[13-documentation](workflow/13-documentation/GUIDE.md)**: Hoàn thiện báo cáo, ghi nhận cấu hình siêu tham số và kết quả thực nghiệm.
14. **[14-definition-of-done](workflow/14-definition-of-done/GUIDE.md)**: Đối chiếu checklist hoàn thành: **đúng yêu cầu đề bài + notebook chạy thông suốt (Run All không lỗi) + đạt chỉ số yêu cầu**.

---

## 📝 Bộ Mẫu Tài Liệu Dùng Riêng Trong Từng Lab (`docs/`)

Trong mỗi thư mục `Lab X/docs/`, bạn có sẵn 9 file để điền thông tin bài làm:

| File | Mục đích áp dụng cho bài Lab |
| :--- | :--- |
| **`00_MINDMAP.md`** | Sơ đồ luồng thực hiện bài Lab |
| **`01_DEFINE.md`** | Xác định đề bài, dữ liệu đầu vào, mục tiêu bài thực hành |
| **`02_REQUIREMENTS.md`** | Danh sách yêu cầu chi tiết (Exercise 1, 2,...) & tiêu chí chấm điểm |
| **`03_DESIGN.md`** | Thiết kế pipeline tiền xử lý, kiến trúc mạng/model, thuật toán |
| **`04_BREAKDOWN.md`** | Danh sách các bước triển khai và trạng thái hoàn thành |
| **`05_TOOLS.md`** | Danh sách thư viện (`transformers`, `torch`,...), phiên bản, lệnh cài đặt |
| **`06_VERIFICATION.md`** | Nhật ký chạy thực nghiệm, bảng metrics (Loss, Accuracy, F1...) |
| **`07_WORK_LOG.md`** | Nhật ký quá trình làm bài và lưu ý |
| **`08_RELEASE.md`** | Checklist trước khi đóng gói nộp bài cho giảng viên/hệ thống |

---

## 🚀 Cách Bắt Đầu Làm Một Bài Lab Mới

1. Mở thư mục `Lab X/`.
2. Mở `Lab X/docs/01_DEFINE.md` và `02_REQUIREMENTS.md` để tóm tắt đề bài và tiêu chí đạt điểm.
3. Kiểm tra môi trường tại `Lab X/docs/05_TOOLS.md` (cài đặt `torch`, `transformers`,...).
4. Triển khai code trong notebook hoặc file script.
5. Ghi nhận kết quả thực nghiệm vào `Lab X/docs/06_VERIFICATION.md`.
6. Trước khi nộp, kiểm tra checklist hoàn thành tại `workflow/14-definition-of-done/GUIDE.md` và `Lab X/docs/08_RELEASE.md`.

