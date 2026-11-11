# Machine Learning & Deep Learning Engineering Framework

Kho lưu trữ và môi trường thực nghiệm các mô hình Học Sâu (Deep Learning), áp dụng quy trình kiểm chứng chất lượng phần mềm máy học chuẩn hóa (**Spec-Driven MLOps & Quality Assurance**).

---

## 📁 Cấu Trúc Tổng Thể

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
│   ├── docs/                     # Hệ thống 7 file đặc tả Module 01
│   ├── README.md
│   └── test.py
├── Lab 2/
│   ├── docs/                     # Hệ thống 7 file đặc tả Module 02 (Pre-trained Neural Networks)
│   ├── configs/                  # Tham số và 4 lần chạy
│   ├── src/lab_2/                # common, data, model, training, evaluation, demo
│   ├── tests/                    # Test logic và smoke test
│   ├── notebooks/                # Notebook chạy trên Colab, notebook cũ
│   ├── reports/                  # metrics.json, figures/, runs/ (tạo khi chạy)
│   ├── requirements.txt
│   └── README.md
├── Lab 3/
│   ├── docs/                     # Hệ thống 7 file đặc tả Module 03 (Transformers Fine-Tuning)
│   ├── configs/                  # Tham số Exercise 2
│   ├── src/                      # exercise_1/, exercise_2/
│   ├── tests/                    # Test logic và smoke test
│   ├── notebooks/                # Notebook chạy trên Colab, notebook cũ
│   ├── reports/                  # metrics.json, figures/ (tạo khi chạy)
│   ├── requirements.txt
│   └── README.md
└── README.md                     # Tài liệu điều phối tổng quan
```

---

## 🔄 Quy Trình Kỹ Nghệ Phát Triển Mô Hình

Mỗi module thực nghiệm được kiểm chứng theo quy trình 5 bước nghiêm ngặt:

1. **Specification (`docs/`)**: Soạn thảo  dữ liệu, kiến trúc mô hình, siêu tham số và chỉ số KPIs trước khi lập trình.
2. **Implementation**: Triển khai mã nguồn đáp ứng đúng  đặc tả.
3. **Smoke Test**: Chạy thử mồi với mini-batch ($1 \sim 2$ batches) để xác thực tính toàn vẹn của tensor shapes, gradient computation và giới hạn VRAM.
4. **Execution (Full Run)**: Huấn luyện đầy đủ trên toàn bộ dữ liệu, tự động lưu 5 mốc Checkpoints.
5. **Verification**: Đo lường định lượng các chỉ số Performance (Accuracy, Throughput, Latency, Memory) và đối chiếu với tiêu chí nghiệm thu.

---

## 📝 Hệ Thống Tài Liệu Đặc Tả Chuẩn Hóa Trong Từng Module (`docs/`)

Trong mỗi thư mục `Lab 1`, `Lab 2`, `Lab 3`, hệ thống tài liệu đặc tả được cấu trúc đồng bộ theo 7 file:

| File Đặc Tả | Phạm Vi & Vai Trò Trong Hệ Thống |
| :--- | :--- |
| **`lab_{số}_description.md`** | Đặc tả bài toán, mục tiêu hệ thống, chỉ số KPIs (Accuracy, Time, Resource) và chính sách quản lý 5 Checkpoints |
| **`data.md`** |  dữ liệu: Nguồn dataset, kích thước, phân vùng train/val/test, tiền xử lý (Tokenizer/Transforms) |
| **`model.md`** | Kiến trúc mô hình: Lựa chọn backbone, số lượng tham số, thiết kế classification head |
| **`training.md`** | Cấu hình huấn luyện: Hyperparameters, Optimizers, Loss function, quy trình Smoke Test & Early Stopping |
| **`evaluation.md`** | Báo cáo định lượng hệ thống: Accuracy, F1, Loss, thời gian huấn luyện, độ trễ suy luận và VRAM/RAM |
| **`demo.md`** | Kịch bản và mã nguồn suy luận nhanh (Quick Inference) với dữ liệu thực tế |
| **`plan_progress_status.md`** | Lộ trình kỹ thuật, bảng phân rã nhiệm vụ (TODO / IN PROGRESS / DONE) và nhật ký thực thi |

---

## 🚀 Cách Bắt Đầu Làm Một Bài Lab Mới

1. Mở thư mục `Lab X/`.
2. Mở `Lab X/docs/01_DEFINE.md` và `02_REQUIREMENTS.md` để tóm tắt đề bài và tiêu chí đạt điểm.
3. Kiểm tra môi trường tại `Lab X/docs/05_TOOLS.md` (cài đặt `torch`, `transformers`,...).
4. Triển khai code trong notebook hoặc file script.
5. Ghi nhận kết quả thực nghiệm vào `Lab X/docs/06_VERIFICATION.md`.
6. Trước khi nộp, kiểm tra checklist hoàn thành tại `workflow/14-definition-of-done/GUIDE.md` và `Lab X/docs/08_RELEASE.md`.

