# Module 03 — Transformers Fine-Tuning & Sentiment Analysis

Thư mục dự án triển khai quy trình tinh chỉnh mô hình ngôn ngữ lớn (Transformers) và phân tích cảm xúc theo chuẩn kỹ nghệ phần mềm máy học (Spec-Driven MLOps).

## 📁 Cấu Trúc Tài Liệu Đặc Tả (`docs/`)
1. **[`docs/lab_3_description.md`](./docs/lab_3_description.md)**: Đặc tả bài toán, mục tiêu hệ thống, chỉ số KPIs & chính sách quản lý 5 Checkpoints.
2. **[`docs/data.md`](./docs/data.md)**: Hợp đồng dữ liệu IMDB, Tokenizer, phân vùng train/val/test split.
3. **[`docs/model.md`](./docs/model.md)**: Kiến trúc DistilBERT, số lượng tham số, thiết kế classification head.
4. **[`docs/training.md`](./docs/training.md)**: Cấu hình Trainer, optimizer AdamW, learning rate, quy trình smoke test và early stopping (0-2%).
5. **[`docs/evaluation.md`](./docs/evaluation.md)**: Báo cáo định lượng hệ thống (Accuracy, F1, Loss, thời gian huấn luyện, latency suy luận, VRAM GPU).
6. **[`docs/demo.md`](./docs/demo.md)**: Kịch bản và mã nguồn suy luận nhanh (Quick Inference) với dữ liệu thực tế.
7. **[`docs/plan_progress_status.md`](./docs/plan_progress_status.md)**: Lộ trình triển khai, bảng phân rã nhiệm vụ (TODO / DONE) và nhật ký kỹ thuật.

> **Notebook thực hành**: [`../Lab_3 (1).ipynb`](../Lab_3%20(1).ipynb)
