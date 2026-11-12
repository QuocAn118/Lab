# Danh mục nguồn kỹ thuật — Lab 3

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Danh mục nguồn kỹ thuật — Lab 3 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-06T21:05:06+07:00 ; HoangDang ; d108ed4 |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Thứ tự ưu tiên

Yêu cầu dự án → đặc tả nội bộ → quyết định triển khai được ghi nhận → tài liệu API chính thức. Phạm vi đối chiếu gồm notebook và các tài liệu Markdown hiện có. KPI là tiêu chí mục tiêu, không phải số liệu thực nghiệm.

## 2. Nguồn bên ngoài đã đối chiếu ngày 2026-11-06

| Mã | Nguồn chính thức | Dùng cho |
| --- | --- | --- |
| S-01 | [Transformers 4.57.1: Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer) | `Trainer`, `processing_class`, `eval_strategy`, AdamW, lưu model |
| S-02 | [Transformers 4.57.1: Callbacks](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/callback) | Callback epoch/evaluate và early stopping |
| S-03 | [IMDB dataset card](https://huggingface.co/datasets/stanfordnlp/imdb) | Dữ liệu đánh giá phim tiếng Anh, train/test riêng |
| S-04 | [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased) | Backbone tiếng Anh; head phân loại cần fine-tune |
| S-05 | [Multilingual sentiment model card](https://huggingface.co/tabularisai/multilingual-sentiment-analysis) | Module A, năm nhãn cảm xúc; khác bài nhị phân Module B |
| S-06 | [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/) | Wheel PyTorch 2.9.1 CUDA 12.8 trên Windows/Linux |
| S-07 | [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) | Accuracy, precision, recall, binary/macro F1, confusion matrix |
| S-08 | [NBClient executing notebooks](https://nbclient.readthedocs.io/en/latest/client.html) | Chạy notebook bằng kernel thực và lưu output |
| S-09 | [Installing the IPython kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) | Đăng ký kernel riêng cho `.venv` |
| S-10 | [nbformat Python API](https://nbformat.readthedocs.io/en/latest/api.html) | Đọc/ghi và kiểm tra schema notebook |
| S-11 | [VS Code: Manage Jupyter Kernels](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) | Chọn Python environment cục bộ cho notebook |
| S-12 | [Transformers 4.57.1: DistilBERT](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/distilbert) | Kiến trúc, input/output và lớp DistilBertForSequenceClassification; đối chiếu ngày 2026-11-07 |

Chỉ diễn giải API cần dùng, không chép toàn văn nguồn. Revision model/dataset thực tải và phiên bản thư viện được ghi trong artifact của mỗi run. Số tham số được đếm từ model, thay vì mặc định thông số marketing.

## Nguồn và phạm vi

- [11_AI_Usage.md §1–2](11_AI_Usage.md): quy tắc khai nguồn.
- [01_lab_3_description.md](01_lab_3_description.md): phạm vi cần nguồn.
- Các liên kết chính thức trong bảng là nguồn trực tiếp của thông tin kỹ thuật.
