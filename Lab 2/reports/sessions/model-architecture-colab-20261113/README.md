# Bằng chứng bổ sung AC-01 trên Colab

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Created date | 2026-11-13T14:13:06+07:00 |
| Created by | OpenAI Codex, theo yêu cầu HoangDang |
| Last updated | 2026-11-13T14:13:06+07:00 |
| Updated by | OpenAI Codex, theo yêu cầu HoangDang |
| Tác giả model gốc | Xuan Bao (hbao2123-debug); tích hợp package: HoangDang |

## Nguồn và phạm vi

[Notebook tải nguyên byte](Lab_2_AC01_architecture_20261113.ipynb) có một ô code, execution 1, không có error output. [Stdout đầy đủ](architecture.txt) in kiến trúc, tên/shape tham số, trạng thái train/frozen và số tham số của ResNet18, DenseNet121, ResNet18 mở layer4. [Biên bản](INTAKE.json) ghi URL Colab, hash gói tải và notebook; [manifest gốc](FILES.json) chốt bốn file xuất từ Colab. File cache `__pycache__/model.cpython-313.pyc` trong archive không nhập vào repo.

Đây là phép kiểm kiến trúc trên **Colab CPU**, Python 3.13.18, torch 2.14.1+cpu, torchvision 0.29.1+cpu, `pretrained=False`; không train, suy luận, tải trọng số pretrained hoặc ghi checkpoint. Source model tại main `3faffbb8da00b773cc408a9c52a93ef5aa1c717f` trùng nguyên byte source training `735e5568f96819a9e52d25298ba5c34e213fa10b`; đã đối chiếu bằng Git và SHA-256. Tác giả nguồn giữ nguyên; người chạy/tích hợp bổ sung là OpenAI Codex theo yêu cầu HoangDang.

Cả ba cặp tổng/trainable khớp `model` trong [metrics T4](../colab-t4-20261113-browser-01/reports/metrics.json): 11.181.642/5.130; 6.964.106/10.250; 11.181.642/8.398.858. [audit.json](audit.json) ghi thông số môi trường và kết quả đối chiếu. Phép kiểm này bổ sung stdout thiếu ở ô model execution 6 của notebook T4; không sửa artifact phiên T4 và không được hiểu là kết quả train hoặc benchmark T4 mới.
