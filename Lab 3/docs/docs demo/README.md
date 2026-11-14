# Phân biệt tài liệu chính và bản Quốc An

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Phân biệt tài liệu chính và bản Quốc An |
| Created date | 2026-11-14T09:29:04+07:00 |
| Created by | OpenAI Codex, theo yêu cầu hoàn thiện tài liệu của chủ dự án |
| Last updated | 2026-11-14T09:34:25+07:00 |
| Updated by | OpenAI Codex (chuẩn hóa theo yêu cầu chủ dự án) |
| Phạm vi | Điều hướng tài liệu, không tạo hoặc thay kết quả thực nghiệm |

## 1. Bản chính

Đọc [01_lab_3_description.md](../01_lab_3_description.md) và bộ 01–12 trong thư mục cha. [Hướng dẫn trình bày](huong_dan_trinh_bay.md) ở đây giải thích bản chính, không phải bài train nhỏ của Quốc An.

| Nội dung | Bản chính | Bản Quốc An tham khảo |
| --- | --- | --- |
| Notebook | Lab 3/Lab_3 (1).ipynb | Lab 3/notebooks/Lab_3_colab.ipynb điều phối src/ |
| Train / validation / test | 20.000 / 5.000 / 25.000 | 1.000 / 200 / 200 |
| Epoch tối đa | 3 | 5; run T4 lịch sử dừng ở 4 |
| Kết quả test đã lưu | 91,012% | 87% |
| Checkpoint | outputs/imported-thanhbinh-40acced/full/checkpoints/ | checkpoints/ |
| Báo cáo | report/ và outputs/ theo run | reports/metrics.json |

Hai bộ kết quả thuộc hai pipeline và môi trường khác nhau. Notebook Lab_3_quocan.ipynb giữ bản kế thừa chưa thực thi trong bản sao hiện tại, không phải notebook điều phối có số liệu T4.

## 2. Tài liệu tham khảo

- [lab_3_description.md](lab_3_description.md): phạm vi và tiêu chí của bản Quốc An.
- [data.md](data.md), [model.md](model.md), [training.md](training.md): dữ liệu, model và train riêng.
- [evaluation.md](evaluation.md), [demo.md](demo.md), [plan_progress_status.md](plan_progress_status.md): số liệu và tiến độ lịch sử.

Lệnh trong các tài liệu này chạy từ Lab 3 với PYTHONPATH=src. Muốn tạo phiên mới cần đổi cả checkpoints_dir và reports_dir trong bản sao config, truyền cùng --config cho các bước. Không chạy đè thư mục có kết quả. Notebook Colab lịch sử dùng !python nên cần kiểm tra lỗi subprocess trước khi tin bảng cuối; không coi việc hiển thị reports cũ là lượt chạy thành công mới.

## Nguồn và phạm vi

- [08_implementation.md §11](../08_implementation.md), [11_AI_Usage.md](../11_AI_Usage.md), [12_traceability.md](../12_traceability.md): phạm vi, truy nguồn và ánh xạ đường dẫn.
- Bảy tài liệu Quốc An giữ nguồn nhập trong từng file và nhật ký đồng bộ. Hướng dẫn này chỉ phân biệt hai bộ, không sửa artifact hoặc số liệu.
