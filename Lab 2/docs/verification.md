# Kiểm chứng chuẩn bị — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Kiểm chứng chuẩn bị — Lab 2 |
| Created date | 2026-11-11T11:06:40+07:00 |
| Created by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Last updated | 2026-11-11T11:13:30+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Chuẩn bị code và bàn giao; chưa huấn luyện thật hoặc nghiệm thu KPI |

## 1. Kết quả và giới hạn

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

Nền main `a7b7ef8`. Model kế thừa Xuân Bảo (`hbao2123-debug`); evaluation kế thừa Quốc An. HoangDang/Codex ghép và hoàn thiện package. `reports/metrics.json` và `notebooks/Lab 2.ipynb` là artifact cũ, không sửa hoặc diễn giải thành lần chạy mới. Bằng chứng chuẩn bị máy đọc được: [preparation_check.json](../reports/preparation_check.json).

Không ghi Đạt cho accuracy, F1, GPU KPI hoặc Run all Colab trước khi người nhận train cung cấp bằng chứng thật. Người nhận thực hiện [training_handoff.md](training_handoff.md).

Kiểm tra bổ sung: 3 smoke test thu thập được (không thực thi); cấu hình phiên của notebook đã chạy với thư mục tạm để kiểm tra giữ nguyên tham số/chặn sửa config. Logic đổi RAM Linux được kiểm tra bằng mock resource. Ba test biểu đồ/luồng demo pass sau khi chỉnh tiêu đề ảnh dài và khoảng trống trên đường loss; đã mở ảnh giả để kiểm tra bố cục, không dùng chúng làm số liệu bài nộp.
