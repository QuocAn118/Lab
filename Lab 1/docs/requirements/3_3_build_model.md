# 3.3. Xây dựng mạng neural

Last updated: 2026-11-10 — Codex (AI hỗ trợ, chờ nhóm review).

Trạng thái: đã có code và `outputs/model_summary.txt`; báo cáo này chưa có sidecar nguồn gốc. Chưa có bằng chứng huấn luyện.

## 1. Mục đích

Xây dựng mạng MLP cho phân loại FashionMNIST, vận dụng kiến thức lớp đầu vào, lớp ẩn, hàm kích hoạt và lớp đầu ra trong Chương 2.

## 2. File thực hiện

- File code: `src/model.py`.
- Định nghĩa lớp `FashionMLP`, kế thừa `torch.nn.Module`, có phương thức `forward`.
- Hàm: `build_model(model_config)` trả về mô hình.
- Được sử dụng bởi `src/train.py` và `src/model_io.py` khi khởi tạo hoặc tải lại mô hình.

## 3. Đầu vào

- Batch tensor ảnh có dạng `[B, 1, 28, 28]` do bước 3.2 tạo.
- `model_config` gồm `name`, `input_shape`, `hidden_dims`, `activation`, `num_classes`.
- Cấu hình cơ sở dự kiến: `name = "FashionMLP"`, `input_shape = [1, 28, 28]`, `hidden_dims = [512, 512]`, `activation = "ReLU"`, `num_classes = 10`.
- Kiến trúc cơ sở là lựa chọn của bài làm; đề cho phép thử nghiệm kiến trúc khác.

## 4. Các bước xử lý

1. Kiểm tra cấu hình có số lớp đầu ra là 10, đúng hai lớp ẩn và kích thước lớp ẩn là số nguyên dương.
2. Làm phẳng mỗi ảnh từ `[1, 28, 28]` thành vector 784 phần tử.
3. Với cấu hình cơ sở, lần lượt áp dụng `Linear(784, 512)`, ReLU, `Linear(512, 512)`, ReLU và `Linear(512, 10)`.
4. Trả về 10 điểm số thô (logits) cho mỗi ảnh. Không thêm Softmax trước `CrossEntropyLoss`.
5. Cho phép thay đổi `hidden_dims` trong cấu hình để thực nghiệm mà vẫn giữ đầu ra 10 lớp.

## 5. Đầu ra, định dạng và cách lưu

| Đầu ra | Nơi sử dụng hoặc lưu | Định dạng |
| --- | --- | --- |
| Định nghĩa kiến trúc | `src/model.py` | Mã nguồn Python UTF-8, đuôi `.py`. |
| Mô hình khởi tạo | Trả về cho hàm gọi trong bộ nhớ | Đối tượng `FashionMLP`; chưa phải mô hình đã huấn luyện. |
| Kết quả forward | Trả về cho huấn luyện hoặc dự đoán | Tensor logits có dạng `[B, 10]`. |
| Mô tả kiểm tra kiến trúc | `outputs/model_summary.txt` | Văn bản UTF-8, lưu khi chạy kiểm tra riêng. |

`model_summary.txt` chứa cấu hình cơ sở, cấu trúc in từ mô hình, tổng số tham số và kích thước đầu vào/đầu ra của batch kiểm tra. Chạy kiểm tra lại sẽ thay thế tệp này. Không lưu trọng số ngẫu nhiên thành checkpoint trong bước 3.3.

## 6. Cách chạy và kiểm tra

Chạy từ thư mục `Lab 1`:

```powershell
python src/model.py --check
```

- Tạo tensor kiểm tra có dạng `[2, 1, 28, 28]`; forward trả về tensor `[2, 10]` với giá trị hữu hạn.
- Kiểm tra mô hình chứa đúng các lớp và số nút đã cấu hình.
- Mở được `outputs/model_summary.txt`.
- Cấu hình thực tế của mỗi lần huấn luyện phải được bước 3.4 lưu trong `config.json` để bước 3.6 khởi tạo lại đúng kiến trúc.

Tham khảo API: [CrossEntropyLoss — PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), sử dụng logits làm đầu vào.
