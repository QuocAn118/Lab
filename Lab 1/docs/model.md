# MLP Architecture for Image Classification — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | MLP Architecture for Image Classification — Lab 1 |
| Created date | 09/11/2026 (nội dung model, tác giả khai báo; ngày tạo file chưa xác nhận) |
| Created by | vananh (Nguyễn Thị Vân Anh) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T13:32:59+07:00 |
| Updated by | vananh(Nguyễn Thị Vân Anh) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Model Name
- Model name: FashionMNIST MLP
- Model class: FashionMLP
- Author: Nguyễn Thị Vân Anh
- Created on: 09/11/2026
- Implementation status: Full pipeline completed; four trained runs are available. Selected by validation: run_004.
- Framework: PyTorch
- Reference: `PyTorch – Learn the Basics – Build the Neural Network`

## 2. Purpose
Mô hình được xây dựng để giải quyết bài toán phân loại ảnh đa lớp trên bộ dữ liệu FashionMNIST. Với mỗi ảnh đầu vào, mô hình dự đoán ảnh thuộc một trong 10 lớp trang phục và phụ kiện:

| Label | Class |
|:---:|:---:|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

Mô hình sử dụng kiến trúc mạng perceptron đa lớp (Multi-Layer Perceptron — MLP) làm mô hình cơ sở cho bài thực hành. Trong quá trình huấn luyện, mô hình cập nhật weight để học mối liên hệ giữa các giá trị pixel và nhãn tương ứng

**Mục tiêu của phần xây dựng mô hình:**
- Định nghĩa mạng bằng một lớp kế thừa từ `torch.nn.Module`
- Xây dựng luồng tính toán lan truyền thuận từ ảnh đầu vào đến điểm số phân loại
- Bảo đảm kích thước đầu vào và đầu ra phù hợp với yêu cầu của bài toán
- Xác định tổng số tham số và số tham số được huấn luyện
- Hỗ trợ lưu và tải lại trọng số để sử dụng sau huấn luyện
- Kiểm tra đầu ra sau khi tải lại khớp với đầu ra trước khi lưu

## 3. Mô hình ban đầu
### 3.1 Kiến trúc mô hình
Mô hình sử dụng kiến trúc MLP gồm:
- Một tầng `Flatten`
- Hai tầng ẩn `Linear`, mỗi tầng có 512 nơ-ron
- Một hàm kích hoạt `ReLU` sau mỗi tầng ẩn
- Một tầng `Linear` output có 10 nơ-ron

Mô hình được định nghĩa trong lớp FashionMLP, kế thừa từ `torch.nn.Module`
Phương thức __init__ khai báo các tầng
Phương thức `forward` xác định thứ tự dữ liệu đi qua các tầng

Cấu hình cơ sở sử dụng hai tầng ẩn, mỗi tầng có 512 nơ-ron<br>
Hiện tại cho phép thay đổi số nơ-ron của hai tầng ẩn này thông qua `model_config["hidden_dims"]`

### Cấu trúc chi tiết 

| Thành phần | Cấu hình | Chức năng |
|---|---|---|
| Flatten | `nn.Flatten()` | Trải mỗi ảnh thành một vector gồm 784 giá trị |
| Hidden layer 1 | `nn.Linear(784, 512)` | Chuyển 784 giá trị input thành một biểu diễn gồm 512 giá trị |
| Activation 1 | `nn.ReLU()` | Tạo tính phi tuyến sau tầng ẩn thứ nhất |
| Hidden layer 2 | `nn.Linear(512, 512)` | Tiếp tục biến đổi biểu diễn của mỗi ảnh |
| Activation 2 | `nn.ReLU()` | Tạo tính phi tuyến sau tầng ẩn thứ hai |
| Output layer | `nn.Linear(512, 10)` | Tạo 10 điểm số tương ứng với 10 lớp |

`Flatten` chỉ thay đổi shape tensor, không làm thay đổi các giá trị pixel và không có tham số học<br>
Mỗi tầng `Linear` thực hiện phép biến đổi sử dụng weight và bias. Các tham số này được cập nhật trong quá trình huấn luyện

`ReLU` thay các giá trị âm bằng 0 và giữ nguyên các giá trị dương<br>

Hàm kích hoạt này giúp mạng học các mối quan hệ phi tuyến. `ReLU` không có tham số học<br>

Tầng output trả về logits. Không đặt `Softmax` trong phương thức `forward` khi huấn luyện với `nn.CrossEntropyLoss` vì loss function này nhận trực tiếp logits

## 3.2 Input của mô hình

Input là một batch ảnh FashionMNIST đã được chuyển thành tensor
- Kiểu dữ liệu: `torch.float32`
- Shape: `(B, 1, 28, 28)`
- B: số lượng ảnh trong batch
- 1: một kênh ảnh xám
- 28 × 28: chiều cao và chiều rộng của ảnh
- Khoảng giá trị: [0, 1] khi áp dụng `ToTensor()`

## 3.3 Output của mô hình 
Output là tensor logits có shape (B, 10)<br>
Mỗi hàng tương ứng với một ảnh trong batch<br>
Mỗi cột tương ứng với điểm số của một lớp<br>
Ví dụ:

| | Cột 0 (T-shirt/top) | Cột 1 (Trouser) | Cột 2 (Pullover) | Cột 3 (Dress) | ... | Cột 9 (Ankle boot) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Ảnh 0 (hàng 0) | 0.2 | -1.1 | 0.5 | -0.3 | ... | 3.8 |
| Ảnh 1 (hàng 1) | 2.9 | 0.1 | 1.4 | 0.0 | ... | -0.7 |
| Ảnh 2 (hàng 2) | -0.5 | 4.2 | -0.9 | 1.0 | ... | 0.3 |

Logits là các điểm số thô, có thể âm hoặc dương<br>
Các giá trị này không bắt buộc phải nằm trong khoảng [0, 1] hoặc có tổng bằng 1

## 3.4 Luồng forward và shape tensor

| Bước | Output shape |
|:---|:---:|
| Input | (B, 1, 28, 28) |
| Flatten | (B, 784) |
| Linear(784, 512) | (B, 512) |
| ReLU | (B, 512) |
| Linear(512, 512) | (B, 512) |
| ReLU | (B, 512) |
| Linear(512, 10) | (B, 10) |

Mỗi ảnh input tạo ra một bộ gồm 10 điểm số output

## 3.5 Số lượng tham số
Với một tầng Linear có bias:<br>
**Số tham số** = `in_features` × `out_features` + `out_features`

Trong đó:
- `in_features` × `out_features` là số phần tử của ma trận weight
- `out_features` là số phần tử của vector bias

| Tầng | Weight | Bias | Tổng |
|---|:---:|:---:|:---:|
| Linear(784, 512) | 401408 | 512 | 401920 |
| Linear(512, 512) | 262144 | 512 | 262656 |
| Linear(512, 10) | 5120 | 10 | 5130 |
| Total | 668672 | 1034 | 669706 |

- **_Tổng tham số_**: 669706
- **_Số tham số được huấn luyện_**: 669706
- Không có tầng nào bị đóng băng trong cấu hình ban đầu
- `Flatten` và `ReLU` không có tham số học

## 3.6 Trạng thái ban đầu
Mô hình được khởi tạo mới bằng cơ chế khởi tạo mặc định của PyTorch, không sử dụng weight đã được huấn luyện trước<br>

Trước khi huấn luyện, mô hình đã có thể thực hiện forward và trả về logits, nhưng các weight chưa được tối ưu cho bài toán phân loại FashionMNIST

## 3.7. Tệp mã nguồn

- Tệp mã nguồn: [src/model.py](../src/model.py)
- Lớp mô hình: `FashionMLP`
- Hàm tạo mô hình: `build_model(model_config)`
- Hàm kiểm tra kiến trúc: `check_model()`
- Lệnh kiểm tra: `python src/model.py --check`
- Báo cáo kiểm tra: `outputs/model_summary.txt`

**_Tài liệu tham khảo:_**
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

## 3.8. Kiểm tra mô hình
Trước khi bàn giao mô hình cho phần huấn luyện:
1. Tạo hoặc lấy một batch ảnh có shape (B, 1, 28, 28)
2. Chạy forward và kiểm tra output có shape (B, 10)
3. Kiểm tra output không chứa giá trị NaN hoặc Inf
4. Đếm tổng số tham số và số tham số được huấn luyện
5. Ghi kết quả thực tế vào `model_summary.txt`

**_Trạng thái kiểm chứng:_**<br>
File [model_summary.txt](../outputs/model_summary.txt) đã lưu ghi nhận input `(2, 1, 28, 28)`, output `(2, 10)` và `669706 trainable parameters`. Đây là kết quả kiểm tra kiến trúc, không phải bằng chứng huấn luyện<br>
Bản tóm tắt được tạo lại từ code đã commit vào ngày `2026-11-11`; file sidecar tương ứng lưu giữ lịch sử tạo trước đó. Kết quả kiểm tra kiến trúc này được tách biệt với bằng chứng huấn luyện

`check_model()` hiện ghi lại file sidecar của bản tóm tắt và lưu giữ lịch sử của nó. Một regression test kiểm tra mã hash của file được tạo.

# 4. Mô hình đã huấn luyện

Run được chọn: **run_004**, dựa trên validation accuracy cao nhất trong bốn run. Không sử dụng kết quả trên tập test để lựa chọn.

- **_Thời gian huấn luyện_**: từ `2026-11-11T09:45:49+07:00` đến `2026-11-11T09:46:06+07:00`, do HoangDang thực hiện trên RTX 5060 Laptop.
- Kiến trúc giữ nguyên `784 → 512 → 512 → 10`; sử dụng `Adam` với `learning rate 0.001`; huấn luyện `4 epoch`, bật early stopping với `patience 2/min_delta 0.01`
- **_Validation accuracy tốt nhất_**: 0.8810; `test accuracy: 0.8770`, `test loss: 0.346382`, trên 10.000 ảnh test.
- **_Weights_**: `checkpoints/run_004/best.pth`; [config](../outputs/run_004/config.json), [checkpoint index](../checkpoints/run_004/checkpoint_index.json).
- [Kiểm tra reload](../outputs/run_004/reload_check.json): đạt, sử dụng 16 mẫu validation, các nhãn khớp nhau, chênh lệch logits lớn nhất bằng 0 trên CPU.
- [Hiệu năng](../outputs/run_004/performance.json), [metrics](../outputs/run_004/metrics.json), [manifest](../outputs/run_004/manifest.json).

Theo hồ sơ được kế thừa, tác giả triển khai ban đầu vẫn là Nguyen Thi Van Anh. Phần hoàn thiện và tích hợp do HoangDang thực hiện. Ngày tạo nội dung model 09/11/2026 được giữ theo khai báo trong bản nhập `origin/vananh` (`a2398cc`); chưa xác minh độc lập. Đây không phải ngày tạo file tài liệu: mốc Git đầu tiên của file là 2026-11-04 như bảng mục 0. Toàn bộ 25 file checkpoint, bao gồm mô hình tốt nhất được chọn, đều được quản lý phiên bản trong repository này thông qua Git LFS. Chạy `git lfs pull origin hoangdang` sau khi clone/pull; không cần file ZIP hoặc liên kết Drive.

## Nguồn và phạm vi

- [Đặc tả](lab_1_description.md), [3.3](requirements/3_3_build_model.md), [3.6](requirements/3_6_save_load.md), mã nguồn mô hình và các file kết quả đo được nêu ở trên.

Ghi nhận tích hợp/hiệu đính 2026-11-12: HoangDang, sửa lỗi chữ và làm rõ mốc ngày khai báo; giữ nguyên tên/ngày cập nhật nguồn của Vân Anh, số đo và artifact thực nghiệm.
