# Kiến Trúc Mô Hình (Model Specification) — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Kiến Trúc Mô Hình (Model Specification) — Lab 2 |
| Created date | Chưa xác nhận |
| Created by | HoangDang (Nguyễn Hoàng Đăng) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T11:13:30+07:00 |
| Updated by — nguồn đặc tả giao diện | Nguyễn Quốc An (Quoc An), `3a86e93`, `00b8d04`, `0f71b4f` |
| Người tích hợp/chuẩn hóa | HoangDang (Nguyễn Hoàng Đăng) |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

> Code gốc của Xuân Bảo (`hbao2123-debug`) đã được tích hợp vào `src/lab_2/model.py`; báo cáo tham số dùng hợp đồng `total`/`trainable`.

Ghi nhận nguồn ngày 2026-11-12: Nguyễn Quốc An thực hiện phần đặc tả giao diện và bộ test nền; phần triển khai model của Xuân Bảo được ghi riêng như trên. HoangDang đối chiếu lịch sử và bổ sung thông tin này; không thay tên tác giả code model hoặc các báo cáo triển khai của Bảo.

Mã nguồn: `src/lab_2/model.py`. Tham số: mục `model` trong `configs/lab_2.yaml`.

> **Phụ trách**: TV2. Đặc tả từng hàm đã triển khai nằm ở mục 7.

> Đã có module và kiểm tra offline. Phần in kiến trúc trên Colab (AC-01) nằm ở ô `lab_2.model` của `notebooks/lab_2_colab.ipynb`; thứ tự chạy xem [training.md](training.md) mục 9.

## 1. Model Sử Dụng

| | ResNet18 | DenseNet121 |
| :--- | :--- | :--- |
| **Vai trò** | Mô hình gốc, Exercise 2 | Exercise 1 |
| **Cách tải** | `torchvision.models.resnet18(weights="DEFAULT")` | `torchvision.models.densenet121(weights="DEFAULT")` |
| **Tầng cuối được thay** | `model.fc` | `model.classifier` |

- **Trọng số**: `weights="DEFAULT"` tải trọng số đã huấn luyện trên ImageNet. Đây là cách viết hiện tại của `pretrained=True` trong đề bài.
- **Lý do không dùng VGG**: Đề bài ghi "VGG, DenseNet, etc." nên không bắt buộc đủ. File trọng số của VGG16 nặng khoảng 530 MB, trong khi file của DenseNet121 chỉ 28,5 MB.

## 2. Xem Kiến Trúc

`python -m lab_2.model` thực hiện bước 3 của đề bài:
- In cấu trúc model bằng `print(model)`.
- In tên, kích thước và trạng thái (`train` hoặc `frozen`) của từng tham số.
- Ghi tổng số tham số và số tham số huấn luyện được vào mục `model` của `reports/metrics.json`.

## 3. Lớp Phân Loại

Tầng cuối 1.000 lớp của ImageNet được thay bằng khối sau:

| Thứ tự | Lớp | ResNet18 | DenseNet121 |
| :---: | :--- | :---: | :---: |
| 1 | `Dropout` | $p = 0.2$ | $p = 0.2$ |
| 2 | `Linear` | $512 \to 10$ | $1024 \to 10$ |

- **Đầu ra của model**: Logits của 10 lớp. Hàm mất mát Cross-Entropy nhận trực tiếp logits.
- **Softmax**: Chỉ dùng trong `demo.py` để đổi logits thành xác suất hiển thị.

## 4. Đóng Băng & Fine-Tuning

- **Mặc định (transfer learning)**: Đóng băng toàn bộ tầng trích xuất đặc trưng bằng `requires_grad = False`, chỉ huấn luyện lớp phân loại. Cả 4 lần chạy đều dùng cách này.
- **Fine-tuning**: `build_model(..., unfreeze=["layer4"])` mở khóa khối `layer4` của ResNet18. Để giảm thời gian dùng GPU, phần này chỉ được kiểm tra bằng smoke test trong `training.md` mục 4, không có lần chạy đầy đủ nên không có số accuracy.
- **BatchNorm**: Khi huấn luyện, chỉ phần được mở khóa ở chế độ `train`. BatchNorm của tầng đóng băng giữ chế độ `eval` để thống kê từ ImageNet không bị thay đổi.

## 5. Tham Số

Đo bằng cách đếm tham số sau khi thay lớp phân loại.

| Model | Tổng tham số | Huấn luyện được | Dung lượng 1 checkpoint |
| :--- | :---: | :---: | :---: |
| ResNet18, đóng băng | 11.181.642 | 5.130 | 44,8 MB |
| DenseNet121, đóng băng | 6.964.106 | 10.250 | 28,5 MB |
| ResNet18, mở khóa `layer4` | 11.181.642 | 8.398.858 | 44,8 MB |

- Khi đóng băng, ResNet18 và DenseNet121 chỉ huấn luyện lớp phân loại: $512 \times 10 + 10 = 5.130$ và $1024 \times 10 + 10 = 10.250$ tham số.
- DenseNet121 có tổng tham số ít hơn ResNet18 nhưng lớp phân loại lớn gấp đôi, vì đặc trưng đầu ra có 1.024 chiều.
- Mở khóa `layer4` làm số tham số huấn luyện được tăng từ 5.130 lên 8.398.858; tổng số tham số không đổi.

## 6. Kỹ Thuật Có Trong Model

- **BatchNorm**: Có sẵn trong backbone của ResNet18 và DenseNet121, nhóm không tự thêm.
- **Dropout**: Có ở lớp phân loại, xem mục 3.
- **Weight decay**: Thuộc optimizer, xem `training.md` mục 1.

---

## 7. Đặc Tả Mã Nguồn (TV2)

**File đã triển khai**: `src/lab_2/model.py`. **Điều kiện trước**: TV5 đã đẩy `common.py`.
**Lấy từ `common.py`**: `argument_parser`, `load_config`, `update_report`.
**Người khác dùng lại**: `build_model`, `save_checkpoint`, `set_train_mode`, `trainable_parameters` (TV3), `load_checkpoint` (TV4, TV5), `head_of` (`test_smoke.py`).

Hằng số: `NUM_CLASSES = 10`, `HEADS = {"resnet18": "fc", "densenet121": "classifier"}`, `SUPPORTED_MODELS = tuple(HEADS)`.

Giao diện triển khai và test tương ứng như dưới đây. Mỗi hàm có các test tương ứng trong `tests/test_model.py`.

#### `build_model(name, dropout, unfreeze=(), pretrained=True)`
- **Trả về**: Model có đầu ra 10 lớp.
- **Thứ tự bắt buộc**:
  1. `name` không có trong `HEADS` thì báo `ValueError`.
  2. Tải bằng `getattr(torchvision.models, name)(weights="DEFAULT" nếu pretrained, ngược lại None)`.
  3. Đóng băng toàn bộ bằng `model.requires_grad_(False)`.
  4. Mở từng khối trong `unfreeze` bằng `model.get_submodule(tên).requires_grad_(True)`.
  5. Thay tầng cuối bằng `nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, NUM_CLASSES))`. Tầng mới tạo ra đã có `requires_grad=True`, nên phải thay sau khi đóng băng.
- **Test**: `test_model_outputs_ten_classes_and_only_head_is_trainable`, `test_head_has_dropout_from_config`, `test_unfreeze_opens_only_the_named_block_and_head`, `test_unknown_model_is_rejected`.

#### `head_of(model)`
- **Trả về**: `model.fc` nếu model có thuộc tính này, ngược lại là `model.classifier`.

#### `trainable_parameters(model)`
- **Trả về**: List các tham số có `requires_grad=True`.

#### `count_parameters(model)`
- **Trả về**: `{"total": tổng số phần tử, "trainable": số phần tử huấn luyện được}`. Đếm bằng `numel()`.
- **Test**: `test_parameter_counts_match_docs`, số phải khớp mục 5.

#### `set_train_mode(model)`
- **Thứ tự**: `model.eval()`, rồi `head_of(model).train()`, rồi gọi `.train()` cho mọi module có tham số trực tiếp với `requires_grad=True` (`module.parameters(recurse=False)`).
- **Lý do**: Dropout không có tham số nên phải bật riêng qua `head_of`. BatchNorm đóng băng phải ở `eval`.
- **Test**: `test_train_mode_keeps_frozen_batchnorm_statistics_unchanged`, `test_train_mode_lets_unfrozen_batchnorm_update`.

#### `save_checkpoint(model, path, *, name, dropout, unfreeze, epoch, val_accuracy, val_loss)`
- **Nội dung file**: dict có các khóa `state_dict`, `name`, `dropout`, `unfreeze` (đổi sang list), `epoch`, `val_accuracy`, `val_loss`.
- **Cách lưu**: Tự tạo thư mục cha. `torch.save` vào file cùng thư mục tên `<tên file>.tmp`, rồi `os.replace` sang `path`. Không để lại file tạm.

#### `load_checkpoint(path, device="cpu")`
- **Trả về**: `(model, info)`. `info` là dict trên, đã bỏ khóa `state_dict`.
- **Cách làm**: `torch.load(path, map_location=device, weights_only=True)`, tạo lại model bằng `build_model(info["name"], info["dropout"], info["unfreeze"], pretrained=False)`, rồi `load_state_dict` và `.to(device)`.
- **Test**: `test_checkpoint_round_trip_restores_weights_and_leaves_no_temp_file`.

#### `main()`
- Đọc cấu hình giống `data.py`. Với từng tên trong `SUPPORTED_MODELS`: `print(model)`, rồi in tên, shape và trạng thái `train` hoặc `frozen` của từng tham số.
- Ghi `update_report(config, "model", ...)` với 3 khóa `resnet18`, `densenet121`, `resnet18_unfreeze_layer4`, mỗi khóa là kết quả của `count_parameters`.

**Tiêu chí đạt**: `python -m pytest tests/test_model.py -q` cho `12 passed`, không cần mạng vì test dùng `pretrained=False`.
