# Kiến Trúc Mô Hình (Model Specification) — Lab 2

Mã nguồn: `src/lab_2/model.py`. Tham số: mục `model` trong `configs/lab_2.yaml`.

> **Phụ trách**: TV2. Đặc tả từng hàm cần viết nằm ở mục 7.

---

## 7. Đặc Tả Mã Nguồn (TV2)

**File cần tạo**: `src/lab_2/model.py`. **Điều kiện trước**: TV5 đã đẩy `common.py`.
**Lấy từ `common.py`**: `argument_parser`, `load_config`, `update_report`.
**Người khác dùng lại**: `build_model`, `save_checkpoint`, `set_train_mode`, `trainable_parameters` (TV3), `load_checkpoint` (TV4, TV5), `head_of` (`test_smoke.py`).

Hằng số: `NUM_CLASSES = 10`, `HEADS = {"resnet18": "fc", "densenet121": "classifier"}`, `SUPPORTED_MODELS = tuple(HEADS)`.

Viết theo thứ tự dưới đây. Mỗi hàm có các test tương ứng trong `tests/test_model.py`.

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
