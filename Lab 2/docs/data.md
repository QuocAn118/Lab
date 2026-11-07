# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 2

Mã nguồn: `src/lab_2/data.py`. Tham số: mục `data` trong `configs/lab_2.yaml`.

> **Phụ trách**: TV1. Đặc tả từng hàm cần viết nằm ở mục 5.

---

## 5. Đặc Tả Mã Nguồn (TV1)

**File cần tạo**: `src/lab_2/data.py`. **Điều kiện trước**: TV5 đã đẩy `common.py`.
**Lấy từ `common.py`**: `argument_parser`, `load_config`, `project_path`, `update_report`.
**Người khác dùng lại**: `load_splits`, `make_loader` (TV3, TV4), `build_transform`, `raw_samples`, `CLASS_NAMES` (TV4, TV5).

Hằng số `CLASS_NAMES` là list 10 tên lớp đúng thứ tự trong mục 1.

Viết theo thứ tự dưới đây. Mỗi hàm có các test tương ứng trong `tests/test_data.py`.

#### `split_indices(total, train_size, val_size, seed)`
- **Trả về**: `(train, val)`, hai list số nguyên không giao nhau, độ dài `train_size` và `val_size`.
- **Cách làm**: Xáo trộn `0..total-1` bằng `torch.randperm` với `torch.Generator().manual_seed(seed)`, rồi cắt hai đoạn liên tiếp.
- **Lỗi**: `train_size + val_size > total` thì báo `ValueError`.
- **Test**: `test_split_has_requested_sizes_and_no_shared_index`, `test_split_is_reproducible_and_depends_on_seed`, `test_split_rejects_more_samples_than_available`, `test_test_subset_is_shuffled_not_the_first_images`.

#### `build_transform(config, train)`
- **Trả về**: `transforms.Compose` theo đúng thứ tự trong mục 3. `train=True` mới có hai bước augmentation.
- **Tham số lấy từ `config["data"]`**: `image_size`, `crop_padding`, `mean`, `std`.
- **Test**: `test_eval_transform_resizes_and_normalizes_with_imagenet_values`, `test_train_transform_is_random_but_eval_transform_is_not`.

#### `load_splits(config)`
- **Trả về**: `(train, val, test)`, ba `torch.utils.data.Subset`.
- **Cách làm**:
  - Tạo 3 đối tượng `datasets.CIFAR10(root, train=..., download=True, transform=...)`, với `root = project_path(config["data"]["data_dir"])`. Train và val cần hai đối tượng riêng vì khác transform.
  - Lấy chỉ số train, val bằng `split_indices(50000, train_size, val_size, config["seed"])`.
  - Lấy chỉ số test bằng `split_indices(10000, test_size, 0, config["seed"])[0]`.
- **Kiểm tra**: Chạy `python -m lab_2.data`, cần tải dữ liệu nên không có trong test logic.

#### `make_loader(dataset, batch_size, config, shuffle=False)`
- **Trả về**: `DataLoader` theo mục 4.
- **Quy tắc**: `shuffle=True` thì truyền `generator=torch.Generator().manual_seed(config["seed"])`, nếu không thì `generator=None`. `num_workers` lấy từ `config["data"]`. `pin_memory=torch.cuda.is_available()`.
- **Test**: `test_loader_uses_batch_size_and_shuffle_is_reproducible`.

#### `raw_samples(subset, count)`
- **Trả về**: List `(ảnh PIL, nhãn số)` của `count` ảnh đầu, theo thứ tự `subset.indices`, chưa qua transform.
- **Cách làm**: `Image.fromarray(subset.dataset.data[index])` và `subset.dataset.targets[index]`.
- **Test**: `test_raw_samples_returns_original_images_with_labels`.

#### `describe(subset)`
- **Trả về**: `{"samples": số ảnh, "class_counts": {tên lớp: số ảnh}}`. Có đủ 10 lớp, lớp không có ảnh ghi 0.
- **Test**: `test_describe_counts_each_class`.

#### `main()`
- Đọc cấu hình bằng `load_config(argument_parser(__doc__).parse_args().config)`.
- Gọi `load_splits`, dùng `assert` để chắc chắn train và val không trùng chỉ số.
- Ghi `update_report(config, "data", {"train": describe(train), "val": describe(val), "test": describe(test)})`.
- In số ảnh mỗi tập và shape của một ảnh sau tiền xử lý.
- Cuối file có `if __name__ == "__main__": main()`.

**Tiêu chí đạt**: `python -m pytest tests/test_data.py -q` cho `9 passed`.
