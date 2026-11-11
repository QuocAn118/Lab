# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 2

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Dữ Liệu Thực Nghiệm (Data Specification) — Lab 2 |
| Created date | Chưa xác nhận |
| Created by | Chưa xác nhận người tạo nội dung |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T08:52:28+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

Mã nguồn: `src/lab_2/data.py`. Tham số: mục `data` trong `configs/lab_2.yaml`.

> **Phụ trách**: TV1. Đặc tả từng hàm cần viết nằm ở mục 5.

## 1. Nguồn Gốc Dataset
- **Tên dataset**: CIFAR-10, tải qua `torchvision.datasets.CIFAR10`.
- **Nhiệm vụ**: Phân loại ảnh vào 10 lớp: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
- **Quy mô gốc**: 60.000 ảnh màu $32 \times 32 \times 3$, gồm 50.000 ảnh train và 10.000 ảnh test. Dataset không có sẵn tập validation.
- **Tải dữ liệu**: Server gốc thường chậm trên Colab nên notebook tải trước file `cifar-10-python.tar.gz` từ bản sao trên Hugging Face. `torchvision` vẫn kiểm tra MD5 của file; nếu không khớp sẽ tải lại từ server gốc.

## 2. Phân Chia Dữ Liệu

| Tập | Số ảnh | Lấy từ | Vai trò |
| :--- | :---: | :--- | :--- |
| `train` | 22.500 | Tách từ `train` gốc | Tính gradient và cập nhật trọng số |
| `val` | 2.500 | Tách từ `train` gốc, ngay sau tập `train` | Theo dõi loss, accuracy; so sánh các lần chạy; chọn checkpoint; dừng sớm |
| `test` | 5.000 | Tách từ `test` gốc | Chỉ đánh giá một lần trên mốc `best` |

- **Lý do chọn số ảnh**: Đề bài không quy định số mẫu. Nhóm dùng một nửa dữ liệu để giảm thời gian dùng GPU.
- **Tái lập**: Xáo trộn với `seed = 42` trước khi lấy mẫu.
- **Chống trùng mẫu**: `train` và `val` là hai đoạn không giao nhau sau khi xáo trộn. `test` lấy từ tập gốc khác.
- **Cân bằng lớp**: Không ép cân bằng, chỉ báo cáo số ảnh mỗi lớp trong từng tập vào mục `data` của `reports/metrics.json`.
- **Xử lý lỗi**: Nếu số ảnh yêu cầu lớn hơn số ảnh của dataset thì báo `ValueError`.

## 3. Tiền Xử Lý

Các bước dùng `torchvision.transforms` và chạy theo đúng thứ tự dưới đây. Augmentation phải chạy trước `Resize` vì `RandomCrop` làm việc trên ảnh $32 \times 32$.

| Thứ tự | Bước | `train` | `val`, `test`, demo |
| :---: | :--- | :---: | :---: |
| 1 | `RandomHorizontalFlip()` | Có | Không |
| 2 | `RandomCrop(32, padding=4)` | Có | Không |
| 3 | `Resize((224, 224))` | Có | Có |
| 4 | `ToTensor()` | Có | Có |
| 5 | `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` | Có | Có |

- **Lý do `Resize` và `Normalize`**: Backbone được huấn luyện trên ảnh ImageNet $224 \times 224$ với các giá trị chuẩn hóa này, nên đầu vào cần có cùng dạng.
- **Lý do tách augmentation**: Tập `val` và `test` phải ổn định giữa các lần đánh giá.

## 4. DataLoader

| Tham số | Train | Validation, test |
| :--- | :---: | :---: |
| `batch_size` | Theo từng lần chạy (64 hoặc 128) | 128 |
| `shuffle` | `True`, với seed cố định | `False` |
| `num_workers` | 2 | 2 |
| `pin_memory` | Chỉ bật khi có GPU | Chỉ bật khi có GPU |

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
