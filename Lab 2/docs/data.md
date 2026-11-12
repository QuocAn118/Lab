# LAB 2 - Chuẩn bị dữ liệu CIFAR-10 

**Module:** `src/lab_2/data.py`  
**Người tạo:** Nguyễn Thị Vân Anh  
**Ngày cập nhật:** 11/11/2026

## 1. Mục đích

Xây dựng quy trình chuẩn bị dữ liệu CIFAR-10 cho bài toán phân loại ảnh 10 lớp. Module chịu trách nhiệm tải dữ liệu, chia tập, tiền xử lý, tạo DataLoader và thống kê dữ liệu để các thành viên sử dụng thống nhất trong huấn luyện, đánh giá và demo

**Các mục tiêu cụ thể:**

- Tạo ba tập train, validation và test đúng số lượng, đúng nguồn dữ liệu
- Tránh trùng mẫu giữa train và validation; giữ test độc lập với quá trình lựa chọn mô hình
- Dùng seed cố định để tái lập việc chọn mẫu và thứ tự shuffle của DataLoader khi khởi tạo cùng điều kiện
- Áp dụng augmentation cho train, dùng tiền xử lý ổn định cho validation và test
- Cung cấp ảnh dạng tensor phù hợp với đầu vào mô hình theo cấu hình của Lab 2
- Báo cáo số mẫu và phân bố lớp để kiểm tra dữ liệu trước khi huấn luyện

Module không thực hiện xây dựng mô hình, tính loss, cập nhật trọng số hay chọn checkpoint. Các phần đó sử dụng dữ liệu do module này cung cấp

## 2. Input / Output

### 2.1. Input

**Dữ liệu gốc:** CIFAR-10, gồm 60.000 ảnh RGB kích thước `32 × 32`, chia sẵn thành 50.000 ảnh train và 10.000 ảnh test. Validation được tách từ train gốc.

| Nhãn | Tên lớp | Ý nghĩa |
| :---: | :---: | :---: |
| 0 | airplane | Máy bay |
| 1 | automobile | Ô tô |
| 2 | bird | Chim |
| 3 | cat | Mèo |
| 4 | deer | Hươu |
| 5 | dog | Chó |
| 6 | frog | Ếch |
| 7 | horse | Ngựa |
| 8 | ship | Tàu thủy |
| 9 | truck | Xe tải |

**Cấu hình:** Dictionary `config` được nạp qua `load_config`

| Cấu hình | Giá trị | Mục đích |
| --- | --- | --- |
| `seed` | `42` | Seed dùng khi chia chỉ số và khởi tạo generator cho shuffle |
| `data.data_dir` | `data` | Đường dẫn dữ liệu được xử lý qua `project_path` |
| `data.train_size` | `22500` | Số ảnh train |
| `data.val_size` | `2500` | Số ảnh validation |
| `data.test_size` | `5000` | Số ảnh test |
| `data.image_size` | `224` | Chiều cao và chiều rộng sau resize |
| `data.crop_padding` | `4` | Padding trước khi random crop ảnh train |
| `data.mean` | `[0.485, 0.456, 0.406]` | Mean chuẩn hóa theo ba kênh RGB |
| `data.std` | `[0.229, 0.224, 0.225]` | Standard deviation chuẩn hóa theo ba kênh RGB |
| `data.num_workers` | `2` | Số worker của DataLoader |
| `training.eval_batch_size` | `128` | Batch size cấu hình cho validation và test |
| `paths.reports_dir` | `reports` | Thư mục báo cáo theo cấu hình |

**Batch size của từng run:**

| Run | Model | Train batch size |
| --- | --- | :---: |
| `resnet18_baseline` | `resnet18` | 64 |
| `densenet121` | `densenet121` | 64 |
| `resnet18_lr1e-2` | `resnet18` | 64 |
| `resnet18_bs128` | `resnet18` | 128 |

`make_loader` nhận `batch_size` qua đối số; hàm không tự đọc `runs` hoặc `training.eval_batch_size`. Phần gọi hàm cần truyền batch size của run cho train và `training.eval_batch_size` cho validation/test

### 2.2. Output

| Output | Kiểu / cấu trúc | Nội dung |
| --- | --- | --- |
| `CLASS_NAMES` | List gồm 10 chuỗi | Ánh xạ nhãn số sang tên lớp |
| `split_indices(...)` | Hai list chỉ số | Chỉ số train và validation không giao nhau |
| `build_transform(...)` | `transforms.Compose` | Pipeline xử lý ảnh |
| `load_splits(config)` | Tuple `(train, val, test)` | Ba đối tượng `Subset` |
| `make_loader(...)` | `DataLoader` | Các batch ảnh và nhãn |
| `raw_samples(...)` | List `(PIL.Image, nhãn số)` | Ảnh gốc chưa qua transform để hiển thị |
| `describe(...)` | Dictionary | Tổng số mẫu và số mẫu của từng lớp |
| `main()` | Báo cáo và thông tin console | Thống kê ba tập, kích thước tensor ảnh |

Với cấu hình trên, mỗi ảnh sau transform có shape `(3, 224, 224)`. Một batch có ảnh shape `(B, 3, 224, 224)` và nhãn shape `(B,)`, trong đó `B` là số mẫu thực tế của batch. Batch cuối có thể nhỏ hơn `batch_size` vì không bật `drop_last`.

Báo cáo được truyền vào `update_report(config, "data", report)`, với cấu trúc:

```text
data
  train: samples, class_counts
  val:   samples, class_counts
  test:  samples, class_counts
```

Theo đặc tả, báo cáo nằm trong `reports/metrics.json`; đường dẫn và cách ghi thực tế phụ thuộc vào `common.py`. Mỗi `class_counts` phải có đủ 10 tên lớp, kể cả lớp có số lượng bằng 0. Module này không xuất checkpoint hoặc lưu toàn bộ ảnh đã transform thành một bộ dữ liệu mới.

## 3. Success criteria

Các tiêu chí dưới đây tương ứng với 9 test trong `tests/test_data.py`

| STT | Tiêu chí | Kết quả cần đạt | Done |
| --- | --- | --- | :---: |
| 1 | Chia dữ liệu đúng số lượng | Train và validation có đúng số mẫu yêu cầu, không trùng chỉ số | [ ] |
| 2 | Tái lập bằng seed | Cùng seed cho cùng kết quả; đổi seed cho kết quả khác trong trường hợp kiểm thử | [ ] |
| 3 | Xử lý yêu cầu vượt số mẫu | Báo `ValueError` khi `train_size + val_size > total` | [ ] |
| 4 | Chọn mẫu sau shuffle | Đúng số mẫu và không lấy nguyên các chỉ số đầu tiên theo thứ tự trong trường hợp kiểm thử | [ ] |
| 5 | Transform đánh giá đúng | Tensor có shape `(3, 224, 224)`; giá trị normalize được kiểm tra ở kênh đỏ đúng | [ ] |
| 6 | Transform phù hợp từng tập | Transform train có tính ngẫu nhiên; transform đánh giá cho kết quả ổn định | [ ] |
| 7 | DataLoader hoạt động đúng | Chia batch thành `4, 4, 2` với 10 mẫu; shuffle tái lập được; giữ thứ tự khi tắt shuffle | [ ] |
| 8 | Lấy ảnh gốc đúng | `raw_samples` trả đúng thứ tự nhãn và ảnh được kiểm tra có kích thước `(32, 32)` | [ ] |
| 9 | Thống kê đúng | `describe` trả đúng tổng mẫu và số mẫu của các lớp được kiểm tra | [ ] |


**Phạm vi kiểm thử:** Bộ test dùng dữ liệu giả lập và fixture `config`, chưa kiểm tra trực tiếp `load_splits()` hoặc `main()`. Vì vậy, cần kiểm tra riêng nguồn dữ liệu từng tập và việc ghi báo cáo.

## 4. Các bước thực hiện

### Bước 1 — Chuẩn bị cấu hình và các thành phần dùng chung

**Điều kiện chạy:** Theo trạng thái repo hiện tại, `src/lab_2/common.py` chưa tồn tại. TV5 cần bổ sung module này với `argument_parser`, `load_config`, `project_path` và `update_report` trước khi kiểm thử runtime. Tài liệu mô tả các hàm đó theo hợp đồng đã thống nhất. Vì `data.py` import `lab_2.common` ngay đầu file, cả chạy module lẫn thu thập test đều bị chặn nếu thiếu module này, dù đã cài các thư viện.

Import các thư viện cần thiết, khai báo `CLASS_NAMES` theo đúng thứ tự nhãn và lấy các hàm hỗ trợ từ `lab_2.common`.

Trong `main()`, đọc tham số dòng lệnh bằng `argument_parser`, sau đó nạp cấu hình qua `load_config(args.config)`. Dùng `project_path(config["data"]["data_dir"])` để xác định thư mục dữ liệu.

**Kết quả cần có:** Cấu hình đầy đủ và đường dẫn dữ liệu được xác định nhất quán với dự án.

### Bước 2 — Tạo hàm chia chỉ số `split_indices`

**Điều kiện input:** `total`, `train_size`, `val_size` phải là số nguyên không âm. Hàm hiện chỉ chủ động kiểm tra tổng số mẫu vượt giới hạn; chưa có kiểm tra riêng cho số âm hoặc kiểu dữ liệu.

Hàm nhận `total`, `train_size`, `val_size`, `seed` và thực hiện:

1. Kiểm tra `train_size + val_size <= total`; nếu sai, báo `ValueError`.
2. Tạo generator riêng bằng `torch.Generator().manual_seed(seed)`.
3. Xáo trộn các chỉ số từ 0 đến `total - 1` bằng `torch.randperm`.
4. Lấy `train_size` chỉ số đầu cho train.
5. Lấy `val_size` chỉ số kế tiếp cho validation.
6. Trả về `(train_indices, val_indices)`.

Với train gốc, chọn 22.500 và 2.500 mẫu; 25.000 mẫu còn lại không sử dụng. Với test gốc, dùng cùng hàm với `train_size=5000`, `val_size=0` và lấy list thứ nhất; tên tham số ở đây chỉ biểu thị số phần tử của đoạn đầu.

Nhóm sử dụng một nửa dữ liệu gốc (25.000 ảnh từ train gốc và 5.000 ảnh từ test gốc) để giảm thời gian thực nghiệm trên GPU. Chọn mẫu ngẫu nhiên, không phân tầng theo nhãn và không ép cân bằng lớp.

**Kết quả cần có:** Với input thỏa điều kiện trên, các chỉ số hợp lệ, không trùng giữa hai đoạn và có thể tái lập. Không lấy cố định 5.000 ảnh đầu của test gốc.

### Bước 3 — Xây dựng pipeline `build_transform`

Thực hiện các bước theo thứ tự:

| Thứ tự | Transform | Train | Validation / test / demo |
| --- | --- | --- | --- |
| 1 | `RandomHorizontalFlip()` | Có | Không |
| 2 | `RandomCrop(32, padding=crop_padding)` | Có | Không |
| 3 | `Resize((image_size, image_size))` | Có | Có |
| 4 | `ToTensor()` | Có | Có |
| 5 | `Normalize(mean=mean, std=std)` | Có | Có |

Demo dùng `build_transform(config, train=False)` để áp dụng cùng pipeline với validation/test, không có augmentation ngẫu nhiên.

Flip và crop làm đa dạng ảnh train<br>
`Crop` chạy trước resize vì kích thước crop là 32, tương ứng ảnh gốc<br>
`Resize` đưa ảnh về kích thước đã chọn cho thực nghiệm<br>
`ToTensor()` đổi ảnh RGB sang tensor theo thứ tự kênh – cao – rộng và đưa pixel uint8 về khoảng `[0, 1]`<br>
`Normalize` tiếp tục chuẩn hóa từng kênh bằng `(x - mean) / std`, sau bước này giá trị không còn bị giới hạn trong `[0, 1]`.

Đặt hai bước augmentation bên trong `if train`, đặt `Resize`, `ToTensor` và `Normalize` bên ngoài nhánh này để cả ba tập đều được xử lý<br>
Các giá trị chuẩn hóa ImageNet được chọn để phù hợp với thiết lập pretrained backbone trong đặc tả

**Kết quả cần có:** `transforms.Compose` đúng thứ tự. Cùng ảnh validation/test cho cùng kết quả transform; ảnh train có thể thay đổi qua các lần truy cập. Seed của hàm chia chỉ số không tự bảo đảm toàn bộ augmentation được tái lập.

### Bước 4 — Tải dữ liệu và tạo ba tập bằng `load_splits`

Tạo ba đối tượng `datasets.CIFAR10` riêng biệt với `download=True`:

| Dataset | Tham số | Transform |
| --- | --- | --- |
| `train_dataset` | `train=True` | `build_transform(config, train=True)` |
| `val_dataset` | `train=True` | `build_transform(config, train=False)` |
| `test_dataset` | `train=False` | `build_transform(config, train=False)` |

Train và validation cần hai đối tượng riêng để dùng transform khác nhau dù cùng đọc train gốc. Không dùng test gốc làm nền cho validation.

Gọi `split_indices` cho 50.000 ảnh train gốc và 10.000 ảnh test gốc. Sau đó tạo `Subset` tương ứng và trả về chính xác **`(train, val, test)`** để thống nhất với nơi gọi.

**Kết quả cần có:** Ba tập đúng nguồn, đúng kích thước, dùng đúng transform. Chỉ so sánh giao chỉ số trực tiếp giữa train và validation vì chúng dùng cùng nguồn; chỉ số giống nhau ở hai nguồn train/test khác nhau không biểu thị cùng ảnh.

### Bước 5 — Tạo batch bằng `make_loader`

Truyền đối tượng `dataset` vào `DataLoader` với các tham số:

- `batch_size`: bên gọi truyền 64 cho ba run đầu hoặc 128 cho run `resnet18_bs128`; validation và test dùng 128 theo `training.eval_batch_size`
- `shuffle`: bên gọi truyền `True` cho train và `False` cho validation/test. Giá trị mặc định của hàm là `False`
- `num_workers`: lấy từ `config["data"]["num_workers"]`, hiện bằng 2
- `pin_memory`: dùng `torch.cuda.is_available()`, bật khi CUDA khả dụng
- `generator`: khi bật shuffle, tạo bằng `torch.Generator().manual_seed(config["seed"])`, với seed hiện bằng 42; khi tắt shuffle dùng `None`
- `drop_last`: không được truyền vào nên giữ mặc định `False`, không bỏ batch cuối

Trong test DataLoader, `num_workers` được đặt thành 0 để kiểm thử. Điều này không thay đổi giá trị 2 trong file YAML của thực nghiệm.

Generator được khởi tạo khi tạo loader, trạng thái của nó tiến lên khi được sử dụng. Vì vậy việc tái lập giữa hai lần khởi tạo không có nghĩa mọi epoch trong cùng một lần chạy đều có thứ tự giống nhau.

**Kết quả cần có:** Các batch tensor và nhãn hợp lệ, có thể chuyển cho phần huấn luyện hoặc đánh giá. `pin_memory` không tự chuyển dữ liệu lên GPU; phần sử dụng batch vẫn cần thực hiện việc đó

### Bước 6 — Cung cấp ảnh gốc bằng `raw_samples`

Lấy tối đa `count` chỉ số đầu trong `subset.indices`. Với mỗi chỉ số, đọc ảnh từ `subset.dataset.data`, chuyển thành ảnh PIL bằng `Image.fromarray`, rồi lấy nhãn tại `subset.dataset.targets`.

Không gọi `subset[index]` trong bước này vì cách đó sẽ áp dụng transform. Nếu `count` lớn hơn số mẫu, phép cắt list hiện tại trả toàn bộ mẫu có sẵn. Bên gọi nên truyền `count` là số nguyên không âm.

**Kết quả cần có:** List ảnh RGB gốc kích thước `32 × 32` và nhãn tương ứng, thích hợp để xem mẫu hoặc phục vụ demo.

### Bước 7 — Thống kê dữ liệu bằng `describe`

Khởi tạo dictionary đếm với đủ 10 lớp và giá trị 0. Duyệt qua `subset.indices`, đọc nhãn gốc, ánh xạ sang tên bằng `CLASS_NAMES` rồi tăng bộ đếm.

Trả về dictionary gồm `samples=len(subset)` và `class_counts`. Thống kê dựa trên nhãn gốc nên không cần chạy transform hay tải từng batch ảnh.

`class_counts` chỉ thống kê phân bố lớp thực tế sau chọn mẫu; không điều chỉnh hay bảo đảm cân bằng lớp.

**Kết quả cần có:** Tổng số mẫu theo lớp bằng kích thước tập; nhận biết được phân bố lớp sau chọn mẫu.

### Bước 8 — Tích hợp trong `main()` và ghi báo cáo

1. Đọc cấu hình.
2. Gọi `train, val, test = load_splits(config)`.
3. Kiểm tra train và validation không giao nhau bằng `set(train.indices).isdisjoint(val.indices)`.
4. Tạo báo cáo bằng `describe` cho từng tập.
5. Gọi `update_report(config, "data", report)`.
6. In số mẫu từng tập và shape ảnh sau tiền xử lý.
7. Thêm điểm vào `if __name__ == "__main__": main()`.

Với cấu hình mục 2, kết quả console kỳ vọng là:

```text
Train: 22500 ảnh
Validation: 2500 ảnh
Test: 5000 ảnh
Shape ảnh sau tiền xử lý: (3, 224, 224)
```

`main()` hiện không gọi `make_loader`; cần kiểm tra loader riêng. Việc in được shape của train cũng chưa chứng minh transform của validation/test đúng.

