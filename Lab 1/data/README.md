# Data — Lab 1, phần 3.1 và 3.2

Toàn bộ code, kiểm thử, dữ liệu tải về và báo cáo của hai phần nằm ở đây.

```text
data/
├── README.md
├── prepare_data.py       # 3.1: tải, kiểm tra, khảo sát và chia tập
├── data_pipeline.py      # 3.2: transforms và ba DataLoader
├── FashionMNIST/raw/     # Dataset gốc, được Git bỏ qua
├── reports/              # JSON, ảnh mẫu và sidecar nguồn gốc
└── tests/test_data.py    # Kiểm thử dữ liệu thật
```

Chạy từ thư mục `Lab 1`:

```powershell
python data/prepare_data.py
python data/data_pipeline.py --check
python -m pytest data/tests tests -q
```

Script dùng đường dẫn từ vị trí file, nên cũng chạy được từ thư mục khác khi truyền đúng đường dẫn script. Bước 3.2 yêu cầu đã chạy 3.1.

Train/validation/test: 54.000/6.000/10.000; split seed 42, không dùng test để chọn checkpoint. Cơ sở chỉ `ToTensor`, batch size 64. `train.py` gọi `data.data_pipeline.create_dataloaders(config)`.

Báo cáo trong `reports/`: `data_summary.json`, `split_indices.json`, `data_samples.png`, `batch_check.json`, mỗi file kèm `.prov.json`. Không sửa tay báo cáo; chạy lại script để tái tạo.

Đặc tả: [3.1](../docs/requirements/3_1_prepare_data.md), [3.2](../docs/requirements/3_2_transforms_dataloader.md).
