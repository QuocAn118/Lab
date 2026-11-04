# Dữ Liệu Thực Nghiệm (Data Specification) — Lab 2

## 1. Nguồn Gốc Dataset
- **Tên dataset**: CIFAR-10 (hoặc dataset thị giác được giao trong đề Lab 2).
- **Quy mô**: 60,000 ảnh màu $32 \times 32 \times 3$, gồm 10 lớp vật thể (máy bay, ô tô, chim, mèo, nai, chó, ếch, ngựa, tàu thủy, xe tải).
- **Phân chia**: 50,000 ảnh Train, 10,000 ảnh Test.

## 2. Tiền Xử Lý Dữ Liệu & Data Augmentation
- Resize / Crop: `transforms.Resize((224, 224))` nếu dùng cho ResNet/VGG pre-trained.
- Data Augmentation: `transforms.RandomHorizontalFlip()`, `transforms.RandomCrop(32, padding=4)`.
- Chuẩn hóa theo ImageNet: `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`.

