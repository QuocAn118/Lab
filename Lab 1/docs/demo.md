# Thử Nghiệm & Trực Quan Hóa (Demo Specification) — Lab 1

## 1. Kịch Bản Demo
Tải mô hình từ checkpoint `best.pth`, nạp một bức ảnh ngẫu nhiên từ tập Test và hiển thị dự đoán cùng độ tin cậy.

## 2. Mã Nguồn Suy Luận Nhanh (Inference Script)
```python
import torch

model.load_state_dict(torch.load("checkpoints/best.pth", weights_only=True))
model.eval()

# Dự đoán mẫu ảnh
with torch.no_grad():
    output = model(sample_image)
    pred_label = output.argmax(1).item()
    print(f"Nhãn dự đoán: {pred_label}")
```

