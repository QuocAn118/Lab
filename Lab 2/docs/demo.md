# Thử Nghiệm & Trực Quan Hóa (Demo Specification) — Lab 2

## 1. Kịch Bản Demo
Tải mô hình tốt nhất từ `checkpoints/best.pth`, nạp ảnh thử nghiệm bất kỳ và hiển thị kết quả phân loại cùng biểu đồ xác suất các lớp.

## 2. Mã Nguồn Suy Luận Nhanh
```python
import torch
import torchvision.transforms as transforms
from PIL import Image

# Tải model
model.load_state_dict(torch.load("checkpoints/best.pth", weights_only=True))
model.eval()

# Dự đoán ảnh
def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = output.argmax(1).item()
    return pred
```

