# Thử Nghiệm & Trực Quan Hóa (Demo Specification) — Lab 3

## 1. Mục Đích File Demo
Kiểm chứng thực tế mô hình sau khi huấn luyện với các câu đánh giá thực tế (Inference Test).

## 2. Mã Nguồn Chạy Thử Nghiệm Nhanh
```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Tải model từ checkpoint tốt nhất
checkpoint_path = "./checkpoints/best"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

test_sentences = [
    "This movie was an absolute masterpiece! The acting and plot were brilliant.",
    "Waste of time. The acting was terrible and the storyline made no sense.",
    "It was an average film, neither great nor awful."
]

for sentence in test_sentences:
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        label = "POSITIVE" if pred_class == 1 else "NEGATIVE"
        confidence = probs[0][pred_class].item() * 100
        print(f"Text: '{sentence}'")
        print(f"-> Dự đoán: {label} (Độ tin cậy: {confidence:.2f}%)\n")
```

## 3. Bảng Kết Quả Thử Nghiệm Mẫu

| Câu Thử Nghiệm | Dự Đoán | Độ Tin Cậy (%) | Nhãn Thực Tế | Đúng/Sai |
| :--- | :---: | :---: | :---: | :---: |
| *"This movie was an absolute masterpiece!..."* | POSITIVE | 99.2% | POSITIVE | Đúng |
| *"Waste of time. The acting was terrible..."* | NEGATIVE | 98.7% | NEGATIVE | Đúng |

