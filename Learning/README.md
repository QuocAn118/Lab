# Tài Liệu Học Tập — Deep Learning (Learning)

Thư mục tổng hợp toàn bộ bài giảng slide lý thuyết và tài liệu Markdown phân tích chuyên sâu cho từng chương môn học Deep Learning.

---

## 📚 Danh Mục Các Chương Học (Slides PDF & Ghi Chú Markdown)

Mỗi chương được đóng gói thành một thư mục riêng biệt gồm file slide gốc (`.pdf`) và file tài liệu phân tích (`.md`):

| Chương | Tên Chương | Thư Mục | Slide Gốc | Bản Phân Tích Markdown | Nội Dung Trọng Tâm |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **01** | **Introduction to Deep Learning** | [`Chapter 01/`](./Chapter%2001/) | [PDF](./Chapter%2001/Chapter%2001%20-%20Introduction%20to%20Deep%20Learning.pdf) | [Markdown](./Chapter%2001/Chapter%2001%20-%20Introduction%20to%20Deep%20Learning.md) | Tổng quan DL vs ML, lịch sử phát triển, 7 kiến trúc mạng và tập dữ liệu kinh điển. |
| **02** | **Artificial Neural Networks (ANN)** | [`Chapter 02/`](./Chapter%2002/) | [PDF](./Chapter%2002/Chapter%2002%20-%20Artificial%20Neural%20Networks.pdf) | [Markdown](./Chapter%2002/Chapter%2002%20-%20Artificial%20Neural%20Networks.md) | Nơ-ron nhân tạo, Forward & Backpropagation, Loss functions, MLP, Ma trận nhầm lẫn & Metrics. |
| **03** | **Training Deep Neural Networks** | [`Chapter 03/`](./Chapter%2003/) | [PDF](./Chapter%2003/Chapter%2003%20-%20Training%20Deep%20Neural%20Networks.pdf) | [Markdown](./Chapter%2003/Chapter%2003%20-%20Training%20Deep%20Neural%20Networks.md) | Hiện tượng Vanishing/Exploding Gradients, Khởi tạo trọng số He/Xavier, Batch Normalization, TensorBoard. |
| **04** | **Fine-Tune Models & Training Algorithms** | [`Chapter 04/`](./Chapter%2004/) | [PDF](./Chapter%2004/Chapter%2004%20-%20Fine-Tune%20Models%20and%20Training%20Algorithms.pdf) | [Markdown](./Chapter%2004/Chapter%2004%20-%20Fine-Tune%20Models%20and%20Training%20Algorithms.md) | Chiến lược Fine-tuning, Optimizers (SGD, Momentum, RMSprop, Adam, AdamW), Gradient Clipping, Hugging Face. |
| **05.1** | **Convolutional Neural Networks (CNNs - Part 1)** | [`Chapter 05.1/`](./Chapter%2005.1/) | [PDF](./Chapter%2005.1/Chapter%2005.1%20-%20CNNs%20(Part%201).pdf) | [Markdown](./Chapter%2005.1/Chapter%2005.1%20-%20CNNs%20(Part%201).md) | Phép tích chập, Pooling, công thức tính chiều Feature Map, tiến hóa 13 kiến trúc CNN (LeNet đến EfficientNet). |
| **05.2** | **CNNs (Part 2) - Object Detection Survey** | [`Chapter 05.2/`](./Chapter%2005.2/) | [PDF](./Chapter%2005.2/Chapter%2005.2%20-%20CNNs%20(Part%202)%20-%20Object%20Detection%20Survey.pdf) | [Markdown](./Chapter%2005.2/Chapter%2005.2%20-%20CNNs%20(Part%202)%20-%20Object%20Detection%20Survey.md) | Two-stage (R-CNN family) vs One-stage (YOLO v1-v8, SSD, RetinaNet), Focal Loss, CIoU Loss, mAP@0.5. |
| **06** | **Recurrent Neural Networks (RNNs)** | [`Chapter 06/`](./Chapter%2006/) | [PDF](./Chapter%2006/Chapter%2006%20-%20RNNs.pdf) | [Markdown](./Chapter%2006/Chapter%2006%20-%20RNNs.md) | Dữ liệu chuỗi, Vanilla RNN, LSTM (3 cổng & Cell State), GRU, Encoder-Decoder Seq2Seq, Word Embeddings. |

---

## 🔗 Liên Kết Giữa Lý Thuyết Và Bài Thực Hành (Labs)

- **[Lab 1](../Lab%201/)**: Ứng dụng kiến thức **Chapter 01 & Chapter 02** (Xây dựng mạng MLP/CNN cơ bản trong PyTorch, phân loại ảnh, ma trận nhầm lẫn Confusion Matrix).
- **[Lab 2](../Lab%202/)**: Ứng dụng kiến thức **Chapter 03 & Chapter 04** (Thử nghiệm các thuật toán tối ưu AdamW, RMSprop, kỹ thuật Batch Normalization, Dropout, chống Overfitting với Early Stopping).
- **[Lab 3](../Lab%203/) & [`Lab_3 (1).ipynb`](../Lab_3%20(1).ipynb)**: Ứng dụng kiến thức **Chapter 04 & Chapter 06** (Fine-tuning mô hình Transformer `distilbert-base-uncased` với thư viện Hugging Face trên tập dữ liệu IMDB).
