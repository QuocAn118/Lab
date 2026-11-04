# Tài Liệu Học Tập — Deep Learning (Learning)

Thư mục tổng hợp toàn bộ bài giảng và slide lý thuyết môn học Deep Learning, được sắp xếp theo đúng tiến trình từ cơ bản đến nâng cao.

---

## 📚 Danh Mục Slide Bài Giảng

| Chương | Tên Bài Giảng | File Tài Liệu | Nội Dung Trọng Tâm |
| :---: | :--- | :--- | :--- |
| **01** | **Introduction to Deep Learning** | [Chapter 01 - Introduction to Deep Learning.pdf](./Chapter%2001%20-%20Introduction%20to%20Deep%20Learning.pdf) | Tổng quan Deep Learning vs Machine Learning, lịch sử phát triển, ứng dụng thực tế. |
| **02** | **Artificial Neural Networks (ANN)** | [Chapter 02 - Artificial Neural Networks.pdf](./Chapter%2002%20-%20Artificial%20Neural%20Networks.pdf) | Kiến trúc Perceptron, Multi-Layer Perceptron (MLP), Activation Functions (Sigmoid, ReLU, Softmax), Forward & Backpropagation. |
| **03** | **Training Deep Neural Networks** | [Chapter 03 - Training Deep Neural Networks.pdf](./Chapter%2003%20-%20Training%20Deep%20Neural%20Networks.pdf) | Vấn đề Vanishing/Exploding Gradients, Khởi tạo trọng số (He, Xavier), Batch Normalization, Gradient Clipping. |
| **04** | **Fine-Tune Models & Training Algorithms** | [Chapter 04 - Fine-Tune Models and Training Algorithms.pdf](./Chapter%2004%20-%20Fine-Tune%20Models%20and%20Training%20Algorithms.pdf) | Optimizers (Momentum, RMSprop, Adam), Learning Rate Scheduling, Kỹ thuật Regularization (Dropout, L1/L2), Chiến lược Fine-tuning Transfer Learning. |
| **05.1** | **Convolutional Neural Networks (CNNs - Part 1)** | [Chapter 05.1 - CNNs (Part 1).pdf](./Chapter%2005.1%20-%20CNNs%20(Part%201).pdf) | Phép tích chập (Convolution), Pooling, Stride, Padding, Kiến trúc kinh điển (LeNet, AlexNet, VGG, ResNet). |
| **05.2** | **CNNs (Part 2) - Object Detection Survey** | [Chapter 05.2 - CNNs (Part 2) - Object Detection Survey.pdf](./Chapter%2005.2%20-%20CNNs%20(Part%202)%20-%20Object%20Detection%20Survey.pdf) | Khảo sát bài toán Object Detection (R-CNN, Fast/Faster R-CNN, YOLO, SSD), IoU, NMS, mAP. |
| **06** | **Recurrent Neural Networks (RNNs)** | [Chapter 06 - RNNs.pdf](./Chapter%2006%20-%20RNNs.pdf) | Xử lý dữ liệu chuỗi/văn bản (Sequential Data), Simple RNN, Vấn đề bộ nhớ dài hạn, LSTM, GRU, bước đệm tiến lên Attention & Transformers. |

---

## 🔗 Liên Kết Giữa Lý Thuyết Và Bài Thực Hành (Labs)

- **[Lab 1](../Lab%201/)**: Ứng dụng kiến thức **Chapter 01 & Chapter 02** (Xây dựng và huấn luyện mô hình mạng nơ-ron cơ bản, MLP, phân lớp dữ liệu dạng bảng/ảnh đơn giản).
- **[Lab 2](../Lab%202/)**: Ứng dụng kiến thức **Chapter 03 & Chapter 04** (Thử nghiệm các thuật toán tối ưu Adam, RMSprop, kỹ thuật Regularization, tránh Overfitting và tối ưu hóa quá trình huấn luyện).
- **[Lab 3](../Lab%203/) & [`Lab_3 (1).ipynb`](../Lab_3%20(1).ipynb)**: Ứng dụng kiến thức **Chapter 04 & Chapter 06** (Fine-tuning mô hình ngôn ngữ lớn/Transformers, phân tích cảm xúc văn bản với thư viện Hugging Face, tính toán Accuracy, Precision, Recall, F1).

