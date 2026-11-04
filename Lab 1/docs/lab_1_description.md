# Module 01 — Artificial Neural Networks & Optimization Specification

## 1. Project Overview & Objectives
- **Domain**: Deep Learning Foundations / Computer Vision & Tabular Classification.
- **Frameworks**: PyTorch Core (`torch`, `torch.nn`, `torch.optim`, `torchvision`).
- **Core Deliverables**:
  - Triển khai kiến trúc mạng nơ-ron truyền thẳng (**Multi-Layer Perceptron - MLP** / CNN cơ bản) kế thừa từ `torch.nn.Module`.
  - Phân tích cơ chế lan truyền xuôi (**Forward Propagation**) và lan truyền ngược (**Backpropagation**).
  - Khảo sát ảnh hưởng của siêu tham số (Learning Rate, Hidden Layer Dimensions, Optimizers) tới tốc độ hội tụ.

---

## 2. Engineering Methodology (Spec-Driven Development)

```mermaid
flowchart LR
    Spec["1. Specification\n(docs/*.md)"] --> Gen["2. Implementation\n(Code Synthesis)"]
    Gen --> Smoke["3. Smoke Test\n(Dry Run 1 Batch)"]
    Smoke --> Run["4. Execution\n(Full Training Run)"]
    Run --> Check["5. Verification\n(Metric Validation)"]
```

---

## 3. Key Performance Indicators (KPIs & Metrics)
1. **Model Accuracy**: Validation Accuracy đạt mục tiêu đề ra trên tập benchmark.
2. **Computational Latency**: Đo lường thời gian huấn luyện mỗi epoch (`seconds/epoch`) và độ trễ suy luận (`latency ms/sample`).
3. **Resource Footprint**: Giám sát mức tiêu hao RAM hệ thống và bộ nhớ GPU.

---

## 4. Checkpoint Policy & Early Stopping Management
- `final_epoch.pth`: Trọng số tại epoch kết thúc.
- `best_val_loss.pth`: Trọng số đạt hàm mất mát nhỏ nhất trên tập kiểm định.
- `best_val_acc.pth`: Trọng số đạt độ chính xác cao nhất trên tập kiểm định.
- `best_val_pause`: Cơ chế Early Stopping ngắt tự động khi mức cải thiện $\le 1.0\%$.
- `best.pth`: Checkpoint tối ưu được đóng gói cho việc suy luận thực tế.
