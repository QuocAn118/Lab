import torch
from torch import nn
from pathlib import Path

class FashionMLP(nn.Module):
  def __init__(self, model_config):
    super().__init__()
    if model_config["num_classes"] != 10:
      raise ValueError("FashionMNIST yêu cầu num_classes = 10")
    
    hidden_dims = model_config["hidden_dims"]
    if len(hidden_dims) != 2:
        raise ValueError("FashionMLP hiện hỗ trợ đúng 2 tầng ẩn")
  
    for dim in hidden_dims:
      if type(dim) is not int or dim <= 0:
        raise ValueError("Số neuron mỗi tầng ẩn phải là số nguyên dương")
      
    self.flatten = nn.Flatten()
    self.linear = nn.Sequential(
        nn.Linear(28*28, hidden_dims[0]),
        nn.ReLU(),
        nn.Linear(hidden_dims[0], hidden_dims[1]),
        nn.ReLU(),
        nn.Linear(hidden_dims[1], 10),
    )

  def forward(self, x):
    x = self.flatten(x)
    logits = self.linear(x)
    return logits

def build_model(model_config):
    return FashionMLP(model_config)

def check_model():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model_config = {
        "name": "FashionMLP",
        "input_shape": [1, 28, 28],
        "hidden_dims": [512, 512],
        "activation": "ReLU",
        "num_classes": 10}

    model = build_model(model_config).to(device)
    print(model)

    images = torch.rand(2, 1, 28, 28, device=device)

    with torch.no_grad():
        outputs = model(images)

    if outputs.shape != (2, 10):
        raise ValueError("Output shape không đúng")

    if not torch.isfinite(outputs).all().item():
        raise ValueError("Output chứa NaN hoặc Inf")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("Cấu hình:", model_config)
    print("Input:", images.shape)
    print("Output:", outputs.shape)
    print("Tổng tham số:", total_params)
    print("Tham số được phép học:", trainable_params)

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "model_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Cấu hình: {model_config}\n\n")
        f.write(f"Cấu trúc model:\n{model}\n\n")
        f.write(f"Input shape: {tuple(images.shape)}\n")
        f.write(f"Output shape: {tuple(outputs.shape)}\n")
        f.write(f"Tổng tham số: {total_params}\n")
        f.write(f"Tham số được phép học: {trainable_params}\n")

    print("Đã lưu báo cáo:", summary_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        check_model()