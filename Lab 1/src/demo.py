"""Ảnh test cố định và demo đọc checkpoint best, không sửa run đã chốt.

Spec: docs/demo.md
Spec: docs/lab_1_description.md
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model_io import load_model
from run_tracking import OUTPUTS


def render_samples(model, dataset, classes, path):
    if len(dataset) < 16:
        raise ValueError("Cần ít nhất 16 ảnh test để vẽ demo")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:16].tolist()
    samples = [dataset[i] for i in indices]
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        logits = model(torch.stack([image for image, _ in samples]).to(device))
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Logits demo không hữu hạn")
    predictions = logits.argmax(1).cpu().tolist()
    fig, axes = plt.subplots(4, 4, figsize=(12, 10))
    try:
        for ax, index, (image, label), prediction in zip(axes.flat, indices, samples, predictions):
            ax.imshow(image.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"#{index} True: {classes[int(label)]}\nPred: {classes[prediction]}",
                         fontsize=9, color="black" if prediction == int(label) else "red")
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dự đoán một ảnh test bằng best.pth, chỉ đọc")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args(argv)
    model = load_model(args.run_id, "cpu", checkpoint="best")
    config = json.loads((OUTPUTS / args.run_id / "config.json").read_text(encoding="utf-8"))
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.data_pipeline import create_dataloaders
    dataset = create_dataloaders(config)[2].dataset
    if not 0 <= args.index < len(dataset):
        parser.error(f"index phải từ 0 đến {len(dataset) - 1}")
    image, label = dataset[args.index]
    with torch.inference_mode():
        logits = model(image.unsqueeze(0))[0]
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Logits demo không hữu hạn")
    prediction = int(logits.argmax())
    print(json.dumps({"run_id": args.run_id, "checkpoint": "best.pth", "sample_index": args.index,
                      "input_shape": list(image.shape), "true_label": int(label),
                      "true_name": config["classes"][int(label)], "predicted_label": prediction,
                      "predicted_name": config["classes"][prediction], "logits": logits.tolist()}))


if __name__ == "__main__":
    main()
