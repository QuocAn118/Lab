"""Demo ảnh CIFAR-10 hoặc ảnh của bạn bằng checkpoint chọn theo validation."""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

from lab_2.common import argument_parser, load_config, project_path, update_report
from lab_2.data import CLASS_NAMES, build_transform, load_splits, raw_samples
from lab_2.evaluation import SERIES, SURFACE, style_axis
from lab_2.model import load_checkpoint

DEFAULT_SAMPLES = 3


def load_inputs(config, paths):
    if paths:
        inputs = []
        for path in paths:
            try:
                with Image.open(path) as image:
                    inputs.append((str(path), image.convert("RGB"), None))
            except OSError as exc:
                raise SystemExit(f"Không đọc được ảnh {path}: {exc}") from exc
        return inputs
    _, _, test = load_splits(config)
    return [(f"Ảnh test số {i + 1}", image.convert("RGB"), CLASS_NAMES[label])
            for i, (image, label) in enumerate(raw_samples(test, DEFAULT_SAMPLES))]


def plot_probabilities(results, path):
    if not results:
        raise ValueError("Chưa có ảnh demo")
    figure, axes = plt.subplots(1, len(results), sharey=True, squeeze=False,
                                figsize=(6 * len(results), 5), facecolor=SURFACE)
    for axis, result in zip(axes[0], results):
        names, values = zip(*result["probabilities"].items())
        axis.barh(names, values, color=SERIES[0][0])
        title = textwrap.fill(Path(result["image"]).name, width=30)
        style_axis(axis, f"{title}\n{result['prediction']}", "Xác suất", "Lớp")
        axis.set_xlim(0, 1)
    axes[0, 0].invert_yaxis()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argument_parser(__doc__)
    parser.add_argument("--image", nargs="+", default=[])
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = project_path(config["paths"]["checkpoints_dir"]) / "best.pth"
    if not checkpoint.exists():
        raise SystemExit("Chưa có best.pth. Chạy python -m lab_2.evaluation trước.")
    model, _ = load_checkpoint(checkpoint, "cpu")
    model.eval()
    transform = build_transform(config, train=False)
    results = []
    for name, image, truth in load_inputs(config, args.image):
        with torch.no_grad():
            probabilities = model(transform(image).unsqueeze(0)).softmax(dim=1)[0]
        if not torch.isfinite(probabilities).all():
            raise FloatingPointError("Xác suất demo không hữu hạn")
        predicted = CLASS_NAMES[int(probabilities.argmax())]
        result = {"image": name, "prediction": predicted, "confidence": float(probabilities.max()),
                  "truth": truth, "correct": predicted == truth if truth is not None else None,
                  "probabilities": dict(zip(CLASS_NAMES, probabilities.tolist()))}
        print(f"{name}: {predicted} ({result['confidence']:.2%}), nhãn thật: {truth}")
        results.append(result)
    figures = project_path(config["paths"]["reports_dir"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_probabilities(results, figures / "demo_probabilities.png")
    update_report(config, "demo", results)


if __name__ == "__main__":
    main()
