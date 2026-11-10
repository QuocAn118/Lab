import json
from pathlib import Path
import torch
from model import build_model

project_root = Path(__file__).resolve().parent.parent

def save_model(model, run_id):
    run_dir = project_root / "outputs" / run_id
    config_path = run_dir / "config.json"

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục lần chạy: {run_dir}")

    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")

    checkpoint_dir = project_root / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "final_epoch.pth"

    with checkpoint_path.open("xb") as f:
        torch.save(model.state_dict(), f)

    return checkpoint_path

def load_model(run_id, device):
    config_path = project_root / "outputs" / run_id / "config.json"
    checkpoint_path = (project_root / "checkpoints" / run_id / "final_epoch.pth")

    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy cấu hình: {config_path}")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_config = config["model"]
    model = build_model(model_config).to(device)

    state_dict = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=True)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model

def verify_reload(model, sample_batch, run_id):
    if not isinstance(sample_batch, torch.Tensor):
        raise TypeError("sample_batch phải là tensor ảnh")
    if tuple(sample_batch.shape) != (16, 1, 28, 28):
        raise ValueError("sample_batch phải có shape (16, 1, 28, 28)")

    model = model.to("cpu")
    model.eval()
    loaded_model = load_model(run_id, device="cpu")
    images = sample_batch.detach().to("cpu")

    with torch.no_grad():
        original_logits = model(images)
        loaded_logits = loaded_model(images)

    if original_logits.shape != (16, 10) or loaded_logits.shape != (16, 10):
        raise ValueError("Logits của cả hai mô hình phải có shape (16, 10)")
    if not (torch.isfinite(original_logits).all().item()
            and torch.isfinite(loaded_logits).all().item()):
        raise ValueError("Logits chứa NaN hoặc Inf, không thể đối chiếu")

    labels_match = torch.equal(
        original_logits.argmax(dim=1), loaded_logits.argmax(dim=1)
    )
    max_difference = (original_logits - loaded_logits).abs().max().item()
    tolerance = 0.000001
    passed = labels_match and max_difference <= tolerance

    report = {
        "run_id": run_id,
        "checkpoint": f"checkpoints/{run_id}/final_epoch.pth",
        "device": "cpu",
        "checked_samples": images.shape[0],
        "labels_match": labels_match,
        "max_abs_logit_difference": max_difference,
        "tolerance": tolerance,
        "passed": passed,
    }
    report_path = project_root / "outputs" / run_id / "reload_check.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

    return passed
