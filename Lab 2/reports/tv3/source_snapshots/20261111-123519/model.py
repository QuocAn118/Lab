


import json
import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models


NUM_CLASSES = 10

HEADS = {
    "resnet18": "fc",
    "densenet121": "classifier",
}

SUPPORTED_MODELS = tuple(HEADS)


def head_of(model):
    """Return the classification head of a supported model."""
    if hasattr(model, "fc"):
        return model.fc

    if hasattr(model, "classifier"):
        return model.classifier

    raise ValueError("Model does not have a supported classification head.")


def build_model(name, dropout=0.2, unfreeze=(), pretrained=True):
    """
    Build a supported model for CIFAR-10.

    Freeze the feature extractor by default.
    Only the classification head is trainable unless
    additional submodules are explicitly unfrozen.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {name}. "
            f"Supported models: {SUPPORTED_MODELS}"
        )

    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the range [0, 1).")

    weights = "DEFAULT" if pretrained else None
    model = getattr(models, name)(weights=weights)

    # Freeze the original model.
    model.requires_grad_(False)

    if isinstance(unfreeze, str):
        unfreeze = [unfreeze]

    # Unfreeze requested feature-extractor modules.
    for module_name in unfreeze:
        try:
            model.get_submodule(module_name).requires_grad_(True)
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                f"Invalid submodule '{module_name}' "
                f"for model '{name}'."
            ) from exc

    # Replace the original classification head.
    old_head = head_of(model)
    in_features = old_head.in_features

    new_head = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )

    if HEADS[name] == "fc":
        model.fc = new_head
    else:
        model.classifier = new_head

    return model


def trainable_parameters(model):
    """Return all parameters that can be optimized."""
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in trainable_parameters(model)
    )

    return {
        "total": total,
        "trainable": trainable,
    }


def set_train_mode(model):
    """
    Enable training mode only for modules with trainable parameters.

    Frozen BatchNorm layers remain in evaluation mode.
    """
    model.eval()

    # The classification head must always be trainable.
    head_of(model).train()

    # Enable training mode for modules owning trainable parameters.
    for module in model.modules():
        direct_parameters = module.parameters(recurse=False)

        if any(
            parameter.requires_grad
            for parameter in direct_parameters
        ):
            module.train()

    return model


def save_checkpoint(
    model,
    path,
    *,
    name,
    dropout,
    unfreeze,
    epoch,
    val_accuracy,
    val_loss,
):
    """Save model weights and metadata atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": model.state_dict(),
        "name": name,
        "dropout": dropout,
        "unfreeze": list(unfreeze),
        "epoch": epoch,
        "val_accuracy": val_accuracy,
        "val_loss": val_loss,
    }

    temp_path = path.with_name(path.name + ".tmp")

    try:
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_checkpoint(path, device="cpu"):
    """
    Load a checkpoint and reconstruct its model architecture.

    Pretrained weights are disabled because the checkpoint
    already contains the trained model parameters.
    """
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model = build_model(
        name=checkpoint["name"],
        dropout=checkpoint["dropout"],
        unfreeze=checkpoint["unfreeze"],
        pretrained=False,
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    info = {
        key: value
        for key, value in checkpoint.items()
        if key != "state_dict"
    }

    return model, info


def main():
    """
    Build supported models and save parameter information
    to Lab 2/reports/metrics.json.
    """
    dropout = 0.2

    model_configs = {
        "resnet18": {
            "name": "resnet18",
            "unfreeze": [],
        },
        "densenet121": {
            "name": "densenet121",
            "unfreeze": [],
        },
        "resnet18_unfreeze_layer4": {
            "name": "resnet18",
            "unfreeze": ["layer4"],
        },
    }

    # Resolve paths relative to this file.
    lab_dir = Path(__file__).resolve().parent
    report_dir = lab_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "metrics.json"

    # Preserve metrics written by other Lab 2 components.
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as file:
            full_report = json.load(file)
    else:
        full_report = {}

    model_report = {}

    for run_name, model_config in model_configs.items():
        print(f"\n{'=' * 70}")
        print(f"Model: {run_name}")

        model = build_model(
            name=model_config["name"],
            dropout=dropout,
            unfreeze=model_config["unfreeze"],
            pretrained=True,
        )

        counts = count_parameters(model)

        print(f"Total parameters: {counts['total']:,}")
        print(f"Trainable parameters: {counts['trainable']:,}")

        model_report[run_name] = {
            "model": model_config["name"],
            "dropout": dropout,
            "unfreeze": model_config["unfreeze"],
            "total_parameters": counts["total"],
            "trainable_parameters": counts["trainable"],
        }

    full_report["model"] = model_report

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            full_report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nModel report saved to: {report_path}")


if __name__ == "__main__":
    main()
