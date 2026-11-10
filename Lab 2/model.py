
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

NUM_CLASSES = 10


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    Build ResNet-18 for CIFAR-10 classification.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)

    # Replace the original classifier (1000 classes)
    # with a classifier for CIFAR-10 (10 classes).
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def freeze_backbone(model):
    """
    Transfer Learning:
    Freeze feature extraction layers and train only the classifier.
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def unfreeze_layer4(model):
    """
    Fine-Tuning:
    Unfreeze ResNet layer4 and the final classifier.
    """
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def get_trainable_parameters(model):
    """Return parameters that can be optimized."""
    return [
        param
        for param in model.parameters()
        if param.requires_grad
    ]


if __name__ == "__main__":
    # pretrained=False avoids downloading ImageNet weights
    # during the initial smoke test.
    model = build_model(num_classes=NUM_CLASSES, pretrained=False)

    # Test Transfer Learning
    freeze_backbone(model)

    transfer_params = sum(
        p.numel() for p in get_trainable_parameters(model)
    )

    print("Transfer Learning mode:")
    print("Trainable parameters:", transfer_params)

    # Test Fine-Tuning
    unfreeze_layer4(model)

    finetune_params = sum(
        p.numel() for p in get_trainable_parameters(model)
    )

    print("\nFine-Tuning mode:")
    print("Trainable parameters:", finetune_params)

    # Test model input and output
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    print("\nInput shape:", dummy_input.shape)
    print("Output shape:", output.shape)

    assert output.shape == (2, NUM_CLASSES)
    assert finetune_params > transfer_params

    print("\nModel smoke test: PASS")
