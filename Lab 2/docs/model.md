# Model Architecture
Created at: 10/11/2026          
 Auth: Xuân Bảo 

## 1. Overview

The project uses **ResNet-18** as the baseline model for image classification on the CIFAR-10 dataset.

ResNet-18 is a convolutional neural network (CNN) that uses residual connections to support deeper network training and improve feature extraction.

The model is adapted from ImageNet classification (1000 classes) to CIFAR-10 classification (10 classes).

## 2. Model Architecture

The implementation uses `torchvision.models.resnet18`.

Main configuration:

- Architecture: ResNet-18
- Framework: PyTorch
- Input shape: `(batch_size, 3, 224, 224)`
- Output classes: 10
- Final classification layer: `nn.Linear(model.fc.in_features, 10)`

The original fully connected layer is replaced with a new layer containing 10 output units to match the CIFAR-10 classes.

## 3. Transfer Learning

Transfer Learning uses pretrained ImageNet weights.

The implementation freezes all feature extraction layers by setting `requires_grad = False`.

Only the final fully connected classification layer (`model.fc`) is trainable.

This allows the model to reuse pretrained visual features while learning to classify the 10 CIFAR-10 classes.

Implementation: `freeze_backbone(model)`.

## 4. Fine-Tuning

Fine-Tuning allows selected pretrained layers to be updated for the target dataset.

The implementation:

- Freezes all model parameters initially.
- Unfreezes the final residual block (`layer4`).
- Keeps the final classification layer (`model.fc`) trainable.

This allows the model to adapt higher-level image features to CIFAR-10 while retaining lower-level pretrained features.

Implementation: `unfreeze_layer4(model)`.

## 5. Trainable Parameters

The function `get_trainable_parameters(model)` returns all parameters whose `requires_grad` attribute is `True`.

This helps configure the optimizer to update only the selected layers in each training mode.

## 6. Model Testing

A smoke test is included in `model.py` to verify:

- The model can be initialized.
- Transfer Learning freezes the backbone.
- Fine-Tuning unfreezes `layer4` and the classifier.
- The model accepts an input tensor with shape `(2, 3, 224, 224)`.
- The model produces an output tensor with shape `(2, 10)`.

### Test Results

| Test | Result |
|---|---|
| Transfer Learning initialization | PASS |
| Fine-Tuning initialization | PASS |
| Input shape | `(2, 3, 224, 224)` |
| Output shape | `(2, 10)` |
| Trainable parameters in Transfer Learning | 5,130 |
| Trainable parameters in Fine-Tuning | 8,398,858 |
| Model smoke test | PASS |

## 7. Notes

- The model uses `pretrained=True` by default to load ImageNet pretrained weights.
- The smoke test uses `pretrained=False` to avoid downloading weights during initial verification.
- Input images must be resized and normalized according to the preprocessing configuration in `data.md`.
- Training configuration and evaluation metrics are documented separately in `training.md` and `evaluation.md`.