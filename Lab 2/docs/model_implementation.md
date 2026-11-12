# Model Architecture

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | Model Architecture |
| Created date | 2026-11-10 (theo hồ sơ kế thừa) |
| Created by | Xuân Bảo (theo hồ sơ kế thừa; Git: hbao2123-debug) |
| Ghi nhận Git đầu tiên | File mới trong lần chuẩn hóa này |
| Last updated | 2026-11-11T11:13:30+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

> Historical implementation report from Xuan Bao. The parameter/report examples below describe the original standalone version. Current package interface and session commands: [model.md](model.md), [training_handoff.md](training_handoff.md). Old verification results remain attributed to their original run.

## 1. Overview

The Model module is responsible for implementing deep learning architectures for the CIFAR-10 image classification task, which consists of 10 classes.

The implementation uses PyTorch and Torchvision and supports two pretrained architectures:

- ResNet18
- DenseNet121

Both architectures use ImageNet pretrained weights through Transfer Learning. The feature extractor is frozen by default, while the original classification head is replaced with a new head suitable for 10-class classification.

**Implementation file:** `Lab 2/src/lab_2/model.py`; `Lab 2/model.py` is the compatibility entry point.

## 2. Objectives

The objectives of this module are:

- Implement and manage multiple model architectures.
- Reuse ImageNet pretrained weights.
- Apply Transfer Learning by freezing the feature extractor.
- Support Fine-tuning by selectively unfreezing model submodules.
- Replace the original classification head with Dropout and Linear layers.
- Calculate total and trainable parameter counts.
- Manage training and evaluation modes, especially BatchNorm behavior.
- Save and restore model checkpoints.
- Export model information to `metrics.json`.

## 3. Model Architectures

### 3.1. Supported Models

| Model | Architecture | Classification Head |
|---|---|---|
| ResNet18 | Residual Network | `fc` |
| DenseNet121 | Densely Connected Network | `classifier` |

The module defines the following constants:

- `NUM_CLASSES = 10`: number of target classes.
- `HEADS`: maps each supported model to its classification head.
- `SUPPORTED_MODELS`: contains the supported model names.

### 3.2. Classification Head

The original ImageNet classification head is replaced with:

```python
nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, NUM_CLASSES)
)
```

The new classification head consists of:

- Dropout with probability 0.2 to reduce overfitting.
- A Linear layer that maps extracted features to 10 output classes.

## 4. Transfer Learning and Fine-tuning

### 4.1. Transfer Learning

By default, the entire feature extractor is frozen using:

```python
model.requires_grad_(False)
```

A new classification head is then attached to the model and remains trainable.

During Transfer Learning, only the classification head parameters are updated.

### 4.2. Fine-tuning

The `build_model()` function supports selective unfreezing through the `unfreeze` argument.

Example:

```python
model = build_model(
    name="resnet18",
    dropout=0.2,
    unfreeze=["layer4"],
    pretrained=True
)
```

In this configuration:

- The classification head is trainable.
- The `layer4` module is unfrozen.
- All other feature-extractor modules remain frozen.

## 5. Main Functions

| Function | Description |
|---|---|
| `head_of(model)` | Returns the classification head of a model. |
| `build_model()` | Builds a model and configures Transfer Learning or Fine-tuning. |
| `trainable_parameters()` | Returns all trainable parameters. |
| `count_parameters()` | Calculates total and trainable parameter counts. |
| `set_train_mode()` | Configures training modes for trainable modules. |
| `save_checkpoint()` | Saves model weights and training metadata. |
| `load_checkpoint()` | Restores a model from a checkpoint. |
| `main()` | Builds supported configurations and exports the model report. |

### 5.1. BatchNorm Management

The `set_train_mode()` function first places the entire model in evaluation mode.

It then enables training mode for the classification head and modules that directly own trainable parameters.

This prevents frozen BatchNorm layers from updating their running statistics during training.

### 5.2. Checkpoint Management

The `save_checkpoint()` function stores:

- `state_dict`: model parameters.
- `name`: model architecture name.
- `dropout`: Dropout probability.
- `unfreeze`: list of unfrozen modules.
- `epoch`: checkpoint epoch.
- `val_accuracy`: validation accuracy.
- `val_loss`: validation loss.

The checkpoint is first written to a temporary file and then moved to its destination using `os.replace()` to reduce the risk of incomplete files.

The `load_checkpoint()` function reconstructs the model with `pretrained=False`, loads the saved state dictionary, moves the model to the requested device, and returns the restored model with its metadata.

## 6. Parameter Count Results

The actual execution of `model.py` produced the following results:

| Model | Total Parameters | Trainable Parameters |
|---|---:|---:|
| ResNet18 | 11,181,642 | 5,130 |
| DenseNet121 | 6,964,106 | 10,250 |
| ResNet18 + layer4 | 11,181,642 | 8,398,858 |

Observations:

- ResNet18 trains only its classification head in the default configuration.
- DenseNet121 has more trainable parameters in its classification head than ResNet18.
- Unfreezing `layer4` significantly increases the number of trainable ResNet18 parameters.
- The total parameter count remains unchanged when switching from Transfer Learning to Fine-tuning.

All three parameter counts match the expected values in the model specification.

## 7. Model Report

After executing `main()`, the model information is saved to:

```text
Lab 2/
└── reports/
    └── metrics.json
```

The report contains:

- Model architecture name.
- Dropout probability.
- Unfrozen modules.
- Total parameter count.
- Trainable parameter count.

The model information is stored under the `model` key. Existing report sections are preserved when the JSON file is updated.

## 8. How to Run

Run the following command from the repository root:

```powershell
python ".\Lab 2\model.py"
```

To check Python syntax:

```powershell
python -m py_compile ".\Lab 2\model.py"
```

The first execution with `pretrained=True` may download ImageNet pretrained weights through Torchvision.

## 9. Verification Results

The following table summarizes the verification results obtained so far:

| Verification | Status | Evidence |
|---|---|---|
| Python syntax check | PASS | `py_compile` completed without errors. |
| ResNet18 initialization | PASS | Expected parameter counts. |
| DenseNet121 initialization | PASS | Expected parameter counts. |
| ResNet18 layer4 fine-tuning | PASS | Expected parameter counts. |
| Model report generation | PASS | `metrics.json` created successfully. |
| Unit tests | PASS | Pytest: 14/14 tests passed, 0 failed in 7.48 seconds .. |

## 10. Conclusion

The Model module implements ResNet18 and DenseNet121 with ImageNet pretrained weights and supports both Transfer Learning and Fine-tuning.

It also provides parameter counting, BatchNorm-aware training mode management, and checkpoint save/load utilities.

Actual execution confirms that all three tested model configurations produce the expected parameter counts and that the model report is generated successfully.

The 7.48-second result above is the contributor-reported historical run. Preparation verification on 2026-11-11 reran all 14 compatibility tests and 49 package tests successfully (63 total). Model implementation is now integrated into src/lab_2/model.py; the legacy JSON report remains unchanged. New package reports use total/trainable keys. Real CIFAR-10 smoke, full training and Colab execution remain pending; see verification.md and training_handoff.md.
