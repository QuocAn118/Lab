# MLP Architecture for Image Classification — Lab 1

## 0. Thông tin tài liệu

| Trường | Nội dung |
| --- | --- |
| Name | MLP Architecture for Image Classification — Lab 1 |
| Created date | Chưa xác nhận |
| Created by | Nguyen Thi Van Anh (theo hồ sơ kế thừa) |
| Ghi nhận Git đầu tiên | 2026-11-04T10:34:00+07:00 ; HoangDang ; aee3dd8 |
| Last updated | 2026-11-11T09:54:14+07:00 |
| Updated by | HoangDang (Nguyễn Hoàng Đăng), hỗ trợ OpenAI Codex |
| Phạm vi | Tài liệu đang dùng; phân biệt đặc tả, code hiện có và kiểm chứng runtime |

## 1. Model Name
- Model name: FashionMNIST MLP
- Model class: FashionMLP
- Author: Nguyen Thi Van Anh
- Created on: chưa xác nhận
- Implementation status: Full pipeline completed; four trained runs are available. Selected by validation: run_004.
- Framework: PyTorch
- Reference: PyTorch – Learn the Basics – Build the Neural Network

## 2. Purpose
The model is designed to perform multiclass image classification on the FashionMNIST dataset. For each input image, the model predicts one of 10 clothing and accessory classes:

| Label | Class |
|:---:|:---:|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

The model uses a Multi-Layer Perceptron (MLP) architecture as the baseline for this lab. During training, the model updates its weights to learn the relationship between image pixel values and their corresponding labels

**Objectives of model construction:**
- Define the network using a class that inherits from `torch.nn.Module`
- Build the forward computation flow from input images to classification scores
- Ensure that the input and output dimensions match the task requirements
- Determine the total number of parameters and the number of trainable parameters
- Support saving and reloading weights for use after training
- Verify that the output after reloading matches the output before saving

## 3. Initial Model
### 3.1 Model architecture
The model uses an MLP architecture consisting of:
- One flatten layer
- Two hidden Linear layers, each with 512 neurons
- A ReLU activation function after each hidden layer
- One output Linear layer with 10 neurons

The model is defined in the FashionMLP class, which inherits from `torch.nn.Module`, and constructed through `build_model(model_config)`.
The `__init__` method declares the layers
The `forward` method defines the order in which data passes through the layers

### Detailed architecture

| Component | Configuration | Function |
|---|---|---|
| Flatten | `nn.Flatten()` | Flattens each image into a vector of 784 values |
| Hidden layer 1 | `nn.Linear(784, 512)` | Transforms 784 input values into a representation of 512 values |
| Activation 1 | `nn.ReLU()` | Introduces nonlinearity after the first hidden layer |
| Hidden layer 2 | `nn.Linear(512, 512)` | Further transforms the representation of each image |
| Activation 2 | `nn.ReLU()` | Introduces nonlinearity after the second hidden layer |
| Output layer | `nn.Linear(512, 10)` | Produces 10 scores corresponding to the 10 classes |

Flatten only changes the tensor shape; it does not change pixel values and has no learnable parameters

Each Linear layer applies a transformation using weights and biases
These parameters are updated during training

ReLU replaces negative values with 0 and leaves positive values unchanged
This activation function enables the network to learn nonlinear relationships. ReLU has no learnable parameters

The output layer returns logits. Softmax is not included in the forward method when training with nn.CrossEntropyLoss because this loss function accepts logits directly

## 3.2 Model input
The input is a batch of FashionMNIST images that have been converted into tensors

- Data type: `torch.float32`
- Shape: (B, 1, 28, 28)
- B: number of images in the batch
- 1: one grayscale channel
- 28 × 28: image height and width
- Value range: [0, 1] when applying `ToTensor()`

## 3.3 Model output
The output is a tensor of logits with shape (B, 10)

Each row corresponds to one image in the batch<br>
Each column corresponds to the score for one class<br>
For example:

| | Column 0 (T-shirt/top) | Column 1 (Trouser) | Column 2 (Pullover) | Column 3 (Dress) | ... | Column 9 (Ankle boot) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Image 0 (row 0) | 0.2 | -1.1 | 0.5 | -0.3 | ... | 3.8 |
| Image 1 (row 1) | 2.9 | 0.1 | 1.4 | 0.0 | ... | -0.7 |
| Image 2 (row 2) | -0.5 | 4.2 | -0.9 | 1.0 | ... | 0.3 |

Logits are raw scores that can be negative or positive<br>
They are not required to fall within [0, 1] or sum to 1

## 3.4 Forward flow and tensor shapes

| Step | Output shape |
|:---|:---:|
| Input | (B, 1, 28, 28) |
| Flatten | (B, 784) |
| Linear(784, 512) | (B, 512) |
| ReLU | (B, 512) |
| Linear(512, 512) | (B, 512) |
| ReLU | (B, 512) |
| Linear(512, 10) | (B, 10) |

Each input image produces a set of 10 output scores

## 3.5 Number of parameters

For a Linear layer with bias:<br>
**Number of parameters** = `in_features` × `out_features` + `out_features`

Where:
- `in_features` × `out_features` is the number of weight elements
- `out_features` is the number of bias elements

| Layer | Weight | Bias | Total |
|---|:---:|:---:|:---:|
| Linear(784, 512) | 401408 | 512 | 401920 |
| Linear(512, 512) | 262144 | 512 | 262656 |
| Linear(512, 10) | 5120 | 10 | 5130 |
| Total | 668672 | 1034 | 669706 |

- **_Total parameters_**: 669706
- **_Trainable parameters_**: 669706
- No layers are frozen in the initial configuration
- Flatten and ReLU have no learnable parameters

## 3.6 Initial state

The model is initialized from scratch using PyTorch's default initialization mechanism, without pretrained weights

Before training, the model can already perform a forward pass and return logits, but its weights have not yet been optimized for FashionMNIST classification

## 3.7. Source file

- Source file: [src/model.py](../src/model.py)
- Model class: `FashionMLP`; factory: `build_model(model_config)`
- The current implementation supports exactly two positive integer hidden dimensions and 10 output classes.

**_Reference:_**
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

## 3.8. Planned verification
Before handing over the model for training:
1. Create or retrieve a batch with shape (B, 1, 28, 28)
2. Run a forward pass and verify that the output shape is (B, 10)
3. Check that the output contains no NaN or Inf values
4. Count the total number of parameters and the number of trainable parameters
5. Record the actual results in `model_summary.txt`

**_Verification status:_** A saved [model_summary.txt](../outputs/model_summary.txt) records input `(2, 1, 28, 28)`, output `(2, 10)` and 669706 trainable parameters. This is an architecture check, not training evidence. The summary was regenerated from committed code on 2026-11-11; its matching sidecar preserves the previous generation history. This remains an architecture check, separate from training evidence.

`check_model()` now records the summary sidecar and preserves its history. A regression test checks the generated file hash.

# 4. Trained Model

Selected run: **run_004**, chosen by highest validation accuracy across the four runs. No test value was used for selection.

- Training: 2026-11-11T09:45:49+07:00 to 2026-11-11T09:46:06+07:00, HoangDang on RTX 5060 Laptop.
- Architecture remains 784 → 512 → 512 → 10; Adam learning rate 0.001; 4 epochs, early stopping true under patience 2/min_delta 0.01.
- Best validation accuracy: 0.8810; test accuracy: 0.8770, test loss: 0.346382, 10,000 test images.
- Weights: `checkpoints/run_004/best.pth`; [config](../outputs/run_004/config.json), [checkpoint index](../checkpoints/run_004/checkpoint_index.json).
- [Reload check](../outputs/run_004/reload_check.json): passed, 16 validation samples, labels equal, max logit difference 0 on CPU.
- [Performance](../outputs/run_004/performance.json), [metrics](../outputs/run_004/metrics.json), [manifest](../outputs/run_004/manifest.json).

The original implementation author remains Nguyen Thi Van Anh according to the inherited profile. Completion/integration is by HoangDang with OpenAI Codex. The original creation date is still unconfirmed; commit dates are not substituted for that date. No verified Drive upload link is available; local weights are included in the handoff ZIP.

## Nguồn và phạm vi

- [Đặc tả](lab_1_description.md), [3.3](requirements/3_3_build_model.md), [3.6](requirements/3_6_save_load.md), source model and measured artifacts above.
