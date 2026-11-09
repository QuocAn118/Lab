# MLP Architecture for Image Classification — Lab 1
## 1. Model Name
- Model name: FashionMNIST MLP
- Model class: FashionMLP
- Author: Nguyen Thi Van Anh
- Created on: 09/11/2026
- Updated on: 09/11/2026
- Updated by: Nguyen Thi Van Anh
- Framework: PyTorch
- Reference: https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
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

The model is defined in the FashionMLP class, which inherits from `torch.nn.Module`
The `__init__` method declares the layers
The `forward` method defines the order in which data passes through the layers

The baseline configuration uses two hidden layers with 512 neurons each<br>
The current implementation allows changing the width of these two hidden layers through `model_config["hidden_dims"]`

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

Each Linear layer applies a transformation using weights and biases<br>
These parameters are updated during training

ReLU replaces negative values with 0 and leaves positive values unchanged<br>
This activation function enables the network to learn nonlinear relationships. ReLU has no learnable parameters

The output layer returns logits. Softmax is not included in the forward method when training with `nn.CrossEntropyLoss` because this loss function accepts logits directly

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

- Source file: `src\model.py`
- Model class: `FashionMLP`
- Model creation function: `build_model(model_config)`
- Architecture verification function: `check_model()`
- Verification command: `python src/model.py --check`
- Verification report: `outputs/model_summary.txt`

**_Reference:_**
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

## 3.8. Model verification
Before handing over the model for training:
1. Create or retrieve a batch with shape (B, 1, 28, 28)
2. Run a forward pass and verify that the output shape is (B, 10)
3. Check that the output contains no NaN or Inf values
4. Count the total number of parameters and the number of trainable parameters
5. Record the actual results in `model_summary.txt`

**_Verification status:_**<br>
The model architecture was manually verified against the configuration
using the printed model structure<br>
| Check | Actual result | Status |
|---|---|---|
| Baseline architecture | 784 → 512 → 512 → 10, with ReLU after each hidden layer | Passed |
| Input shape | (2, 1, 28, 28) | Passed |
| Output shape | (2, 10) | Passed |
| Output values | All values are finite; no NaN or Inf | Passed |
| Total parameters | 669706 | Matches the calculated count |
| Trainable parameters | 669706 | All parameters are trainable |
| Report generation | `outputs/model_summary.txt` generated and inspected | Passed |