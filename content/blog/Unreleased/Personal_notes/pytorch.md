---
tags:
  - Computer
  - Vision
aliases: 
publish: false
slug: python-notes
title: python-notes
description: notes on Python
date: 2024-11-26
image: /thumbnails/backbones.png
---
# Pytorch

PyTorch is an open-source machine learning framework.

**Used for**
- Building and training neural networks

**Why Use PyTorch?**
- Efficient (GPU)
- Flexible
- Ease of use
- Community support and comprehensive ecosystem

**Key features**
* constructs **computation graphs** dynamically during runtime, enabling easier debugging and experimentation.
* provides support for multi-dimensional arrays (tensors) and operations on them, leveraging GPUs for accelerated computation
* `torch.nn` provides pre-built layers, activation functions, loss functions and utilities for training
* `torch.autograd`implements automatic differentiation, computing gradients for tensor operations and handling backpropagation automatically 
* Hardware acceleration: integrates seamlessly with CUDA and supports GPU computations
* Vibrant ecosystem with many libraries such as `torchtext` (NLP), `torchvision` (CV) 
* `TorchServe` for deploying Pytorch models in production

*On computation graphs:*  In PyTorch, the computation graph is **created on-the-fly during runtime**. Every time you execute a forward pass, PyTorch dynamically constructs the graph for that specific computation, based on the input and operations you use.

# Modules

## torch module

`torch` is the base module of Pytorch, providing the `tensor` class along with mathematical operations

```
import torch
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
c = a + b  # Element-wise addition
```

## torch.nn

The neural network module provides:
* Built in layers
* Built in loss functions

| Layer                  | Description                                                                                         |
|------------------------|-----------------------------------------------------------------------------------------------------|
| `nn.Linear`            | Fully connected layer (dense layer), performs a linear transformation.                             |
| `nn.Conv2d`            | 2D convolutional layer, commonly used in image processing tasks.                                   |
| `nn.Conv1d`            | 1D convolutional layer, commonly used in sequence data like time series or audio.                  |
| `nn.LSTM`              | Long Short-Term Memory (LSTM) layer for sequence modeling and recurrent neural networks.            |
| `nn.GRU`               | Gated Recurrent Unit (GRU) layer, similar to LSTM, but with fewer parameters.                       |
| `nn.RNN`               | Basic Recurrent Neural Network layer for sequence modeling.                                        |
| `nn.BatchNorm2d`       | 2D batch normalization layer, used to normalize the inputs of each layer to improve training.       |
| `nn.Dropout`           | Regularization technique that randomly zeros some of the elements of the input tensor during training. |
| `nn.MaxPool2d`         | Max pooling layer for downsampling in 2D (typically used in CNNs).                                 |
| `nn.AvgPool2d`         | Average pooling layer for downsampling in 2D.                                                      |
| `nn.Embedding`         | Layer for learning dense representations of categorical variables, often used in NLP tasks.        |
| `nn.AdaptiveAvgPool2d` | Adaptive average pooling layer that outputs a fixed-size output regardless of the input size.       |
| `nn.LayerNorm`         | Layer normalization, normalizes the input across the features for each sample.                      |
| `nn.Transformer`       | Transformer architecture layer, used for sequence-to-sequence tasks such as translation.            |

| Loss Function                 | Description                                                                                         |
|--------------------------------|-----------------------------------------------------------------------------------------------------|
| `nn.MSELoss`                   | Mean Squared Error Loss, commonly used for regression tasks.                                        |
| `nn.CrossEntropyLoss`          | Cross-entropy loss, used for classification tasks (combines `softmax` and `negative log likelihood`).|
| `nn.BCELoss`                   | Binary Cross-Entropy Loss, used for binary classification tasks.                                    |
| `nn.BCEWithLogitsLoss`         | Binary Cross-Entropy Loss with logits, for binary classification with raw scores (logits).           |
| `nn.NLLLoss`                   | Negative Log Likelihood Loss, used for classification when the outputs are log probabilities.        |
| `nn.L1Loss`                    | L1 Loss, also known as Mean Absolute Error (MAE), used for regression tasks where we minimize absolute differences. |
| `nn.HingeEmbeddingLoss`        | Hinge embedding loss, often used for support vector machines and binary classification with margins.  |
| `nn.SmoothL1Loss`              | Smooth L1 Loss, a combination of L1 and L2 loss, used for regression tasks and robust to outliers.    |
| `nn.KLDivLoss`                 | Kullback-Leibler Divergence Loss, measures the difference between two probability distributions.      |
| `nn.MarginRankingLoss`         | Loss function that provides margin-based ranking for pairwise comparison tasks.                      |
| `nn.CosineEmbeddingLoss`       | Cosine similarity loss used for measuring the similarity between two tensors in embedding space.     |
| `nn.MultiLabelMarginLoss`      | Multi-label classification loss for tasks where each example can belong to more than one class.      |
| `nn.PoissonNLLLoss`            | Poisson negative log-likelihood loss, typically used for count-based predictions.                    |
 Offers utilities for defining, training, and running neural networks.

```
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)
```

- `torch.nn.Module`, which is the base class for all neural network layers, loss functions and other modules.
	- stores and registers parameters, like weights and biases
	- manages child modules
	- defines the `forward()` method that describes how input tensors flow through the module
	- Module tracks the parameters of the model and is integral to managing the gradients
- `torch.nn.Sequential` is a container for stacking layers in a linear fashion
- `torch.nn.init` provides functions for initializing the parameters of a model
- `torch.nn.functional` contains common functions such as `relu` , `sigmoid`, etc

example of custom layer:

``` python
import torch
import torch.nn as nn

# Custom Layer Example
class MyCustomLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyCustomLayer, self).__init__()
        # Define a custom linear layer as an example
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Custom forward pass, here we just apply the linear layer
        return self.linear(x)

# Example usage
model = MyCustomLayer(10, 5)  # Input size 10, output size 5
input_data = torch.randn(1, 10)  # Batch size of 1
output_data = model(input_data)  # Forward pass
```

example of custom loss function:

```python
import torch
import torch.nn as nn

class MyCustomLoss(nn.Module):
    def __init__(self):
        super(MyCustomLoss, self).__init__()

    def forward(self, input, target):
        # Custom loss function (mean absolute error)
        return torch.mean(torch.abs(input - target))

# Example usage
loss_fn = MyCustomLoss()
input_data = torch.randn(10)
target_data = torch.randn(10)
loss = loss_fn(input_data, target_data)
```

## torch.autograd

The automatic differentiation engine. Tracks operations on tensors to compute gradients automatically for backpropagation.

`tensor.backward()` or `torch.autograd.grad` computes gradients:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # Output: 4.0
```

PyTorch automatically tracks operations on tensors that require gradients. When you perform operations like addition, multiplication, or more complex operations (e.g., matrix multiplication, convolution), the resulting tensor will record the operation that created it, as well as the tensors involved. This is done via the **`requires_grad=True`** flag on tensors.

Once the computation is done, you can call the **`backward()`** method on the final tensor (usually the loss). This triggers **autograd** to traverse the computation graph and compute gradients using the **chain rule** of calculus.

For each operation in the graph, `autograd` knows the derivative of that operation (e.g., for multiplication, the derivative with respect to one of the operands is just the other operand).

## torch.optim

Provides optimizers like SGD, Adam, RMSprop, etc., for updating model parameters during training.

```
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update parameters
```

## torch.utils.data

Data handling utilities. Supports creating datasets and data loaders for managing training/validation data.

Includes `Dataset` and `DataLoader` classes for easy data manipulation and batching.

```python
from torch.utils.data import DataLoader, TensorDataset
dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
loader = DataLoader(dataset, batch_size=16, shuffle=True)
for x, y in loader:
    pass  # Your training loop here
```

The `Dataset` class in PyTorch is the base class for creating and working with datasets, and requires you to implement two essential methods:
- **`__len__`**: Returns the total number of samples in the dataset.
- **`__getitem__(self, index)`**: Retrieves a single sample by its index.
- the code that handles opening and reading data from files on disk is typically placed in the **constructor `__init__`** 

The `Dataset` class works seamlessly with PyTorch’s `DataLoader`, which provides features like batching, shuffling, and parallel loading.

```python
from torch.utils.data import DataLoader

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

# Iterate through batches
for batch in dataloader:
    x_batch, y_batch = batch
    print(f"Batch shape: {x_batch.shape}, Labels: {y_batch}")

```
## torchvision

CV specific library providing:

- `torchvision.models` - several popular deep learning models pre-trained on large datasets like ImageNet e.g. classification (ResNet, VGG) ; OD Faster R-CNN ; SS DeepLabV3, ETC.
- `torchvision.datasets`- standard datasets for training CV models, often used for benchmarks e.g. ImageNet, MNIST, etc.
- `torchvision.transforms` - Image transformation utilities to preprocess images e.g. Resize, RnadomCrop, RandomHorizontalFlip, etc.
- etc

e.g.

```python
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision import datasets

# Transformations (e.g., resizing and normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load a dataset (e.g., CIFAR-10)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load a pre-trained model
model = models.resnet18(pretrained=True)

# Fine-tune the model on your custom dataset
for images, labels in train_loader:
    outputs = model(images)
    # Further training code here
```

# Complete example using Pytorch vanilla

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. Define the Network
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(5408, 10)  # Adjust dimensions to match input size
        )

    def forward(self, x):
        return self.network(x)

# 2. Prepare Data
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# 3. Initialize Model, Loss, and Optimizer
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 5. Evaluation
# Add test logic

```

# Pytorch lightning

PyTorch itself provides the building blocks for defining and training neural networks, **PyTorch Lightning** simplifies the process by standardizing the structure of deep learning projects. It abstracts much of the boilerplate code, allowing you to focus on the model logic and experimentation.

The previous example becomes:

```python
import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define LightningModule
class MyLightningModel(pl.LightningModule):
    def __init__(self):
        super(MyLightningModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(5408, 10)  # Adjust dimensions to match input size
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        loss = self.criterion(output, target)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.001)

# Define DataModule (Optional but encouraged)
class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        datasets.MNIST(root='./data', train=True, download=True)
        datasets.MNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = datasets.MNIST(root='./data', train=True, transform=self.transform)
        self.val_dataset = datasets.MNIST(root='./data', train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

# Instantiate and Train
data_module = MyDataModule()
model = MyLightningModel()

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule=data_module)
```

The key features of Pytorch Lightning are:
- Abstraction of boilerplate code
- Standardized workflow
	- Model definition goes to `LightningModule`
	- Data loading goes to `LightningDataModule`
	- Training and evaluation go to `Trainer`

# Common directory layout - lightning init app 

We can use

```bash
pip install lightning lightning init app --name my_project
```

to generate a predefined project structure:

```bash
my_project/
├── configs/
│   └── config.yaml      # Configuration file for the project
├── data/
│   └── data_module.py   # Data module for handling data loading and preprocessing
├── models/
│   └── my_model.py      # Example model architecture
├── training/
│   └── train.py         # Main training script
├── tests/
│   └── test_model.py    # Example test case
├── logs/                # Placeholder for training logs
├── README.md            # Overview of the project
└── requirements.txt     # Dependencies
```

Example workflow:

1. Modify the Data Module. Update `data_module.py` to load and preprocess your dataset.
2. Define the Model. Customize `my_model.py` to implement your architecture.
3. Update Training Script. Configure `train.py` to handle additional hyperparameters or custom callbacks.
4. Train the Model `python training/train.py`
5. Evaluate the Model - Add evaluation scripts and validate your model's performance.