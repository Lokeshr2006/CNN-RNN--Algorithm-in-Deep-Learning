# CNN from Scratch (NumPy) — MNIST Digit Classifier

A convolutional neural network built completely from scratch using **only NumPy** — no PyTorch/TensorFlow autograd, no pre-built layers. Every component (convolution, pooling, dense layers, backpropagation) is implemented manually to understand exactly how a CNN learns under the hood.

## Why this project
Most CNN tutorials rely on `model.fit()` or `loss.backward()`, which hides the actual math. This project implements the full forward and backward pass by hand — including manual gradient computation for convolution filters, max-pool index tracking, and dense layer weight updates — to build a real understanding of backpropagation mechanics in CNNs.

## Features
- **Convolution layer**: custom sliding-window 2D convolution with configurable filter count, size, and stride
- **ReLU activation**: applied after convolution and hidden dense layer
- **Max-pooling**: 2x2 pooling with max-index caching (needed to route gradients correctly during backprop)
- **Fully connected layers**: hidden layer (32 neurons) → output layer (10 classes)
- **Softmax + Cross-Entropy Loss**: for multi-class digit classification
- **Manual backpropagation**: gradients derived and coded layer-by-layer, no autograd
- **Training loop**: configurable number of images, epochs, and learning rate, with live loss/accuracy logging
- **Testing loop**: evaluates trained weights on unseen MNIST test images

## Architecture
```
Input (28x28 grayscale)
      ↓
Conv2D (4 filters, 3x3, stride 1)
      ↓
ReLU
      ↓
MaxPool (2x2, stride 2)
      ↓
Flatten (4 x 13 x 13 = 676)
      ↓
Dense (32 neurons) → ReLU
      ↓
Dense (10 neurons) → Softmax
      ↓
Predicted Digit (0-9)
```

## Requirements
```
torch          # only used for torchvision.datasets
torchvision
numpy
```

Install with:
```bash
pip install torch torchvision numpy
```

## Usage
```bash
python cnn_mnist.py
```
You'll be prompted to enter:
1. Number of training images to use
2. Number of training epochs (default: 3) and learning rate (default: 0.01) — set in code
3. Whether to run testing after training (`y`/`n`)

## Example Output
```
Full training set available: 60000 images
How many images should I train on? 2000
Epoch 1/3 - avg loss: 1.2345 - accuracy: 68.20%
Epoch 2/3 - avg loss: 0.7891 - accuracy: 79.55%
Epoch 3/3 - avg loss: 0.5432 - accuracy: 85.10%
Test the images? (y/n): y
850/1000 images is correct
```

## Limitations
- Pure Python loops for convolution/pooling make training slow on large datasets — intended for learning, not production speed
- Fixed architecture (filter count, layer sizes) — not yet parameterized via config
- No validation split; only train/test accuracy tracked

## Possible Extensions
- Vectorize convolution using `im2col` for speed
- Add configurable architecture (more conv layers, dropout, batch norm)
- Add momentum/Adam optimizer instead of plain SGD
- Save/load trained weights

## License
Educational project — free to use and modify.
