import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.solvers.diophantine.diophantine import Linear
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(), # Convert images to pyTorch tensors (0-1 range)
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize with MNIST mean and std
])

# Download and load training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST( root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Visualize a few examples
def plot_sample_images(loader):
    images, labels = next(iter(loader))
    fig, axes = plt.subplots(1, 5, figsize=(10,2))
    for i in range(5): # range() gives a array of integers between 0 and 4 (5 elements).
        axes[i].imshow(images[i][0], cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}') # item() is converting the label[i] from a PyTorch Tensor into a Python scalar
        axes[i].axis('off') # Hides the axis lines and tick marks on the i-th subplot for a cleaner display
    plt.show()

plot_sample_images(train_loader)

# Step 3: Define the neural network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten() # Flatten 28x28 images to 784-element vector
        self.layers = nn.Sequential(
            nn.Linear(28*28, 128), # Input: 784 pixels, output: 128 neurons
            nn.ReLU(), # Introduces the non linearity needed for efficient learning.
            nn.Linear(128, 64), # Hidden layer: 128 -> 64 neurons
            nn.ReLU(),
            nn.Linear(64, 10) # Output 64-> 10 (one per digit)
        )
