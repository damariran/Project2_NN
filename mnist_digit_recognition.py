import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Step 1: Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(), # Convert images to pyTorch tensors (0-1 range)
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize with MNIST mean and std
])

# Download and load training and test datasets
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)