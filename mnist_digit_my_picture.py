from tkinter import Image

import matplotlib.pyplot as plt
from simple_functions import simple_scatter
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

# model learning parameters
epochs =10  # The number of training cycles.
learning_rate = 0.005
number_of_pictures_in_batch =64
first_layer = 128
second_layer = 64
# Step 1: Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(), # Convert images to pyTorch tensors (0-1 range)
    transforms.Normalize((0.1307,), (0.3081,)) # Normalize with MNIST mean and std
])

# Step 1: Download and load training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST( root='./data', train=False, download=True, transform=transform)

# Step 2: Create data loaders
train_loader = DataLoader(train_dataset, batch_size=number_of_pictures_in_batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=number_of_pictures_in_batch, shuffle=False)

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
            nn.Linear(28*28, first_layer), # Input: 784 pixels, output: 128 neurons
            nn.ReLU(), # Introduces the non linearity needed for efficient learning.
            nn.Linear(first_layer, second_layer), # Hidden layer: 128 -> 64 neurons
            nn.ReLU(),
            nn.Linear(second_layer, 10) # Output 64-> 10 (one per digit)
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

#Instantiate the model
model  = DigitClassifier()

# Step 4: Define loss function and optimizer
criterion = nn.CrossEntropyLoss() # For multi-class classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 5: Train the model
print('Starting Training loop')
train_losses = []  # for plotting later
for epoch in range(epochs):
    model.train() # set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad() # clear all the gradients
        outputs = model(images) # forward pass
        loss = criterion(outputs, labels) # compute the loss with the CrossEntropyLoss() function
        loss.backward() # Backward pass
        optimizer.step() # update weights
        running_loss +=loss.item()
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/ {epochs}], Loss: {(avg_loss*100):.4f}%')

# Step 6: Function to preprocess and predict my own image
def predict_custom_image(image_path, model):
    # Load the image
    img = Image.open(image_path).convert('L') # Convert to grayscale ('L' mode)
    # resize to 28x28 pixels
    img = img.resize((28, 28), Image.Resampling.LANCZOS) # High-quality resizing
    # Convert to tensor and normalize (Match MNIST Preprocessing)
    img_tensor = transforms.ToTensor()(img) # Shape: [1,28,28], values 0-1
    # Invert the image (to white image on black background)
    img_tensor = 1- img_tensor # flip pixel values
    # normalize to match MNIST
    img_tensor = transforms.Normalize((0.1307,), (0.3081,))(img_tensor)
    # Add batch dimension (the model expects [batch_size, channels, height, width])
    img_tensor = img_tensor.unsqueeze(0) # Shape: [1,1,28,28]

    # predict with the model
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_digit = predicted.item()

    #Display the processed image and prediction
    plt. imshow(img, cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.axis('off')
    plt.show()

    return predicted_digit

# Step 7: Test your image
image_path = "C:\Python\Project2_NN\my_pictures\my_digit_1.jpg"
predicted_digit = predict_custom_image(image_path, model)
print(f'The model predicts your image is: {predicted_digit}')


# Step 6: Evaluate the model
model.eval() # set the model at evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # get index of the max score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * (correct / total)
print(f'Test accuracy: {accuracy:.2f}%')

# Step 7 Plot training loss
simple_scatter(
    range(len(train_losses)), train_losses,
    my_x_label= 'Epoch',
    my_y_label= 'Loss',
    my_title= 'Training Loss Over Time',
    close=True
)

# Step 8: Test on a few examples
def test_samples(loader):
    model.eval() # set the model to evaluation mode
    images, labels = next(iter(loader))
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        fig, axes = plt.subplots(1, 5, figsize=(10,2))
        for i in range(5):
            axes[i].imshow(images[i][0], cmap='gray')
            axes[i].set_title(f'Prediction {predicted[i].item()} \nTrue: {labels[i].item()}')
            axes[i].axis('off')
        plt.show()

test_samples(test_loader)






