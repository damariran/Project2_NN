import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Setup parameters
model_factor = 2
input_layer = 28*28
first_layer = model_factor*256
second_layer = model_factor*128
third_layer = model_factor*64
output_layer = 10
learning_rate = 0.0005
epochs = 10

# Step 1: Load and preprocess MNIST data set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 3: Define a neural network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_layer, first_layer),
            nn.ReLU(),
            nn.Linear(first_layer, second_layer),
            nn.ReLU(),
            nn.Linear(second_layer, third_layer),
            nn.ReLU(),
            nn.Linear(third_layer, output_layer),
        )
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# Instantiate the model
model = DigitClassifier()

# Step 3: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Step 4: Train the model
for epoch in range(epochs):
    model.train() # set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

# Step 5: Evaluate on test set
model.eval() # set the model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Test {accuracy: 2f}%')

# Step 6: Save the trained model
torch.save(model.state_dict(), 'mnist_digit_model.pth')
print("Model saved as 'mnist_digit_model.pth'")