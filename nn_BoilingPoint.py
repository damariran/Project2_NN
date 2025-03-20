import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from simple_functions import simple_scatter, simple_plot

# simulation parameters:
test_group_size = 20 # [%]
layer2_size =[16,8]
epochs = 200 # The number of cycles of training
learning_rate = 0.01  # How fast is the step changes in learning

# Step 1: Create the dataset (same as before)
data = {
    'n_carbons': [1, 2, 3, 4, 5, 6, 7, 8],
    'mol_weight': [16.04, 30.07, 44.10, 58.12, 72.15, 86.18, 100.20, 114.23],
    'boiling_point': [111.7, 184.6, 231.1, 272.7, 309.2, 341.9, 371.6, 398.8]
}
df = pd.DataFrame(data) # Basically takes the data and makes it into a table.

# Step 2: Prepare data
X = df[['n_carbons', 'mol_weight']].values  # These are the features.
y = df['boiling_point'].values  # Target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to pytorch tensors (like matlab matrices, but for pytorch)
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
# .view () explained '-1' his is a placeholder dimension that tells PyTorch to infer the
# appropriate size for this dimension based on the total number
# of elements in the tensor and the other specified dimensions.
# It essentially "automatically calculates" the size needed to maintain
# the total number of elements. "1" This specifies that the second dimension
# of the tensor should have size 1.
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

class BoilingPointNN(nn.Module):
    def __init__(self):
        super(BoilingPointNN, self).__init__()
        self.layer1 = nn.Linear(2, layer2_size[0]) # Input 2 features, output 16 neurons
        self.layer2 = nn.Linear(layer2_size[0], layer2_size[1]) # Hidden layer: 16->8 neurons
        self.layer3 = nn.Linear(layer2_size[1], 1) #Out put layer: 8->1 (Boiling point)
        self.relu = nn.ReLU() # Rectified Linear Unit (ReLU) activation function. introduces non-linearity by transforming inputs as max(0, x).

    def forward(self, x):
        x = self.relu(self.layer1(x)) # first layer  + activation
        x = self.relu(self.layer2(x)) # second layer + activation
        x = self.layer3(x) # output layer (no activation for regression)
        return x

# Instantiate the model
model = BoilingPointNN()

# Step 4: Define loss function and optimizer
criterion = nn.MSELoss() # Mean squared error loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer. lr is the learning rate

# Step 5: Train the model

train_losses = []
for epoch in range(epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Clear previous gradients
    outputs = model(X_train_tensor)  # Forward pass
    loss = criterion(outputs, y_train_tensor)  # Compute loss
    loss.backward()  # Backward pass (compute gradients)
    optimizer.step()  # Update weights
    train_losses.append(loss.item())  # Store loss for plotting
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 6: Evaluate the model
model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient computation
    y_pred_tensor = model(X_test_tensor)
    test_loss = criterion(y_pred_tensor, y_test_tensor)
    print(f'Test Loss (MSE): {test_loss.item():.2f}')

# Convert predictions to numpy for plotting
y_pred = y_pred_tensor.numpy()
y_test = y_test_tensor.numpy()

# Step 7: Plot results
simple_scatter(
    y_test, y_pred,
    my_legend='Predicted vs Actual',
    my_x_label='Actual Boiling Point[K]',
    my_y_label='Predicted Boiling Point[K]',
    my_title='Neural Network Prediction of Alkane Boiling points (Pytorch)',
    sub_plot=(2,1,1),
    close=False
)
# Plot training loss
epoch_time = np.linspace(1, len(train_losses), len(train_losses))
simple_plot(
    epoch_time,train_losses,
    my_legend='Training loss',
    my_x_label='Epoch',
    my_y_label='Loss [MSE]',
    my_title='Training Loss over Time',
    sub_plot=(2,1,2)
)