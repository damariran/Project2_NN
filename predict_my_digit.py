import torch
import torch.nn as nn
import numpy as np
from  torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# setup parameters
model_factor = 2
input_layer = 28*28
first_layer = model_factor*256
second_layer = model_factor*128
third_layer = model_factor*64
output_layer = 10
image_path = 'C:\Python\Project2_NN\my_pictures\my_digit_9.PNG'

# Step 1: Define the model class (must match the trained model)
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
            nn.Linear(third_layer, output_layer)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x

# Step 2: Load the saved model
model = DigitClassifier()
model.load_state_dict(torch.load('mnist_digit_model.pth'))
model.eval() # Set to evaluation mode
print('Model Loaded successfully')

# Step 3: Function to preprocess and predict your image
def predict_custom_image(image_path, model):
    # Load and preprocess the image
    img = Image.open(image_path).convert('L') # Grayscale
    n = int(np.sqrt(input_layer))
    img = img.resize((n,n), Image.Resampling.LANCZOS) # Resize to 28x28
    # convert to tensor
    img_tensor = transforms.ToTensor()(img)
    # Normalize to match MNIST training data
    img_tensor = transforms.Normalize((0.1307,),(0.3081,))(img_tensor)
    # Add batch dimention
    img_tensor = img_tensor.unsqueeze(0) # Shape: [1,1,28,28]

    # predict
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted  = torch.max(output, 1)
        predicted_digit = predicted.item()

    # Display the processed image and prediction
    plt.imshow(img_tensor[0][0], cmap='gray')
    plt.title(f'Predicted Digit: {predicted_digit}')
    plt.axis('off')
    plt.show()

    return predicted_digit

# Step 4: Test your image
predicted_digit = predict_custom_image(image_path, model)
print(f'The model predicts your digit is: {predicted_digit}')
