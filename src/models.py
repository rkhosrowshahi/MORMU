import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_units=[128, 64]):
        """
        Initialize the Flexible MLP model.
        
        Args:
            input_size (int): The number of input features.
            num_classes (int): The number of output classes.
            hidden_units (list): List of hidden layer sizes.
        """
        super(MLP, self).__init__()
        
        # Create a list of layers dynamically based on the hidden_units provided
        layers = []
        in_features = input_size
        
        for hidden_unit in hidden_units:
            layers.append(nn.Linear(in_features, hidden_unit))
            layers.append(nn.ReLU())
            in_features = hidden_unit
        
        # The final layer to match the number of output classes
        layers.append(nn.Linear(in_features, num_classes))
        
        # Register the layers as a sequential container
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Flatten the input tensor if it has more than 2 dimensions
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten the input data (batch_size, -1)
        return self.network(x)


# Define LeNet-5 Model
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Forward pass through the network
        x = F.tanh(self.conv1(x))  # Conv1
        x = self.pool1(x)          # Pool1
        x = F.tanh(self.conv2(x))  # Conv2
        x = self.pool2(x)          # Pool2

        x = x.view(x.size(0), -1)  # Flatten
        x = F.tanh(self.fc1(x))     # FC1
        x = F.tanh(self.fc2(x))     # FC2
        x = self.fc3(x)             # FC3 (Output layer)
        return x
