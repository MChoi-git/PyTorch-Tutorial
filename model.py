import torch
import torch.nn as nn
import torch.nn.functional as F


# Define neural net
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolution 3 input channels to 6 output channels, kernel is 5x5
        self.conv1 = nn.Conv2d(3, 12, 5)
        # Max pooling kernel is 2x2, stride is 2
        self.pool = nn.MaxPool2d(2, 2)
        # Convolution 6 input channels to 16 output channels, kernel is 5x5
        self.conv2 = nn.Conv2d(12, 16, 5)
        # Linear 16 * 5 * 5 in features, 120 out features
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Linear 120 in features, 85 out features
        self.fc2 = nn.Linear(120, 85)
        # Linear 85 in features, 10 out features
        self.fc3 = nn.Linear(85, 10)

    # Forward computation definition
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)   # Flatten all but batch dim (batch dim is at 0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x