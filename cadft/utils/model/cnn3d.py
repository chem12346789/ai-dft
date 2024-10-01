# An 3d cnn model
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN3D(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self):
        super().__init__()
        # input size = torch.Size([1, 1, 40, 194])
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output size = torch.Size([1, 32, 18, 96])
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # output size = torch.Size([1, 64, 8, 48])
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # # output size = torch.Size([1, 128, 3, 24])
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 24, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x
