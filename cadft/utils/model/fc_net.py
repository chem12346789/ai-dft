import torch.nn as nn


class FCNet(nn.Module):
    """
    Fully connected neural network with two hidden layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.laynorm1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.laynorm2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.laynorm3 = nn.LayerNorm(hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu4 = nn.ReLU()
        self.laynorm4 = nn.LayerNorm(hidden_size)
        # self.fc5 = nn.Linear(hidden_size, hidden_size)
        # self.relu5 = nn.ReLU()
        # self.laynorm5 = nn.LayerNorm(hidden_size)
        # self.fc6 = nn.Linear(hidden_size, hidden_size)
        # self.relu6 = nn.ReLU()
        # self.laynorm6 = nn.LayerNorm(hidden_size)
        self.fc7 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Standard forward function
        """
        out = self.fc1(x)
        out = self.laynorm1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.laynorm2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.laynorm3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        # out = self.laynorm4(out)
        out = self.relu4(out)
        # out = self.fc5(out)
        # out = self.laynorm5(out)
        # out = self.relu5(out)
        # out = self.fc6(out)
        # out = self.relu6(out)
        out = self.fc7(out)
        return out
