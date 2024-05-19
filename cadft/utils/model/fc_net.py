import torch.nn as nn


class FCNet(nn.Module):
    """
    Fully connected neural network with two hidden layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.laynorm1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.laynorm2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.laynorm3 = nn.LayerNorm(hidden_size)
        self.relu3 = nn.ReLU()

        # self.fc4 = nn.Linear(hidden_size, hidden_size)
        # self.laynorm4 = nn.LayerNorm(hidden_size)
        # self.relu4 = nn.ReLU()

        # self.fc5 = nn.Linear(hidden_size, hidden_size)
        # self.laynorm5 = nn.LayerNorm(hidden_size)
        # self.relu5 = nn.ReLU()

        self.fcout1 = nn.Linear(hidden_size, hidden_size)
        self.reluout = nn.ReLU()
        self.fcout2 = nn.Linear(hidden_size, output_size)

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
        out = self.laynorm4(out)
        out = self.relu4(out)

        out = self.fc5(out)
        out = self.laynorm5(out)
        out = self.relu5(out)

        out = self.fcout1(out)
        out = self.reluout(out)
        out = self.fcout2(out)
        return out
