import torch.nn as nn


class FCNet(nn.Module):
    """
    Fully connected neural network with two hidden layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.norm5 = nn.LayerNorm(hidden_size)
        self.relu5 = nn.ReLU()

        # self.fc6 = nn.Linear(hidden_size, hidden_size)
        # self.norm6 = nn.LayerNorm(hidden_size)
        # self.relu6 = nn.ReLU()

        # self.fc7 = nn.Linear(hidden_size, hidden_size)
        # self.norm7 = nn.LayerNorm(hidden_size)
        # self.relu7 = nn.ReLU()

        self.fcout1 = nn.Linear(hidden_size, hidden_size)
        self.reluout = nn.ReLU()
        self.fcout2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Standard forward function
        """
        x = self.fc1(x)
        x = self.norm1(x)
        x1 = self.relu1(x)

        x2 = self.fc2(x1)
        x2 = self.norm2(x2)
        x2 = self.relu2(x2)

        x3 = self.fc3(x2)
        x3 = self.norm3(x3)
        x3 = self.relu3(x3)

        x4 = self.fc4(x3)
        x4 = self.norm4(x4)
        x4 = self.relu4(x4)

        x5 = self.fc5(x4)
        x5 = self.norm5(x5)
        x5 = self.relu5(x5)

        # x6 = self.fc6(x5)
        # x6 = self.norm6(x6)
        # x6 = self.relu6(x6)

        # x7 = self.fc7(x6)
        # x7 = self.norm7(x7)
        # x7 = self.relu7(x7)

        out4 = self.fcout1(x5)
        out4 = self.reluout(out4)
        out4 = self.fcout2(out4)
        return out4
