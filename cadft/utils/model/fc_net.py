import torch
import primefac


class Transformer(nn.Module):
    """
    Fully connected neural network with two hidden layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Transformer, self).__init__()
        head_list = list(primefac.primefac(input_size))
        nhead = head_list[0]
        if nhead < 4:
            nhead = head_list[0] * head_list[1]

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_size, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=6
        )
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Standard forward function
        """
        x = torch.unsqueeze(x, 1)
        out = self.transformer_encoder(x)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.squeeze(out, 1)
        return out


# class FCNet(nn.Module):
#     """
#     Fully connected neural network with two hidden layer
#     """

#     def __init__(self, input_size, hidden_size, output_size):
#         super(FCNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.laynorm1 = nn.LayerNorm(hidden_size)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.laynorm2 = nn.LayerNorm(hidden_size)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(hidden_size, hidden_size)
#         self.laynorm3 = nn.LayerNorm(hidden_size)
#         self.relu3 = nn.ReLU()
#         self.fc4 = nn.Linear(hidden_size, hidden_size)
#         self.relu4 = nn.ReLU()
#         self.fc5 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         """
#         Standard forward function
#         """
#         out = self.fc1(x)
#         out = self.laynorm1(out)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.laynorm2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.laynorm3(out)
#         out = self.relu3(out)
#         out = self.fc4(out)
#         out = self.relu4(out)
#         out = self.fc5(out)
#         return out
