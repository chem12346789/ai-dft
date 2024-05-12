import torch
import primefac


class Transformer(torch.nn.Module):
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
            encoder_layer, num_layers=3
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
