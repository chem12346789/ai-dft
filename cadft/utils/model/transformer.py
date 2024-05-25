import torch
import primefac


class Transformer(torch.nn.Module):
    """
    Fully connected neural network with two hidden layer
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(Transformer, self).__init__()
        # head_list = list(primefac.primefac(input_size))
        # nhead = head_list[0]
        # if nhead < 4:
        #     nhead = head_list[0] * head_list[1]

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=input_size, nhead=1, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=5
        )

    def forward(self, x):
        """
        Standard forward function
        """
        x = torch.unsqueeze(x, 1)
        out = self.transformer_encoder(x)
        out = torch.squeeze(out, 1)
        return out
