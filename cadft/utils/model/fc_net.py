import torch
import torch.nn as nn


class FCNet(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, input_size, hidden_size, output_size, args):
        super(FCNet, self).__init__()

        self.hidden_size = hidden_size
        self.args = args
        sizes = [input_size] + [hidden_size] * args.num_layers + [output_size]

        self.layers = nn.ModuleList(
            [
                nn.Linear(input_size, output_size)
                for input_size, output_size in zip(sizes, sizes[1:])
            ]
        )
        self.actv_fn = nn.ReLU()

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        if self.args.residual == 2:
            res_tmp = torch.zeros(self.hidden_size, device=x.device)
            num_res = 0
        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if i < len(self.layers) - 1:
                tmp = self.actv_fn(tmp)
            if layer.in_features == layer.out_features:
                if self.args.residual == 2:
                    num_res = num_res + 1
                    res_tmp = res_tmp + tmp
                    x = x + res_tmp / num_res
                if self.args.residual == 1:
                    x = x + tmp
                else:
                    x = tmp
            else:
                x = tmp
        return x
