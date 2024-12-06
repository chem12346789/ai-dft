import torch
import torch.nn as nn
import torch.nn.functional as F

from cadft.utils.model.unet_parts import DoubleConv, Down, Up, OutConv
from cadft.utils.model.transformer import PredictorSmall


class UNet(nn.Module):
    """
    TODO
    Documentation for a class.
    """

    def __init__(
        self,
        input_channels,
        hidden_channels,
        output_channels,
        residual,
        num_layers,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.residual = residual
        self.num_layers = num_layers

        print(
            f"Model: UNet, residual: {self.residual}"
            f"num_layers: {self.num_layers}"
            f"hidden_channels: {self.hidden_channels}"
            f"input_channels: {self.input_channels}"
            f"output_channels: {self.output_channels}"
        )

        if self.residual < 10:
            if self.residual == 0:
                norm_layer = "BatchNorm2d"
                affine = True
            elif self.residual == 1:
                norm_layer = "BatchNorm2d"
                affine = False
            else:
                norm_layer = "NoNorm2d"
                affine = True

            print(f"norm_layer: {norm_layer}" f"affine: {affine}")

            self.inc = DoubleConv(
                self.input_channels,
                self.hidden_channels,
                norm_layer=norm_layer,
                affine=affine,
            )

            self.down_layers = nn.ModuleList(
                [
                    Down(
                        self.hidden_channels * 2 ** (i),
                        self.hidden_channels * 2 ** (i + 1),
                        norm_layer=norm_layer,
                        affine=affine,
                    )
                    for i in range(self.num_layers)
                ]
            )
            self.up_layers = nn.ModuleList(
                [
                    Up(
                        self.hidden_channels * 2 ** (i + 1),
                        self.hidden_channels * 2**i,
                        norm_layer=norm_layer,
                        affine=affine,
                    )
                    for i in range(self.num_layers)[::-1]
                ]
            )
            self.outc = OutConv(self.hidden_channels, self.output_channels)
        else:
            if self.residual == 10:
                self.model = PredictorSmall()

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        if self.residual < 10:
            x = self.inc(x)
            x_down = []
            for down in self.down_layers:
                x_down.append(x)
                x = down(x)
            for i, up in enumerate(self.up_layers):
                x = up(x, x_down[-i - 1])
            logits = self.outc(x)
            return logits
        else:
            return self.model(x)
