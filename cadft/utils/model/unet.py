import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import DoubleConv, Down, Up, OutConv

# class UNet(nn.Module):
#     """
#     Fully connected neural network (dense network)
#     """

#     def __init__(self, input_size, hidden_size, output_size, residual, num_layers):
#         super().__init__()
#         self.model = smp.UnetPlusPlus(
#             encoder_name="timm-mobilenetv3_small_minimal_100",
#             encoder_depth=3,
#             decoder_channels=(128, 64, 32),
#             in_channels=1,
#             classes=1,
#             # decoder_use_batchnorm=False,
#     )

# def forward(self, x):
#     """
#     Standard forward function, required for all nn.Module classes
#     """
#     x = F.pad(x, (9, 9, 0, 0), "reflect")
#     x = self.model(x)
#     x = x[:, :, :, 9:-9]
#     return x


class UNet(nn.Module):
    """Documentation for a class.

    TODO
    """

    def __init__(self, input_size, hidden_size, output_size, residual, num_layers):
        super().__init__()
        self.in_channels = input_size
        self.classes = output_size
        if residual == 0:
            norm_layer = "BatchNorm2d"
            affine = True
        if residual == 1:
            norm_layer = "InstanceNorm2d"
            affine = True
        if residual == 2:
            norm_layer = "BatchNorm2d"
            affine = False
        if residual == 3:
            norm_layer = "InstanceNorm2d"
            affine = False

        self.inc = DoubleConv(
            self.in_channels,
            hidden_size,
            norm_layer=norm_layer,
            affine=affine,
        )

        self.down_layers = nn.ModuleList(
            [
                Down(
                    hidden_size * 2**i,
                    hidden_size * 2 ** (i + 1),
                    norm_layer=norm_layer,
                    affine=affine,
                )
                for i in range(num_layers)
            ]
        )
        self.up_layers = nn.ModuleList(
            [
                Up(
                    hidden_size * 2 ** (i + 1),
                    hidden_size * 2**i,
                    norm_layer=norm_layer,
                    affine=affine,
                )
                for i in range(num_layers)[::-1]
            ]
        )
        self.outc = OutConv(hidden_size, self.classes)

    def forward(self, x):
        x = self.inc(x)
        x_down = []
        for down in self.down_layers:
            x_down.append(x)
            x = down(x)
        for i, up in enumerate(self.up_layers):
            x = up(x, x_down[-i - 1])
        logits = self.outc(x)
        return logits
