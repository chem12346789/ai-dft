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
        self.in_channels = 1
        self.classes = 1

        self.inc = DoubleConv(self.in_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.up3 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up1 = Up(32, 16)
        self.outc = OutConv(16, self.classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits
