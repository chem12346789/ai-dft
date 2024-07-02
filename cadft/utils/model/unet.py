import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """Documentation for a class.

    TODO
    """

    def __init__(self, input_size, hidden_size, output_size, residual, num_layers):
        super().__init__()
        self.in_channels = input_size
        self.classes = output_size
        self.residual = residual
        
        if self.residual <= 4:
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
        else:
            decoder_channels = []
            for i in range(num_layers):
                decoder_channels.append(hidden_size * 2 ** (num_layers - i))

            if self.residual == 5:
                self.model = smp.UnetPlusPlus(
                encoder_name="timm-mobilenetv3_small_minimal_100",
                encoder_depth=num_layers,
                decoder_channels=decoder_channels,
                in_channels=1,
                classes=1,
                # decoder_use_batchnorm=False,
                )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        if self.residual <= 4:
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
            x = F.pad(x, (9, 9, 10, 11), "reflect")
            x = self.model(x)
            x = x[:, :, 10:-11, 9:-9]
            return x
