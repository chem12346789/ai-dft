import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadft.utils.model.unet_parts import DoubleConv, Down, Up, OutConv
from cadft.utils.model.transformer import PredictorSmall


def bn_no_track(module):
    """
    Set BatchNorm layers to not track running statistics
    """
    module_output = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module_output = nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            track_running_stats=False,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = None
        module_output.running_var = None
        module_output.num_batches_tracked = None
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig

    for name, child in module.named_children():
        module_output.add_module(name, bn_no_track(child))

    del module
    return module_output


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

        if self.residual < 81:
            self.inc = DoubleConv(self.input_channels, self.hidden_channels)

            self.down_layers = nn.ModuleList(
                [
                    Down(
                        self.hidden_channels * 2 ** (i),
                        self.hidden_channels * 2 ** (i + 1),
                    )
                    for i in range(self.num_layers)
                ]
            )
            self.up_layers = nn.ModuleList(
                [
                    Up(
                        self.hidden_channels * 2 ** (i + 1),
                        self.hidden_channels * 2**i,
                    )
                    for i in range(self.num_layers)[::-1]
                ]
            )
            self.outc = OutConv(self.hidden_channels, self.output_channels)
        else:
            decoder_channels = []
            for i in range(self.num_layers):
                decoder_channels.append(
                    self.hidden_channels * 2 ** (self.num_layers - i)
                )

            if self.residual == 81:
                self.model = smp.UnetPlusPlus(
                    encoder_name="resnet18",
                    encoder_depth=self.num_layers,
                    decoder_channels=decoder_channels,
                    in_channels=self.input_channels,
                    classes=self.output_channels,
                    encoder_weights=None,
                )
                self.model = bn_no_track(self.model)
            if self.residual == 82:
                self.model = smp.UnetPlusPlus(
                    encoder_name="timm-mobilenetv3_small_100",
                    encoder_depth=self.num_layers,
                    decoder_channels=decoder_channels,
                    in_channels=self.input_channels,
                    classes=self.output_channels,
                    encoder_weights=None,
                )
                self.model = bn_no_track(self.model)
            if self.residual == 101:
                self.model = PredictorSmall(
                    depth=self.num_layers,
                    hidden_channels=self.hidden_channels,
                    in_channels=self.input_channels,
                    classes=self.output_channels,
                    encoder_weights=None,
                )
            

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x = x ** (1 / 3)
        if self.residual < 81:
            x = self.inc(x)
            x_down = []
            for down in self.down_layers:
                x_down.append(x)
                x = down(x)
            for i, up in enumerate(self.up_layers):
                x = up(x, x_down[-i - 1])
            logits = self.outc(x)
            return logits
        elif self.residual < 101:
            x = F.pad(x, (9, 9, 10, 11), "reflect")
            x = self.model(x)
            x = x[:, :, 10:-11, 9:-9]
            return x
        else:
            x = self.model(x)
