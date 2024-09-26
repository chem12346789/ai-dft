import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadft.utils.model.unet_parts import DoubleConv, Down, Up, OutConv


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

    def __init__(self, input_size, hidden_size, output_size, residual, num_layers):
        super().__init__()
        self.in_channels = input_size
        self.classes = output_size
        self.residual = residual

        if self.residual < 81:
            if residual == -1:
                norm_layer = "NoNorm2d"
                affine = True
            if residual == 0:
                norm_layer = "BatchNorm2d"
                affine = True
            if residual == 1:
                norm_layer = "BatchNorm2d"
                affine = False
            if residual == 2:
                norm_layer = "InstanceNorm2d"
                affine = True
            if residual == 3:
                norm_layer = "InstanceNorm2d"
                affine = False
            if self.residual == 4:
                norm_layer = "GroupNorm1"
                affine = True
            if self.residual == 5:
                norm_layer = "GroupNorm1"
                affine = False
            if self.residual == 6:
                norm_layer = "GroupNorm2"
                affine = True
            if self.residual == 7:
                norm_layer = "GroupNorm2"
                affine = False
            if self.residual == 8:
                norm_layer = "GroupNorm4"
                affine = True
            if self.residual == 9:
                norm_layer = "GroupNorm4"
                affine = False
            if self.residual == 10:
                norm_layer = "GroupNorm8"
                affine = True
            if self.residual == 11:
                norm_layer = "GroupNorm8"
                affine = False
            if self.residual == 12:
                norm_layer = "GroupNorm16"
                affine = True
            if self.residual == 13:
                norm_layer = "GroupNorm16"
                affine = False

            if "GroupNorm" in norm_layer:
                self.inc = DoubleConv(
                    self.in_channels,
                    hidden_size,
                    norm_layer="NoNorm2d",
                    affine=True,
                )
            else:
                self.inc = DoubleConv(
                    self.in_channels,
                    hidden_size,
                    norm_layer=norm_layer,
                    affine=affine,
                )

            self.down_layers = nn.ModuleList(
                [
                    Down(
                        hidden_size * 2 ** (i),
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

            if self.residual == 81:
                self.model = smp.UnetPlusPlus(
                    encoder_name="resnet18",
                    encoder_depth=num_layers,
                    decoder_channels=decoder_channels,
                    in_channels=self.in_channels,
                    classes=self.classes,
                    encoder_weights=None,
                )
                self.model = bn_no_track(self.model)
            if self.residual == 82:
                self.model = smp.UnetPlusPlus(
                    encoder_name="timm-mobilenetv3_small_100",
                    encoder_depth=num_layers,
                    decoder_channels=decoder_channels,
                    in_channels=self.in_channels,
                    classes=self.classes,
                    encoder_weights=None,
                )
                self.model = bn_no_track(self.model)

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
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
        else:
            x = F.pad(x, (9, 9, 10, 11), "reflect")
            x = self.model(x)
            x = x[:, :, 10:-11, 9:-9]
            return x
