import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    Fully connected neural network (dense network)
    """

    def __init__(self, input_size, hidden_size, output_size, residual, num_layers):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="timm-mobilenetv3_small_minimal_100",
            encoder_depth=3,
            decoder_channels=(128, 64, 32),
            in_channels=1,
            classes=1,
        )

    def forward(self, x):
        """
        Standard forward function, required for all nn.Module classes
        """
        x = F.pad(x, (9, 9, 0, 0), "reflect")
        x = self.model(x)
        x = x[:, :, :, 9:-9]
        return x
