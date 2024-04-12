import logging

import numpy as np
from torch import nn
import torch

import segmentation_models_pytorch as smp
from .unet.unet_model import UNet
from .transformer.models import Extractor as Transformer


def numpy2str(data: np.ndarray) -> str:
    """
    Documentation for a function.

    More details.
    """
    if isinstance(data, np.ndarray):
        return np.array2string(data, precision=4, separator=",", suppress_small=True)
    if isinstance(data, torch.Tensor):
        return np.array2string(
            data.cpu().numpy(), precision=4, separator=",", suppress_small=True
        )


class Criterion:
    """
    Documentation for a function.

    More details.
    """

    def __init__(
        self,
        factor: float = 0.01,
        loss1=nn.MSELoss(),
    ):
        self.factor = factor
        self.loss1 = loss1

    def change_factor(self, factor: float):
        """
        Change the factor.
        """
        self.factor = factor

    def val(self, mask_pred, mask_true):
        """
        Validate loss.
        """
        return self.loss1(mask_pred, mask_true)


def process(data, device):
    """
    Load the whole data to the device.
    """
    if len(data.shape) == 4:
        return data.to(
            device=device,
            dtype=torch.float64,
            memory_format=torch.channels_last,
        )
    else:
        return data.to(
            device=device,
            dtype=torch.float64,
        )


def load_to_gpu(dataloader, device):
    """
    Load the whole data to the device.
    """

    dataloader_gpu = []
    for batch in dataloader:
        batch_gpu = {}
        # move images and labels to correct device and type
        batch_gpu["image"], batch_gpu["mask"] = (
            process(batch["image"], device),
            process(batch["mask"], device),
        )
        dataloader_gpu.append(batch_gpu)
    return dataloader_gpu


def gen_model(args):
    """
    Change here to adapt to your data
    n_channels=1 for rho only
    n_classes is the output channels of the network
    """
    if "unet" in args.name:
        if "unetplusplus" in args.name:
            if "efficient" in args.name:
                model = smp.UnetPlusPlus(
                    encoder_name="timm-efficientnet-b2",
                    in_channels=1,
                    classes=2,
                )
                print("Using unetplusplus and efficientnet")
            else:
                if "32" in args.name:
                    model = smp.UnetPlusPlus(
                        encoder_name="resnet34",
                        decoder_channels=(512, 256, 128, 64, 32),
                        in_channels=1,
                        classes=2,
                    )
                    print("Using unetplusplus with 32 decoder_channels and resnet34")
                elif "64" in args.name:
                    model = smp.UnetPlusPlus(
                        encoder_name="resnet34",
                        decoder_channels=(1024, 512, 256, 128, 64),
                        in_channels=1,
                        classes=2,
                    )
                    print("Using unetplusplus with 64 decoder_channels and resnet34")
                elif "128" in args.name:
                    model = smp.UnetPlusPlus(
                        encoder_name="resnet34",
                        decoder_channels=(2048, 1024, 512, 256, 128),
                        in_channels=1,
                        classes=2,
                    )
                    print("Using unetplusplus with 128 decoder_channels and resnet34")
                else:
                    model = smp.UnetPlusPlus(
                        encoder_name="resnet34",
                        in_channels=1,
                        classes=2,
                    )
                    print("Using unetplusplus with resnet34")
        else:
            model = UNet(in_channels=1, classes=args.classes, bilinear=args.bilinear)
            args.if_pad = False
            print("Using unet")
    elif "transform" in args.name:
        model = Transformer()
        args.if_pad = False
        args.if_flatten = True
        print("Using transformer")
    elif "manet" in args.name:
        if "32" in args.name:
            model = smp.MAnet(
                encoder_name="resnet34",
                decoder_channels=(512, 256, 128, 64, 32),
                in_channels=1,
                classes=2,
            )
            print("Using manet with 32 decoder_channels and resnet34")
        elif "64" in args.name:
            model = smp.MAnet(
                encoder_name="resnet34",
                decoder_channels=(1024, 512, 256, 128, 64),
                in_channels=1,
                classes=2,
            )
            print("Using manet with 64 decoder_channels and resnet34")
        else:
            model = smp.MAnet(
                encoder_name="resnet34",
                in_channels=1,
                classes=2,
            )
            print("Using manet with resnet34")
    elif "linknet" in args.name:
        model = smp.Linknet(
            encoder_name="resnet34",
            in_channels=1,
            classes=2,
        )
        print("Using linknet")
    else:
        logging.info(
            "Network is not supported. Please choose one of the following: unet, unetplusplus, transform, manet",
        )
        model = None
    return model
