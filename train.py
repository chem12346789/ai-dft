import argparse
import logging
import torch
from pathlib import Path
from aidft import train_model
from aidft import parser_model

import segmentation_models_pytorch as smp
from aidft import UNet, Transformer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser_model(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Change here to adapt to your data
    # n_channels=1 for rho only
    # n_classes is the output channels of the network
    if "unet" in args.name:
        if "unetplusplus" in args.name:
            if "efficient" in args.name:
                model = smp.UnetPlusPlus(
                    encoder_name="timm-efficientnet-b0",
                    in_channels=1,
                    classes=2,
                )
            else:
                model = smp.UnetPlusPlus(
                    encoder_name="resnet34",
                    in_channels=1,
                    classes=2,
                )
        else:
            model = UNet(in_channels=1, classes=args.classes, bilinear=args.bilinear)
            args.if_pad = False
    elif "transform" in args.name:
        model = Transformer()
        args.if_pad = False
        args.if_flatten = True
    elif "manet" in args.name:
        model = smp.MAnet(
            encoder_name="resnet34",
            in_channels=1,
            classes=2,
        )
    elif "linknet" in args.name:
        model = smp.Linknet(
            encoder_name="resnet34",
            in_channels=1,
            classes=2,
        )
    else:
        logging.info(
            "Network is not supported. Please choose one of the following: unet, unetplusplus, transform, manet",
        )

    model.double()
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        "Network: %s",
        f"""
        Using device {device}.
        """,
    )

    if args.load:
        dir_model = Path(args.name) / "checkpoints"
        list_of_path = dir_model.glob("*.pth")
        load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info("Model loaded from %s", load_path)

    model.to(device=device)

    try:
        train_model(
            model=model,
            device=device,
            args=args,
            save_checkpoint=True,
        )

    except torch.cuda.OutOfMemoryError:
        logging.error("Detected OutOfMemoryError!")
