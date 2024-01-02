import argparse
import logging
import torch

from src.aidft.get_args import get_args_train, get_args_model
from src.aidft.train_model import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    get_args_train(parser)
    get_args_model(parser)
    args = parser.parse_args()

    if args.model == "unet_small":
        from src.aidft.unet.unet_model_small import UNet
    elif args.model == "unet":
        from src.aidft.unet.unet_model import UNet
    else:
        raise ValueError("Unknown model")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Change here to adapt to your data
    # n_channels=1 for rho only
    # n_classes is the output channels of the network
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model.double()
    model = model.to(memory_format=torch.channels_last)

    logging.info(
        "Network: %s",
        f"""
        {model.n_channels} input channels,
        {model.n_classes} output channels,
        {"Bilinear" if model.bilinear else "Transposed conv"} upscaling,
        Using device {device}.
        """,
    )

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info("Model loaded from %s", args.load)

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            name=args.name,
            val=args.val,
            train=args.train,
            amp=args.amp,
        )

    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! %s",
            """
            Enabling checkpointing to reduce memory usage, but this slows down training. 
            Consider enabling AMP (--amp) for fast and memory efficient training"
            """,
        )
