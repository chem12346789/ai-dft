import argparse
import logging
import torch
from pathlib import Path
from aidft import train_model
from aidft import parser_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser_model(parser)
    args = parser.parse_args()

    if args.model == "unet_small":
        from aidft import UNet_small as UNet
    elif args.model == "unet":
        from aidft import UNet
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
        dir_checkpoint = Path(args.name) / "checkpoints/"
        load_path = dir_checkpoint / f"checkpoint_epoch-{args.load}.pth"
        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info("Model loaded from %s", load_path)

    model.to(device=device)
    try:
        train_model(
            model=model,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            name=args.name,
            val=args.val,
            train=args.train,
            save_checkpoint=True,
            amp=args.amp,
            weight_decay=1e-3,
            momentum=0.99,
            gradient_clipping=1.0,
        )

    except torch.cuda.OutOfMemoryError:
        logging.error(
            "Detected OutOfMemoryError! %s",
            """
            Enabling checkpointing to reduce memory usage, but this slows down training. 
            Consider enabling AMP (--amp) for fast and memory efficient training"
            """,
        )
