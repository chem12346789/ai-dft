import argparse
import logging
import torch
from pathlib import Path
from aidft import train_model
from aidft import parser_model
from aidft import UNet

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
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
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
