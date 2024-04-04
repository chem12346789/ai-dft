import argparse
import logging
from pathlib import Path

import torch

from aidft import train_model
from aidft import parser_model
from aidft import gen_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks"
    )
    parser_model(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import segmentation_models_pytorch as smp

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        in_channels=1,
        classes=2,
    )

    # model = gen_model(args)

    model.double()
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        dir_model = Path(args.name) / "checkpoints"
        list_of_path = dir_model.glob("*.pth")
        load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
        state_dict = torch.load(load_path, map_location=device)
        model.load_state_dict(state_dict)
        logging.info("Model loaded from %s", load_path)

    logging.info("Network: %s", f"Using device {device}.")
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
