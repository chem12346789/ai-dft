"""@package docstring
Documentation for this module.
 
More details.
"""

import torch
import wandb
import numpy as np


def numpy2str(data: np.ndarray) -> str:
    """
    Documentation for a function.

    More details.
    """
    return np.array2string(
        data.detach().cpu().numpy(),
        precision=4,
        separator=",",
        suppress_small=True,
    )


@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    amp,
    logging,
    criterion,
    experiment,
):
    """Documentation for a function.

    More details.
    """
    net.eval()
    sum_error = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in dataloader:
            image, mask_true, weight = (
                batch["image"],
                batch["mask"],
                batch["weight"],
            )

            # move images and labels to correct device and type
            image = image.to(
                device=device,
                dtype=torch.float64,
                memory_format=torch.channels_last,
            )
            mask_true = mask_true.to(
                device=device,
                dtype=torch.float64,
                memory_format=torch.channels_last,
            )
            weight = weight.to(
                device=device,
                dtype=torch.float64,
                memory_format=torch.channels_last,
            )

            # predict the mask
            mask_pred = net(image)
            sum_error += criterion(mask_pred, mask_true)

            logging.info("image %s", numpy2str(image))
            logging.info("mask_true %s", numpy2str(mask_true))
            logging.info("mask_pred %s", numpy2str(mask_pred))

    experiment.log(
        {
            "masks": {
                "true_0": wandb.Image(mask_true[0].float().cpu()),
                "pred_0": wandb.Image(mask_pred[0].float().cpu()),
            },
        }
    )

    net.train()
    return sum_error
