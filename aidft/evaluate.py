"""@package docstring
Documentation for this module.
 
More details.
"""

import torch
import wandb
import numpy as np
from .aux import numpy2str


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
    criterion,
    logging,
    experiment,
):
    """Documentation for a function.

    More details.
    """
    net.eval()
    sum_error = 0
    iter_ = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in dataloader:
            image, mask_true = (
                batch["image"],
                batch["mask"],
            )

            # predict the mask
            mask_pred = net(image)
            sum_error += criterion.val(mask_pred, mask_true)
            iter_ += 1

            for i in range(image.shape[0]):
                logging.debug("image %s", numpy2str(image[i]))
                logging.debug("mask_true %s", numpy2str(mask_true[i]))
                logging.debug("mask_pred %s", numpy2str(mask_pred[i]))

    experiment.log(
        {
            "masks": {
                "true_0": wandb.Image(mask_true[0].float().cpu()),
                "pred_0": wandb.Image(mask_pred[0].float().cpu()),
            },
        }
    )

    net.train()
    return sum_error / iter_
