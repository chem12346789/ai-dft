"""@package docstring
Documentation for this module.
 
More details.
"""

import torch
import wandb
import numpy as np
from .aux import numpy2str


@torch.inference_mode()
def evaluate(
    net,
    dataloader,
    device,
    amp,
    criterion,
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
            image = batch["image"]
            mask_true = batch["mask"]

            # predict the mask
            mask_pred = net(image)
            sum_error += criterion.val(mask_pred, mask_true)
            iter_ += 1

            # for i in range(image.shape[0]):
            #     print("image %s", numpy2str(image[i]))
            #     print("mask_true %s", numpy2str(mask_true[i]))
            #     print("mask_pred %s", numpy2str(mask_pred[i]))

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
