"""@package docstring
Documentation for this module.
 
More details.
"""

import torch
import wandb


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion, experiment):
    """Documentation for a function.

    More details.
    """
    net.eval()
    sum_error = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        print("evaluate")
        for batch in dataloader:
            image, mask_true = batch["image"], batch["mask"]

            # move images and labels to correct device and type
            image = image.to(
                device=device, dtype=torch.float64, memory_format=torch.channels_last
            )
            mask_true = mask_true.to(device=device, dtype=torch.float64)

            # predict the mask
            mask_pred = net(image)
            sum_error += criterion(mask_pred.float(), mask_true.float())

    experiment.log(
        {
            "masks": {
                "true_0": wandb.Image(mask_true[0].float().cpu()),
                "pred_0": wandb.Image(mask_pred[0].float().cpu()),
                "true_1": wandb.Image(mask_true[1].float().cpu()),
                "pred_1": wandb.Image(mask_pred[1].float().cpu()),
            },
        }
    )

    net.train()
    return sum_error
