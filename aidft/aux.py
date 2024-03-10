import numpy as np
from torch import nn
import torch


def numpy2str(data: np.ndarray) -> str:
    """
    Documentation for a function.

    More details.
    """
    return np.array2string(
        data.numpy(), precision=4, separator=",", suppress_small=True
    )


class Criterion:
    """
    Documentation for a function.

    More details.
    """

    def __init__(
        self,
        factor: float = 1.0,
        loss1=nn.MSELoss(),
        loss2=nn.MSELoss(),
    ):
        self.factor = factor
        self.loss1 = loss1
        self.loss2 = loss2

    def change_factor(self, factor: float):
        """
        Change the factor.
        """
        self.factor = factor

    def val(self, mask_pred, mask_true, weight):
        """
        Validate loss.
        """

        if mask_pred.shape[1] == 1:
            return (
                self.loss1(mask_pred, mask_true)
                + self.loss2(mask_pred * weight, mask_true * weight) * self.factor
            )

        if mask_pred.shape[1] == 2:
            return (
                self.loss1(mask_pred, mask_true)
                + self.loss2(mask_pred * weight, mask_true * weight) * self.factor
            )


def process(data, device):
    """
    Load the whole data to the device.
    """
    return data.to(
        device=device,
        dtype=torch.float64,
        memory_format=torch.channels_last,
    )


def load_to_gpu(dataloader, device):
    """
    Load the whole data to the device.
    """

    dataloader_gpu = []
    for batch in dataloader:
        batch_gpu = {}
        # move images and labels to correct device and type
        batch_gpu["image"], batch_gpu["mask"], batch_gpu["weight"] = (
            process(batch["image"], device),
            process(batch["mask"], device),
            process(batch["weight"], device),
        )
        dataloader_gpu.append(batch_gpu)
    return dataloader_gpu
