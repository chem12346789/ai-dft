import numpy as np
from torch import nn
import torch


def numpy2str(data: np.ndarray) -> str:
    """
    Documentation for a function.

    More details.
    """
    if isinstance(data, np.ndarray):
        return np.array2string(data, precision=4, separator=",", suppress_small=True)
    if isinstance(data, torch.Tensor):
        return np.array2string(
            data.cpu().numpy(), precision=4, separator=",", suppress_small=True
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

    def val(self, mask_pred, mask_true):
        """
        Validate loss.
        """
        return self.loss1(mask_pred, mask_true)


def process(data, device):
    """
    Load the whole data to the device.
    """
    if len(data.shape) == 4:
        return data.to(
            device=device,
            dtype=torch.float64,
            memory_format=torch.channels_last,
        )
    else:
        return data.to(
            device=device,
            dtype=torch.float64,
        )


def load_to_gpu(dataloader, device):
    """
    Load the whole data to the device.
    """

    dataloader_gpu = []
    for batch in dataloader:
        batch_gpu = {}
        # move images and labels to correct device and type
        batch_gpu["image"], batch_gpu["mask"] = (
            process(batch["image"], device),
            process(batch["mask"], device),
        )
        dataloader_gpu.append(batch_gpu)
    return dataloader_gpu
