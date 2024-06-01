"""
Create a basic dataset class for torch DataLoader.
"""

import torch
from torch.utils.data import DataLoader


def process(data):
    """
    Load the whole data to the gpu.
    """
    if len(data.shape) == 4:
        return data.to(
            device="cuda",
            dtype=torch.float64,
            memory_format=torch.channels_last,
        )
    else:
        return data.to(
            device="cuda",
            dtype=torch.float64,
        )


def load_to_gpu(dataloader):
    """
    Load the whole data to the device.
    """

    dataloader_gpu = []
    for batch in dataloader:
        batch_gpu = {}
        # move images and labels to correct device and type
        (
            batch_gpu["input"],
            batch_gpu["middle"],
            batch_gpu["output"],
            batch_gpu["weight"],
        ) = (
            process(batch["input"]),
            process(batch["middle"]),
            process(batch["output"]),
            process(batch["weight"]),
        )
        dataloader_gpu.append(batch_gpu)
    return dataloader_gpu


class BasicDataset:
    """
    Documentation for a class.
    """

    def __init__(self, input_, middle_, output_, weight_, batch_size):
        self.input = input_
        self.middle = middle_
        self.weight = weight_
        self.output = output_
        self.ids = list(input_.keys())
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return {
            "input": self.input[self.ids[idx]],
            "middle": self.middle[self.ids[idx]],
            "output": self.output[self.ids[idx]],
            "weight": self.weight[self.ids[idx]],
        }

    def load_to_gpu(self):
        """
        Load the whole data to the device.
        """
        train_loader = DataLoader(
            self,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
        )
        return load_to_gpu(train_loader)
