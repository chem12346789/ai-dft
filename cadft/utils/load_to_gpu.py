import torch


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
        (
            batch_gpu["input"],
            batch_gpu["output"],
        ) = (
            process(batch["input"], device),
            process(batch["output"], device),
        )
        dataloader_gpu.append(batch_gpu)
    return dataloader_gpu
