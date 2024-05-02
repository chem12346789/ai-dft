"""@package docstring
Documentation for this module.
 
More details.
"""

from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def load_numpy(filename):
    """Documentation for a function.

    More details.
    """
    return np.load(filename)


class BasicDataset(Dataset):
    """Documentation for a class."""

    def __init__(
        self,
        images_dir: Path,
        mask_dir: Path,
        if_pad=True,
        if_flatten=True,
        mask_suffix: str = "",
    ):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix

        self.data = {}

        self.ids = [
            splitext(file)[0]
            for file in sorted(listdir(images_dir))
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

        for idx, name in enumerate(self.ids):
            img_file = list(self.images_dir.glob(name + ".*"))
            mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))

            assert (
                len(img_file) == 1
            ), f"Either no image or multiple images found for the ID {name}: {img_file}"
            assert (
                len(mask_file) == 1
            ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
            img = load_numpy(img_file[0])
            mask = load_numpy(mask_file[0])

            img = torch.as_tensor(img.copy()).to(torch.float64).contiguous()
            mask = torch.as_tensor(mask.copy()).to(torch.float64).contiguous()

            if if_pad:
                img = F.pad(img, (9, 9, 10, 11), "reflect")
                mask = F.pad(mask, (9, 9, 10, 11), "reflect")

            if if_flatten:
                img = np.squeeze(img, axis=0)  # feature.shape = (75, 302)
                mask = np.reshape(mask, (150, 302))  # label.shape = (150, 302)

            self.data[idx] = {
                "image": img,
                "mask": mask,
                "name": name,
            }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.data[idx]
