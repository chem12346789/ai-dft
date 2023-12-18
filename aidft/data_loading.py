"""@package docstring
Documentation for this module.
 
More details.
"""
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset


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
        mask_suffix: str = "",
    ):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix

        self.ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        if not self.ids:
            raise RuntimeError(
                f"No input file found in {images_dir}, make sure you put your images there"
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]

        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        img = load_numpy(img_file[0])
        mask = load_numpy(mask_file[0])

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).float().contiguous(),
        }
