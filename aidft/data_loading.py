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
        weight_dir: Path,
        mask_suffix: str = "",
    ):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        self.weight_dir = weight_dir

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

        img_file = list(self.images_dir.glob(name + ".*"))
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        weight_file = list(self.weight_dir.glob(name + self.mask_suffix + ".*"))

        assert (
            len(img_file) == 1
        ), f"Either no image or multiple images found for the ID {name}: {img_file}"
        assert (
            len(mask_file) == 1
        ), f"Either no mask or multiple masks found for the ID {name}: {mask_file}"
        assert (
            len(weight_file) == 1
        ), f"Either no weight or multiple weights found for the ID {name}: {weight_file}"
        img = load_numpy(img_file[0])
        mask = load_numpy(mask_file[0])
        weight = load_numpy(weight_file[0])

        # check the size of the img, mask and weight
        assert (
            img.shape == mask.shape
        ), f"Image and mask {name} should be the same size but are {img.shape} and {mask.shape}"
        assert (
            img.shape == weight.shape
        ), f"Image and weight {name} should be the same size but are {img.shape} and {weight.shape}"

        # # filling the data to 32 base
        # right_size = [
        #     img.shape[0],
        #     (img.shape[1] // 32 + 1) * 32,
        #     (img.shape[2] // 32 + 1) * 32,
        # ]
        # img_fill = np.zeros(right_size)
        # mask_fill = np.zeros(right_size)
        # weight_fill = np.zeros(right_size)

        # img_fill[:, : img.shape[1], : img.shape[2]] = img
        # mask_fill[:, : mask.shape[1], : mask.shape[2]] = mask
        # weight_fill[:, : weight.shape[1], : weight.shape[2]] = weight

        # for i in range(img.shape[1], right_size[1]):
        #     img_fill[:, i, :] = img_fill[:, img.shape[1] - 1, :]
        #     mask_fill[:, i, :] = mask_fill[:, mask.shape[1] - 1, :]
        #     weight_fill[:, i, :] = weight_fill[:, weight.shape[1] - 1, :]

        # for i in range(img.shape[2], right_size[2]):
        #     img_fill[:, :, i] = img_fill[:, :, img.shape[2] - 1]
        #     mask_fill[:, :, i] = mask_fill[:, :, mask.shape[2] - 1]
        #     weight_fill[:, :, i] = weight_fill[:, :, weight.shape[2] - 1]

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).float().contiguous(),
            "weight": torch.as_tensor(weight.copy()).float().contiguous(),
        }
