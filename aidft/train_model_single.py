"""
Module providing a single loop training method.
Used for fast training of a small dataset (batch == size of dataset).
"""
import logging
import os
from pathlib import Path
from tqdm import tqdm
import wandb

import time

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from .evaluate import evaluate
from .data_loading import BasicDataset


def train_model(
    model,
    device,
    epochs: int = 5000,
    batch_size: int = 20,
    learning_rate: float = 1e-4,
    name: str = "./firstrun",
    val: int = 20,
    train: int = 200,
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-3,
    momentum: float = 0.9,
    gradient_clipping: float = 1.0,
):
    """Documentation for a function.

    More details.
    """