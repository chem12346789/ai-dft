"""Module providing a training method."""
import logging
import os
from pathlib import Path
import time
from tqdm import tqdm
import wandb
import numpy as np


import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from .evaluate import evaluate
from .data_loading import BasicDataset


def numpy2str(data: np.ndarray) -> str:
    """
    Documentation for a function.

    More details.
    """
    return np.array2string(
        data.numpy(), precision=4, separator=",", suppress_small=True
    )


def train_model(
    model,
    device,
    args,
    save_checkpoint: bool = True,
):
    """Documentation for a function.

    More details.
    """

    # 1. Create dataset
    dir_img = Path(args.name) / "data" / "imgs/"
    dir_mask = Path(args.name) / "data" / "masks/"
    dir_weight = Path(args.name) / "data" / "weights/"
    dir_checkpoint = Path(args.name) / "checkpoints/"
    dataset = BasicDataset(dir_img, dir_mask, dir_weight)

    # 2. Split into train / validation partitions note we cut off the last
    # batch if it's not full
    n_remaining = len(dataset) - args.train - args.val
    train_set, val_set, _ = random_split(
        dataset,
        [args.train, args.val, n_remaining],
        generator=torch.Generator().manual_seed(0),
    )

    logging.info("""Split into train / validation partitions.""")

    # 3. Create data loaders
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # print the data
    for batch in train_loader:
        logging.info("image %s", numpy2str(batch["image"]))
        logging.info("mask %s", numpy2str(batch["mask"]))
        logging.info("weight %s", numpy2str(batch["weight"]))
    for batch in val_loader:
        logging.info("image %s", numpy2str(batch["image"]))
        logging.info("mask %s", numpy2str(batch["mask"]))
        logging.info("weight %s", numpy2str(batch["weight"]))

    # (Initialize logging)
    experiment = wandb.init(
        project="UNet",
        resume="allow",
        name=f"HH-Unet-{args.optimizer}-{args.scheduler}",
    )
    experiment.config.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "save_checkpoint": save_checkpoint,
            "amp": args.amp,
        }
    )

    logging.info(
        "Starting training: %s",
        f"""
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
        Training size:   {args.train}
        Validation size: {args.val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {args.amp}
        Data_img:        {dir_img}
        Data_mask:       {dir_mask}
        Checkpoints:     {dir_checkpoint}
    """,
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the
    # loss scaling for AMP
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "adamax":
        optimizer = optim.Adamax(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "radam":
        optimizer = optim.RAdam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "nadam":
        optimizer = optim.NAdam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "adadelta":
        optimizer = optim.Adadelta(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            foreach=True,
        )
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            foreach=True,
        )
    elif args.optimizer == "lbfgs":
        optimizer = optim.LBFGS(
            model.parameters(),
            lr=args.learning_rate,
        )
    else:
        raise ValueError("Unknown optimizer")

    # goal: minimize the error
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    elif args.scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise ValueError("Unknown scheduler")

    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.MSELoss()
    division_epoch = 50
    save_epoch = 2500
    val_score = None

    # 5. Begin training
    with tqdm(total=args.epochs, unit="epoch") as pbar:
        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            model.train()
            end_time = time.time()
            pbar.update()

            for batch in train_loader:
                images, true_masks, weight = (
                    batch["image"],
                    batch["mask"],
                    batch["weight"],
                )

                assert images.shape[1] == model.n_channels, (
                    f"Network has been defined with {model.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that the images are loaded correctly."
                )

                images = images.to(
                    device=device,
                    dtype=torch.float64,
                    memory_format=torch.channels_last,
                )

                true_masks = true_masks.to(
                    device=device,
                    dtype=torch.float64,
                    memory_format=torch.channels_last,
                )

                weight = weight.to(
                    device=device,
                    dtype=torch.float64,
                    memory_format=torch.channels_last,
                )

                with torch.autocast(device.type, enabled=args.amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clipping
                )
                grad_scaler.step(optimizer)
                grad_scaler.update()

                experiment.log(
                    {
                        "train loss": loss.item(),
                        "epoch": epoch,
                        " time": end_time - start_time,
                    }
                )
                if epoch > division_epoch:
                    pbar.set_postfix(
                        **{"loss (batch)": loss.item(), "error": val_score.item()}
                    )
                else:
                    pbar.set_postfix(**{"loss (batch)": loss.item(), "error": "N/A"})

            if epoch % division_epoch == 0:
                val_score = evaluate(
                    model, val_loader, device, args.amp, logging, criterion, experiment
                )
                scheduler.step(val_score)
                pbar.set_postfix(
                    **{"loss (batch)": loss.item(), "error": val_score.item()}
                )
                experiment.log(
                    {
                        "val loss": loss.item(),
                        "learning rate": optimizer.param_groups[0]["lr"],
                    }
                )

            if epoch % save_epoch == 0:
                if save_checkpoint:
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict_ = model.state_dict()
                    torch.save(
                        state_dict_,
                        str(
                            dir_checkpoint
                            / f"checkpoint_epoch-{args.optimizer}-{args.scheduler}-{epoch}.pth"
                        ),
                    )
