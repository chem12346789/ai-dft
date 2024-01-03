"""Module providing a training method."""
import logging
import os
from pathlib import Path
from tqdm import tqdm
import wandb

import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.aidft.evaluate import evaluate
from src.aidft.data_loading import BasicDataset


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
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    """Documentation for a function.

    More details.
    """

    # 1. Create dataset
    dir_img = Path(name) / "data" / "imgs/"
    dir_mask = Path(name) / "data" / "masks/"
    dir_checkpoint = Path(name) / "checkpoints/"
    dataset = BasicDataset(dir_img, dir_mask)

    # 2. Split into train / validation partitions note we cut off the last
    # batch if it's not full
    n_val = val
    n_train = train
    n_remaining = len(dataset) - n_train - n_val
    train_set, val_set, _ = random_split(
        dataset,
        [n_train, n_val, n_remaining],
        generator=torch.Generator().manual_seed(0),
    )

    logging.info("""Split into train / validation partitions.""")

    # 3. Create data loaders
    loader_args = dict(
        batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project="UNet", resume="allow", name=name)
    experiment.config.update(
        dict(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            save_checkpoint=save_checkpoint,
            amp=amp,
        )
    )

    logging.info(
        "Starting training: %s",
        f"""
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
        Data_img:        {dir_img}
        Data_mask:       {dir_mask}
        Checkpoints:     {dir_checkpoint}
    """,
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the
    # loss scaling for AMP
    optimizer = optim.RMSprop(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        foreach=True,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5
    )  # goal: minimize the error

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    division_epoch = 100
    val_score = None

    # 5. Begin training

    with tqdm(total=epochs, unit="epoch") as pbar:
        for epoch in range(1, epochs + 1):
            model.train()
            pbar.update()
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]

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

                with torch.autocast(device.type, enabled=amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred.float(), true_masks.float())

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                experiment.log({"train loss": loss.item(), "epoch": epoch})
                if epoch > division_epoch:
                    pbar.set_postfix(
                        **{"loss (batch)": loss.item(), "error": val_score.item()}
                    )
                else:
                    pbar.set_postfix(**{"loss (batch)": loss.item(), "error": "N/A"})

            # Evaluation round
            if epoch % division_epoch == 0:
                val_score = evaluate(
                    model, val_loader, device, amp, criterion, experiment
                )
                scheduler.step(val_score)
                pbar.set_postfix(
                    **{"loss (batch)": loss.item(), "error": val_score.float().cpu()}
                )

                if save_checkpoint:
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict_ = model.state_dict()
                    torch.save(
                        state_dict_, str(dir_checkpoint / "checkpoint_epoch.pth")
                    )
