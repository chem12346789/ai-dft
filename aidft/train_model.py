"""Module providing a training method."""

import logging
from pathlib import Path
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader, random_split

from .evaluate import evaluate
from .data_loading import BasicDataset
from .select_optimizer_scheduler import select_optimizer_scheduler
from .aux import numpy2str, Criterion, load_to_gpu


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
    n_val = int(len(dataset) * args.val_precent)
    n_train = len(dataset) - n_val  # 1 - val% of the data is used for training
    train_set, val_set = random_split(dataset, [n_train, n_val])
    logging.info("""Split into train / validation partitions.""")

    # 3. Create data loaders
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.batch_size,
        pin_memory=True,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # print the data
    logging.info("train_loader\n")
    for batch in train_loader:
        logging.debug("image %s", numpy2str(batch["image"]))
        logging.debug("mask %s", numpy2str(batch["mask"]))
        logging.info("name %s\n", batch["name"])
    logging.info("val_loader\n")
    for batch in val_loader:
        logging.debug("image %s", numpy2str(batch["image"]))
        logging.debug("mask %s", numpy2str(batch["mask"]))
        logging.info("name %s\n", batch["name"])

    # # load the whole data to the device
    train_loader_gpu = load_to_gpu(train_loader, device)
    val_loader_gpu = load_to_gpu(val_loader, device)

    # Set up the loss function
    criterion = Criterion()

    # (Initialize logging)
    experiment = wandb.init(
        project="UNet",
        resume="allow",
        name=f"{args.optimizer}-{args.scheduler}-{args.name}",
        dir="/home/chenzihao/workdir/tmp",
    )

    experiment.config.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "Device": device.type,
            "Mixed Precision": args.amp,
            "Data_img": dir_img,
            "Data_mask": dir_mask,
            "n_val": n_val,
            "n_train": n_train,
            "Checkpoints": dir_checkpoint,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
        }
    )

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the
    # loss scaling for AMP
    optimizer, scheduler = select_optimizer_scheduler(model, args)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    val_score = None

    # 5. Begin training
    with tqdm(total=args.epochs, unit="epoch") as pbar:
        for epoch in range(1, args.epochs + 1):
            model.train()

            for batch in train_loader_gpu:
                optimizer.zero_grad(set_to_none=True)

                image = batch["image"]
                mask_true = batch["mask"]

                with torch.autocast(device.type, enabled=args.amp):
                    mask_pred = model(image)
                    loss = criterion.val(mask_pred, mask_true)

                print(f"loss: {loss}, type of loss: {type(loss)}")
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

            if args.scheduler != "plateau":
                scheduler.step()

            experiment.log(
                {
                    "epoch": epoch,
                    "learning rate": optimizer.param_groups[0]["lr"],
                    "train loss": loss.item(),
                    "val loss": 0 if (val_score is None) else val_score.item(),
                }
            )

            pbar.update()

            if epoch % args.division_epoch == 0:
                if n_val != 0:
                    val_score = evaluate(
                        model,
                        val_loader_gpu,
                        device,
                        args.amp,
                        criterion,
                        logging,
                        experiment,
                    )
                    if args.scheduler == "plateau":
                        scheduler.step(val_score)
                else:
                    val_score = torch.tensor(0)

            if epoch % args.save_epoch == 0:
                if save_checkpoint:
                    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                    state_dict_ = model.state_dict()
                    torch.save(
                        state_dict_,
                        dir_checkpoint
                        / f"{args.optimizer}-{args.scheduler}-{epoch}.pth",
                    )
