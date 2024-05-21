"""Module providing a training method."""

import argparse
from pathlib import Path
import copy
import datetime
from itertools import product

from tqdm import trange
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import numpy as np
import wandb

from cadft.utils import (
    add_args,
    load_to_gpu,
    gen_model_dict,
    load_model,
)
from cadft.utils import DataBase, BasicDataset


def train_model(TRAIN_STR_DICT, EVAL_STR_DICT):
    """
    Train the model.
    TRAIN_STR_DICT: list of training molecules
    EVAL_STR_DICT: list of evaluation molecules
    Other parameter are from the argparse.
    """
    # 0. Init the criterion and the model
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    experiment = wandb.init(
        project="DFT2CC",
        resume="allow",
        name=f"ccdft-{args.hidden_size}",
        dir="/home/chenzihao/workdir/tmp",
    )
    wandb.define_metric("*", step_metric="global_step")

    today = datetime.datetime.today()
    dir_checkpoint = Path(
        f"checkpoints/checkpoint-ccdft-{today:%Y-%m-%d-%H-%M-%S}-{args.hidden_size}/"
    )
    print(
        f"Start training at {today:%Y-%m-%d-%H-%M-%S} with hidden size as {args.hidden_size}"
    )
    dir_checkpoint.mkdir(parents=True, exist_ok=True)
    (dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = gen_model_dict(args.hidden_size, device)

    optimizer_dict = {}
    scheduler_dict = {}
    optimizer_dict["1"] = optim.Adam(
        model_dict["1"].parameters(),
        lr=1e-4,
    )
    scheduler_dict["1"] = optim.lr_scheduler.ExponentialLR(
        optimizer_dict["1"],
        gamma=1 - 1e-4,
    )

    optimizer_dict["2"] = optim.Adam(
        model_dict["2"].parameters(),
        lr=1e-4,
    )
    scheduler_dict["2"] = optim.lr_scheduler.ExponentialLR(
        optimizer_dict["2"],
        gamma=1 - 1e-4,
    )
    load_model(model_dict, args.load, args.hidden_size, device)

    database_train = DataBase(args, TRAIN_STR_DICT, device)
    database_eval = DataBase(args, EVAL_STR_DICT, device)

    dataset = BasicDataset(
        database_train.input,
        database_train.middle,
        database_train.output,
    )
    train_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    train_set = load_to_gpu(train_loader, device)
    ntrain_set = len(database_train.input)

    dataset = BasicDataset(
        database_eval.input,
        database_eval.middle,
        database_eval.output,
    )
    eval_loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    eval_set = load_to_gpu(eval_loader, device)
    neval_set = len(database_eval.input)

    update_d = {
        "batch_size": args.batch_size,
        "n_train": ntrain_set,
        "n_val": neval_set,
    }
    print(update_d)
    experiment.config.update(update_d)

    loss_fn = nn.L1Loss()

    pbar = trange(args.epoch + 1)
    for epoch in pbar:
        model_dict["1"].train(True)
        model_dict["2"].train(True)
        optimizer_dict["1"].zero_grad(set_to_none=True)
        optimizer_dict["2"].zero_grad(set_to_none=True)
        train_loss_1 = 0
        train_loss_2 = 0

        for batch in train_set:
            with torch.autocast(device.type):
                input_mat = batch["input"]
                middle_mat_real = batch["middle"]

                middle_mat = model_dict["1"](input_mat)
                loss_1 = loss_fn(middle_mat, middle_mat_real)
                loss_1.backward()
                train_loss_1 += loss_1.item() * input_mat.shape[0] / ntrain_set
                optimizer_dict["1"].step()

                output_mat_real = batch["output"]
                output_mat = model_dict["2"](input_mat + middle_mat_real)
                loss_2 = loss_fn(output_mat, output_mat_real)
                loss_2.backward()
                train_loss_2 += loss_2.item() * input_mat.shape[0] / ntrain_set
                optimizer_dict["2"].step()

        scheduler_dict["1"].step()
        scheduler_dict["2"].step()

        if epoch % args.eval_step == 0:
            model_dict["1"].eval()
            model_dict["2"].eval()

            eval_loss_1 = 0
            eval_loss_2 = 0

            for batch in eval_set:
                input_mat = batch["input"]
                middle_mat_real = batch["middle"]
                output_mat_real = batch["output"]
                with torch.no_grad():
                    middle_mat = model_dict["1"](input_mat)
                    eval_loss_1 += (
                        loss_fn(middle_mat, middle_mat_real).item()
                        * input_mat.shape[0]
                        / neval_set
                    )

                    output_mat = model_dict["2"](input_mat + middle_mat_real)
                    eval_loss_2 += (
                        loss_fn(output_mat, output_mat_real).item()
                        * input_mat.shape[0]
                        / neval_set
                    )

            lod_d = {
                "epoch": epoch,
                "global_step": epoch,
                "mean train1 loss": train_loss_1,
                "mean eval1 loss": eval_loss_1,
                "mean train2 loss": train_loss_2,
                "mean eval2 loss": eval_loss_2,
            }
            experiment.log(lod_d)

            pbar.set_description(
                f"epoch: {epoch}, "
                f"train1: {train_loss_1:7.4e}, "
                f"train2: {train_loss_2:7.4e}, "
                f"eval1: {eval_loss_1:7.4e}, "
                f"eval2: {eval_loss_2:7.4e}."
            )

        if epoch % 5000 == 0:
            for i_str in ["1", "2"]:
                state_dict_ = model_dict[i_str].state_dict()
                torch.save(
                    state_dict_,
                    dir_checkpoint / f"{i_str}-{epoch}.pth",
                )
    pbar.close()
