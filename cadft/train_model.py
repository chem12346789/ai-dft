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
    gen_keys_l,
    gen_model_dict,
    load_model,
)
from cadft.utils import DataBase, BasicDataset


def train_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT):
    """
    Train the model.
    ATOM_LIST: list of atom names
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
        name=f"run1-{args.hidden_size}",
        dir="/home/chenzihao/workdir/tmp",
    )
    wandb.define_metric("*", step_metric="global_step")

    today = datetime.datetime.today()
    dir_checkpoint = Path(
        f"checkpoints/checkpoint-{today:%Y-%m-%d-%H-%M-%S}-{args.hidden_size}/"
    )
    print(
        f"Start training at {today:%Y-%m-%d-%H-%M-%S} with hidden size as {args.hidden_size}"
    )
    dir_checkpoint.mkdir(parents=True, exist_ok=True)
    (dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

    keys_l = gen_keys_l(ATOM_LIST)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = gen_model_dict(keys_l, args, device)

    optimizer_dict = {}
    scheduler_dict = {}
    for key in keys_l:
        optimizer_dict[key + "1"] = optim.Adam(
            model_dict[key + "1"].parameters(),
            lr=1e-4,
        )
        scheduler_dict[key + "1"] = optim.lr_scheduler.ExponentialLR(
            optimizer_dict[key + "1"],
            gamma=1 - 1e-4,
        )

        optimizer_dict[key + "2"] = optim.Adam(
            model_dict[key + "2"].parameters(),
            lr=1e-4,
        )
        scheduler_dict[key + "2"] = optim.lr_scheduler.ExponentialLR(
            optimizer_dict[key + "2"],
            gamma=1 - 1e-4,
        )
    load_model(model_dict, keys_l, args, device)

    database_train = DataBase(args, keys_l, TRAIN_STR_DICT, device)
    database_eval = DataBase(args, keys_l, EVAL_STR_DICT, device)

    train_dict = {}
    ntrain_dict = {}
    eval_dict = {}
    neval_dict = {}
    for key in keys_l:
        dataset = BasicDataset(
            database_train.input[key],
            database_train.middle[key],
            database_train.output[key],
        )
        train_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
        )
        train_dict[key] = load_to_gpu(train_loader, device)
        ntrain_dict[key] = len(database_train.input[key])

        dataset = BasicDataset(
            database_eval.input[key],
            database_eval.middle[key],
            database_eval.output[key],
        )
        eval_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
        )
        eval_dict[key] = load_to_gpu(eval_loader, device)
        neval_dict[key] = len(database_eval.input[key])

    update_d = {
        "batch_size": args.batch_size,
    }
    for k, v in ntrain_dict.items():
        update_d[f"n_train_{k}"] = v
    for k, v in neval_dict.items():
        update_d[f"n_val_{k}"] = v
    experiment.config.update(update_d)

    loss_fn = nn.L1Loss()

    pbar = trange(1, args.epoch + 1)
    for epoch in pbar:
        train_loss_sum_1 = {}
        train_loss_sum_2 = {}
        for key in keys_l:
            model_dict[key + "1"].train(True)
            model_dict[key + "2"].train(True)
            train_loss_1 = []
            train_loss_2 = []
            optimizer_dict[key + "1"].zero_grad(set_to_none=True)
            optimizer_dict[key + "2"].zero_grad(set_to_none=True)

            for batch in train_dict[key]:
                with torch.autocast(device.type):
                    input_mat = batch["input"]
                    middle_mat_real = batch["middle"]

                    middle_mat = model_dict[key + "1"](input_mat)
                    loss_1 = loss_fn(middle_mat, middle_mat_real)
                    loss_1.backward()
                    train_loss_1.append(
                        loss_1.item() * input_mat.shape[0] / ntrain_dict[key]
                    )
                    optimizer_dict[key + "1"].step()

                    output_mat_real = batch["output"]
                    output_mat = model_dict[key + "2"](middle_mat_real)
                    loss_2 = loss_fn(output_mat, output_mat_real)
                    loss_2.backward()
                    train_loss_2.append(
                        loss_2.item() * input_mat.shape[0] / ntrain_dict[key]
                    )
                    optimizer_dict[key + "2"].step()

            scheduler_dict[key + "1"].step()
            scheduler_dict[key + "2"].step()
            train_loss_sum_1[key] = np.sum(train_loss_1)
            train_loss_sum_2[key] = np.sum(train_loss_2)

        if epoch % args.eval_step == 0:
            eval_loss_sum_1 = {}
            eval_loss_sum_2 = {}
            for key in keys_l:
                eval_loss_1 = []
                eval_loss_2 = []
                model_dict[key + "1"].eval()
                model_dict[key + "2"].eval()

                for batch in eval_dict[key]:
                    input_mat = batch["input"]
                    middle_mat_real = batch["middle"]
                    output_mat_real = batch["output"]
                    with torch.no_grad():
                        middle_mat = model_dict[key + "1"](input_mat)
                        eval_loss_1.append(
                            loss_fn(middle_mat, middle_mat_real).item()
                            * input_mat.shape[0]
                            / neval_dict[key]
                        )

                        output_mat = model_dict[key + "2"](middle_mat_real)
                        eval_loss_2.append(
                            loss_fn(output_mat, output_mat_real).item()
                            * input_mat.shape[0]
                            / neval_dict[key]
                        )

                eval_loss_sum_1[key] = np.sum(eval_loss_1)
                eval_loss_sum_2[key] = np.sum(eval_loss_2)

            lod_d = {
                "epoch": epoch,
                "global_step": epoch,
                "mean train1 loss": np.mean(list(train_loss_sum_1.values())),
                "mean train2 loss": np.mean(list(train_loss_sum_2.values())),
                "mean eval1 loss": np.mean(list(eval_loss_sum_1.values())),
                "mean eval2 loss": np.mean(list(eval_loss_sum_2.values())),
            }

            for k, v in train_loss_sum_1.items():
                lod_d[f"train1 loss/ {k}"] = v
            for k, v in train_loss_sum_2.items():
                lod_d[f"train2 loss/ {k}"] = v
            for k, v in eval_loss_sum_1.items():
                lod_d[f"eval1 loss/ {k}"] = v
            for k, v in eval_loss_sum_2.items():
                lod_d[f"eval2 loss/ {k}"] = v

            experiment.log(lod_d)

            pbar.set_description(
                f"epoch: {epoch}, "
                f"train1: {np.mean(list(train_loss_sum_1.values())):.4f}, "
                f"eval1: {np.mean(list(eval_loss_sum_1.values())):.4f}, "
                f"train2: {np.mean(list(train_loss_sum_2.values())):.4f}, "
                f"eval2: {np.mean(list(eval_loss_sum_2.values())):.4f}"
            )

        if epoch % 5000 == 0:
            for key in keys_l:
                for i_str in ["1", "2"]:
                    state_dict_ = model_dict[key + i_str].state_dict()
                    torch.save(
                        state_dict_,
                        dir_checkpoint / f"{key}-{i_str}-{epoch}.pth",
                    )
    pbar.close()
