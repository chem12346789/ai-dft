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

from cadft.utils import load_to_gpu, NAO
from cadft.utils import add_args, save_csv_loss, FCNet, DataBase, BasicDataset


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

    today = datetime.datetime.today()
    dir_checkpoint = Path(f"./checkpoint-{today:%Y-%m-%d-%H-%M-%S}-{args.hidden_size}/")
    print(
        f"Start training at {today:%Y-%m-%d-%H-%M-%S} with hidden size as {args.hidden_size}"
    )
    dir_checkpoint.mkdir(parents=True, exist_ok=True)
    (dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

    key_l = []
    model_dict = {}
    optimizer_dict = {}
    scheduler_dict = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i_atom, j_atom in product(ATOM_LIST, ATOM_LIST):
        atom_name = i_atom + j_atom
        key_l.append(atom_name)

        model_dict[atom_name] = FCNet(
            NAO[i_atom] * NAO[j_atom], args.hidden_size, 1
        ).to(device)
        model_dict[atom_name].double()

    if args.load != "":
        dir_load = Path(f"./checkpoint-{args.load}-{args.hidden_size}/")
        for i_atom, j_atom in product(ATOM_LIST, ATOM_LIST):
            atom_name = i_atom + j_atom
            list_of_path = list(dir_load.glob(f"{atom_name}*.pth"))
            if len(list_of_path) == 0:
                print(f"No model found for {atom_name}, use random initialization.")
                continue
            load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
            state_dict = torch.load(load_path, map_location=device)
            model_dict[atom_name].load_state_dict(state_dict)
            print(f"Model loaded from {load_path}")

    database_train = DataBase(args, ATOM_LIST, TRAIN_STR_DICT, device)
    database_eval = DataBase(args, ATOM_LIST, EVAL_STR_DICT, device)

    experiment.config.update(
        {
            "batch_size": args.batch_size,
            "n_val": len(database_eval.input["HH"]),
            "n_train": len(database_train.input["HH"]),
        }
    )

    train_dict = {}
    ntrain_dict = {}
    eval_dict = {}
    neval_dict = {}
    for i_atom, j_atom in product(ATOM_LIST, ATOM_LIST):
        atom_name = i_atom + j_atom
        dataset = BasicDataset(
            database_train.input[atom_name],
            database_train.output[atom_name],
        )
        train_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
        )
        train_dict[atom_name] = load_to_gpu(train_loader, device)
        ntrain_dict[atom_name] = len(database_train.input[atom_name])

        dataset = BasicDataset(
            database_eval.input[atom_name],
            database_eval.output[atom_name],
        )
        eval_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=8,
            pin_memory=True,
        )
        eval_dict[atom_name] = load_to_gpu(eval_loader, device)
        neval_dict[atom_name] = len(database_eval.input[atom_name])

        optimizer_dict[atom_name] = optim.Adam(
            model_dict[atom_name].parameters(),
            lr=1e-4,
        )
        scheduler_dict[atom_name] = optim.lr_scheduler.ExponentialLR(
            optimizer_dict[atom_name],
            gamma=1 - 1e-4,
        )

    loss_fn = nn.L1Loss()

    pbar = trange(1, args.epoch + 1)
    for epoch in pbar:
        train_loss_sum = []
        for key in key_l:
            model_dict[key].train(True)
            train_loss = []
            optimizer_dict[key].zero_grad(set_to_none=True)

            for batch in train_dict[key]:
                with torch.autocast(device.type):
                    input_mat = batch["input"]
                    output_mat_real = batch["output"]

                    output_mat = model_dict[key](input_mat)
                    loss = loss_fn(output_mat, output_mat_real)
                    train_loss.append(loss.item() * input_mat.shape[0])
                    loss.backward()
                    optimizer_dict[key].step()

            scheduler_dict[key].step()

            train_loss_sum.append(np.sum(train_loss) / ntrain_dict[key])
            if epoch % args.eval_step == 0:
                experiment.log(
                    {f"train loss {key}": np.sum(train_loss) / ntrain_dict[key]}
                )

        if epoch % args.eval_step == 0:
            experiment.log({"epoch": epoch})
            experiment.log({"train loss": np.mean(train_loss_sum)})
            pbar.set_description(f"train loss: {np.mean(train_loss_sum):5.3e}")
            eval_loss_sum = []
            for key in key_l:
                eval_loss = []
                model_dict[key].eval()
                for batch in eval_dict[key]:
                    input_mat = batch["input"]
                    output_mat_real = batch["output"]
                    with torch.no_grad():
                        output_mat = model_dict[key](input_mat)
                        eval_loss.append(
                            loss_fn(output_mat, output_mat_real).item()
                            * input_mat.shape[0]
                        )

                experiment.log(
                    {f"eval loss {key}": np.sum(eval_loss) / neval_dict[key]}
                )
                eval_loss_sum.append(np.mean(eval_loss) / neval_dict[key])

            experiment.log({"eval loss": np.mean(eval_loss_sum)})

        if epoch % 10000 == 0:
            for key in key_l:
                state_dict_ = model_dict[key].state_dict()
                torch.save(
                    state_dict_,
                    dir_checkpoint / f"{key}-{epoch}.pth",
                )
    pbar.close()
