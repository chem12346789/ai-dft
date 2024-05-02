"""Module providing a training method."""

import argparse
from pathlib import Path
import copy
from tqdm import trange
import datetime

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

import numpy as np
import pandas as pd
import opt_einsum as oe
import pyscf
import wandb

from cadft.utils import load_to_gpu, NAO
from cadft.utils import add_args, save_csv_loss, FCNet, DataBase, BasicDataset


ATOM_LIST = [
    "H",
    "C",
]
TRAIN_STR_DICT = [
    "Methane",
    "Ethane",
    "Ethylene",
    "Acetylene",
    "Allene",
    "Cyclopropene",
    "Propyne",
    "Cyclopropane",
    "Propylene",
    "Propane",
    "Isobutane",
    "Butane",
    "Butadiene",
    "Butyne",
    "Bicyclobutane",
    "Cyclopropylmethyl",
    "Cyclobutane",
    "Spiropentane",
    "Benzene",
]
EVAL_STR_DICT = [
    "Pentane",
    "Isopentane",
    "Neopentane",
    "Cyclopentane",
]

# 0. Init the criterion and the model
parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
args = add_args(parser)

experiment = wandb.init(
    project="DFT2CC",
    resume="allow",
    name="run1",
    dir="/home/chenzihao/workdir/tmp",
)

today = datetime.datetime.today()
dir_checkpoint = Path(f"./checkpoint{today:%Y-%m-%d-%H-%M-%S}/")
dir_checkpoint.mkdir(parents=True, exist_ok=True)
(dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

key_l = []
model_dict = {}
dataset_dict = {}
train_dict = {}
eval_dict = {}
optimizer_dict = {}
scheduler_dict = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i_atom in ATOM_LIST:
    for j_atom in ATOM_LIST:
        atom_name = i_atom + j_atom
        key_l.append(atom_name)

        model_dict[atom_name + "1"] = FCNet(
            NAO[i_atom] * NAO[j_atom], 100, NAO[i_atom] * NAO[j_atom]
        ).to(device)
        model_dict[atom_name + "1"].double()

        model_dict[atom_name + "2"] = FCNet(NAO[i_atom] * NAO[j_atom], 100, 1).to(
            device
        )
        model_dict[atom_name + "2"].double()

if args.load:
    dir_load = Path(f"./{args.load}/")
    for i_atom in ATOM_LIST:
        for j_atom in ATOM_LIST:
            for i_str in ["1", "2"]:
                atom_name = i_atom + j_atom
                list_of_path = dir_load.glob(f"{atom_name}-{i_str}*.pth")
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                state_dict = torch.load(load_path, map_location=device)
                model_dict[atom_name + i_str].load_state_dict(state_dict)
                print(f"Model loaded from {load_path}")

database_train = DataBase(args, ATOM_LIST, TRAIN_STR_DICT, device)
database_eval = DataBase(args, ATOM_LIST, EVAL_STR_DICT, device)
print(database_train.check())
print(database_eval.check())

for i_atom in ATOM_LIST:
    for j_atom in ATOM_LIST:
        atom_name = i_atom + j_atom
        dataset = BasicDataset(
            database_train.input[atom_name],
            database_train.middle[atom_name],
            database_train.output[atom_name],
        )
        train_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
        )
        train_dict[atom_name] = load_to_gpu(train_loader, device)

        dataset = BasicDataset(
            database_eval.input[atom_name],
            database_eval.middle[atom_name],
            database_eval.output[atom_name],
        )
        eval_loader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
        )
        eval_dict[atom_name] = load_to_gpu(eval_loader, device)

        optimizer_dict[atom_name + "1"] = optim.Adam(
            model_dict[atom_name + "1"].parameters(),
            lr=0.0001,
        )
        scheduler_dict[atom_name + "1"] = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_dict[atom_name + "1"],
            T_max=250,
        )
        optimizer_dict[atom_name + "2"] = optim.Adam(
            model_dict[atom_name + "2"].parameters(),
            lr=0.0001,
        )
        scheduler_dict[atom_name + "2"] = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_dict[atom_name + "2"],
            T_max=250,
        )

loss_fn = nn.MSELoss()

pbar = trange(1, 100000 + 1)
for epoch in pbar:
    train_loss1 = []
    train_loss2 = []
    for key in key_l:
        optimizer_dict[key + "1"].zero_grad(set_to_none=True)
        optimizer_dict[key + "2"].zero_grad(set_to_none=True)

        for batch in train_dict[key]:
            with torch.autocast(device.type):
                input_mat = batch["input"]
                middle_mat_real = batch["middle"]
                output_mat_real = batch["output"]

                middle_mat = model_dict[key + "1"](input_mat)
                loss = loss_fn(middle_mat, middle_mat_real)
                train_loss1.append(loss.item())
                loss.backward()
                optimizer_dict[key + "1"].step()
                scheduler_dict[key + "1"].step()

                output_mat = model_dict[key + "2"](middle_mat_real)
                loss = loss_fn(output_mat, output_mat_real)
                train_loss2.append(loss.item())
                loss.backward()
                optimizer_dict[key + "2"].step()
                scheduler_dict[key + "2"].step()

    pbar.set_description(
        f"train loss: {np.mean(train_loss1):5.3e} {np.mean(train_loss2):5.3e}"
    )

    if epoch % args.eval_step == 0:
        eval_loss1 = []
        eval_loss2 = []
        for key in key_l:
            for batch in eval_dict[key]:
                input_mat = batch["input"]
                middle_mat_real = batch["middle"]
                output_mat_real = batch["output"]

                middle_mat = model_dict[key + "1"](input_mat)
                eval_loss1.append(loss_fn(middle_mat, middle_mat_real).item())

                output_mat = model_dict[key + "2"](middle_mat_real)
                eval_loss2.append(loss_fn(output_mat, output_mat_real).item())
        experiment.log({"epoch": epoch})
        experiment.log({"train loss1": np.mean(train_loss1)})
        experiment.log({"train loss2": np.mean(train_loss2)})
        experiment.log({"eval loss1": np.mean(eval_loss1)})
        experiment.log({"eval loss2": np.mean(eval_loss2)})

    if epoch % 10000 == 0:
        for key in key_l:
            for i_str in ["1", "2"]:
                state_dict_ = model_dict[key + i_str].state_dict()
                torch.save(
                    state_dict_,
                    dir_checkpoint / f"{key}-{i_str}-{epoch}.pth",
                )

dice_after_train = database_train.check(model_dict, if_equilibrium=False)
save_csv_loss(dice_after_train, dir_checkpoint / "loss" / f"train-{epoch}.csv")
dice_after_train = database_eval.check(model_dict, if_equilibrium=False)
save_csv_loss(dice_after_train, dir_checkpoint / "loss" / f"eval-{epoch}.csv")
pbar.close()
