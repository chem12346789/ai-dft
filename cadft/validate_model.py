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


def validate_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT):
    """
    Validate the model.
    """
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    today = datetime.datetime.today()
    dir_validate = Path(f"./validate-{today:%Y-%m-%d-%H-%M-%S}-{args.hidden_size}/")
    print(
        f"Start training at {today:%Y-%m-%d-%H-%M-%S} with hidden size as {args.hidden_size}"
    )
    dir_validate.mkdir(parents=True, exist_ok=True)

    key_l = []
    model_dict = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i_atom, j_atom in product(ATOM_LIST, ATOM_LIST):
        atom_name = i_atom + j_atom
        key_l.append(atom_name)

        model_dict[atom_name + "1"] = FCNet(
            NAO[i_atom] * NAO[j_atom], args.hidden_size, NAO[i_atom] * NAO[j_atom]
        ).to(device)
        model_dict[atom_name + "1"].double()

        model_dict[atom_name + "2"] = FCNet(
            NAO[i_atom] * NAO[j_atom], args.hidden_size, 1
        ).to(device)
        model_dict[atom_name + "2"].double()

    if args.load != "":
        dir_load = Path(f"./checkpoint-{args.load}-{args.hidden_size}/")
        for i_atom, j_atom, i_str in product(ATOM_LIST, ATOM_LIST, ["1", "2"]):
            atom_name = i_atom + j_atom
            list_of_path = dir_load.glob(f"{atom_name}-{i_str}*.pth")
            load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
            state_dict = torch.load(load_path, map_location=device)
            model_dict[atom_name + i_str].load_state_dict(state_dict)
            print(f"Model loaded from {load_path}")

    database_train = DataBase(args, ATOM_LIST, TRAIN_STR_DICT, device)
    database_eval = DataBase(args, ATOM_LIST, EVAL_STR_DICT, device)
    dice_after_train = database_train.check(model_dict, if_equilibrium=False)
    save_csv_loss(dice_after_train, dir_validate / "train.csv")
    dice_after_train = database_eval.check(model_dict, if_equilibrium=False)
    save_csv_loss(dice_after_train, dir_validate / "eval.csv")
