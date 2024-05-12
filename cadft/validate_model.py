import argparse
from pathlib import Path
import datetime
from itertools import product

import torch

from cadft.utils import NAO
from cadft.utils import add_args, save_csv_loss, FCNet, DataBase


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

    if args.load != "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    keys_l = []
    model_dict = {}

    for i_atom, j_atom in product(ATOM_LIST, ATOM_LIST):
        if i_atom != j_atom:
            atom_name = f"{i_atom}-{j_atom}"
            keys_l.append(atom_name)
        else:
            atom_name = f"{i_atom}-{i_atom}-D"
            keys_l.append(atom_name)
            atom_name = f"{i_atom}-{i_atom}-O"
            keys_l.append(atom_name)

    for key in keys_l:
        model_dict[key] = FCNet(
            NAO[key.split("-")[0]] * NAO[key.split("-")[1]], args.hidden_size, 1
        ).to(device)
        model_dict[key].double()

    database_train = DataBase(args, keys_l, TRAIN_STR_DICT, device)
    database_eval = DataBase(args, keys_l, EVAL_STR_DICT, device)

    if args.load != "":
        dir_load = Path(f"./checkpoint-{args.load}-{args.hidden_size}/")
        for key in keys_l:
            list_of_path = list(dir_load.glob(f"{key}*.pth"))
            if len(list_of_path) == 0:
                print(f"No model found for {key}, use random initialization.")
                continue
            load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
            state_dict = torch.load(load_path, map_location=device)
            model_dict[key].load_state_dict(state_dict)
            print(f"Model loaded from {load_path}")

        dice_train = database_train.check(model_dict, if_equilibrium=False)
        dice_eval = database_eval.check(model_dict, if_equilibrium=False)
    else:
        dice_train = database_train.check(if_equilibrium=False)
        dice_eval = database_eval.check(if_equilibrium=False)

    save_csv_loss(dice_train, dir_validate / "train.csv")
    save_csv_loss(dice_eval, dir_validate / "eval.csv")
