import argparse
from pathlib import Path
import datetime
from itertools import product

import torch

from cadft.utils import NAO
from cadft.utils import (
    add_args,
    gen_keys_l,
    gen_model_dict,
    load_model,
)
from cadft.utils import add_args, save_csv_loss, DataBase

from cadft.utils import FCNet as Model

# from cadft.utils import Transformer as Model


def validate_model(ATOM_LIST, TRAIN_STR_DICT, EVAL_STR_DICT):
    """
    Validate the model.
    """
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    today = datetime.datetime.today()
    dir_validate = Path(f"validate/validate-{args.load}-{args.hidden_size}/")
    print(f"Start training at {args.load} with hidden size as {args.hidden_size}")
    dir_validate.mkdir(parents=True, exist_ok=True)

    if args.load != "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    keys_l = gen_keys_l(ATOM_LIST)
    model_dict = gen_model_dict(keys_l, args, device)

    database_train = DataBase(args, keys_l, TRAIN_STR_DICT, device)
    database_eval = DataBase(args, keys_l, EVAL_STR_DICT, device)

    if args.load != "":
        load_model(model_dict, keys_l, args, device)
        ai_train = database_train.check(model_dict, if_equilibrium=False)
        ai_eval = database_eval.check(model_dict, if_equilibrium=False)
    else:
        ai_train = database_train.check(if_equilibrium=False)
        ai_eval = database_eval.check(if_equilibrium=False)

    save_csv_loss(ai_train, dir_validate / "train.csv")
    save_csv_loss(ai_eval, dir_validate / "eval.csv")

    # dft_train = database_train.check_dft(if_equilibrium=False)
    # dft_eval = database_eval.check_dft(if_equilibrium=False)
    # save_csv_loss(dft_train, dir_validate / "train_dft.csv")
    # save_csv_loss(dft_eval, dir_validate / "eval_dft.csv")
