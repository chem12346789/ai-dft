"""Module providing a training method."""

import argparse
from pathlib import Path

import torch

from cadft.utils import add_args, save_csv_loss
from cadft.utils import DataBase, ModelDict


def validate_model(TRAIN_STR_DICT):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Modeldict = ModelDict(
        args.hidden_size,
        args.num_layers,
        args.residual,
        device,
        if_mkdir=False,
    )
    Modeldict.load_model(args.load)

    database_eval = DataBase(
        TRAIN_STR_DICT,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
        args.batch_size,
        args.ene_grid_factor,
        device,
    )

    eval_loss_1, eval_loss_2, eval_loss_3 = Modeldict.eval_model(database_eval)

    Path(f"validate/validate-ccdft-{args.load}-{args.hidden_size}/").mkdir(
        parents=True, exist_ok=True
    )
    save_csv_loss(
        database_eval.name_list,
        Path(f"validate/validate-ccdft-{args.load}-{args.hidden_size}/") / "train.csv",
        eval_loss_1,
        eval_loss_2,
        eval_loss_3,
    )
