"""Module providing a training method."""

import argparse
import os

from tqdm import trange
import torch

import numpy as np
import wandb

from cadft.utils import add_args, save_csv_loss
from cadft.utils import DataBase, ModelDict


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Modeldict = ModelDict(
        args.hidden_size,
        args.num_layers,
        args.residual,
        device,
        args.precision,
    )
    (Modeldict.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)
    Modeldict.load_model(args.load)

    database_train = DataBase(
        TRAIN_STR_DICT,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
        args.basis,
        args.batch_size,
        device,
        args.precision,
    )
    database_eval = DataBase(
        EVAL_STR_DICT,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
        args.basis,
        args.batch_size,
        device,
        args.precision,
    )

    experiment.config.update(
        {
            "batch_size": args.batch_size,
            "n_train": len(database_train.name_list),
            "n_eval": len(database_eval.name_list),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "residual": args.residual,
            "jobid": os.environ.get("SLURM_JOB_ID"),
            "checkpoint": Modeldict.dir_checkpoint,
        }
    )

    pbar = trange(args.epoch + 1)
    for epoch in pbar:
        train_loss_1, train_loss_2 = Modeldict.train_model(database_train)
        if not isinstance(
            Modeldict.scheduler_dict["1"],
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            Modeldict.scheduler_dict["1"].step()
        if not isinstance(
            Modeldict.scheduler_dict["2"],
            torch.optim.lr_scheduler.ReduceLROnPlateau,
        ):
            Modeldict.scheduler_dict["2"].step()

        if epoch % args.eval_step == 0:
            eval_loss_1, eval_loss_2 = Modeldict.eval_model(database_eval)
            if isinstance(
                Modeldict.scheduler_dict["1"],
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                Modeldict.scheduler_dict["1"].step(np.mean(eval_loss_1))
            if isinstance(
                Modeldict.scheduler_dict["2"],
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                Modeldict.scheduler_dict["2"].step(np.mean(eval_loss_2))

            experiment.log(
                {
                    "epoch": epoch,
                    "global_step": epoch,
                    "mean train1 loss": np.mean(train_loss_1),
                    "mean train2 loss": np.mean(train_loss_2),
                    "mean eval1 loss": np.mean(eval_loss_1),
                    "mean eval2 loss": np.mean(eval_loss_2),
                    "lr1": Modeldict.optimizer_dict["1"].param_groups[0]["lr"],
                    "lr2": Modeldict.optimizer_dict["2"].param_groups[0]["lr"],
                }
            )

            pbar.set_description(
                f"t/e1: {np.mean(train_loss_1):.1e}/{np.mean(eval_loss_1):.1e}, "
                f"t/e2: {np.mean(train_loss_2):.1e}/{np.mean(eval_loss_2):.1e}, "
                f"lr1: {Modeldict.optimizer_dict['1'].param_groups[0]['lr']:.1e}"
                f"lr2: {Modeldict.optimizer_dict['2'].param_groups[0]['lr']:.1e}"
            )

        if epoch % 500 == 0:
            save_csv_loss(
                database_train.name_list,
                Modeldict.dir_checkpoint / "loss" / f"train-loss-{epoch}.csv",
                train_loss_1,
                train_loss_2,
            )
            save_csv_loss(
                database_eval.name_list,
                Modeldict.dir_checkpoint / "loss" / f"eval-loss-{epoch}.csv",
                eval_loss_1,
                eval_loss_2,
            )
            Modeldict.save_model(epoch)
    pbar.close()
