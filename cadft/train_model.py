"""Module providing a training method."""

import argparse
import os

from tqdm import trange
import torch

import numpy as np
import wandb

from cadft.utils import add_args, save_csv_loss
from cadft.utils import DataBase, ModelDictUnet as ModelDict


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
        name=f"ccdft_{args.hidden_size}_{args.num_layers}_{args.residual}",
        dir="/home/chenzihao/workdir/tmp",
    )
    wandb.define_metric("*", step_metric="global_step")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modeldict = ModelDict(
        args.load,
        args.input_size,
        args.hidden_size,
        args.output_size,
        args.num_layers,
        args.residual,
        device,
        args.precision,
        with_eval=args.with_eval,
        ene_weight=args.ene_weight,
        pot_weight=args.pot_weight,
    )
    modeldict.load_model()

    database_train = DataBase(
        TRAIN_STR_DICT,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
        args.train_atom_list,
        args.input_size,
        args.output_size,
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
        args.train_atom_list,
        args.input_size,
        args.output_size,
        args.basis,
        args.batch_size,
        device,
        args.precision,
    )

    experiment_dict = {
        "batch_size": args.batch_size,
        "n_train": len(database_train.name_list),
        "n_eval": len(database_eval.name_list),
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "residual": args.residual,
        "ene_weight": args.ene_weight,
        "pot_weight": args.pot_weight,
        "precision": args.precision,
        "basis": args.basis,
        "with_eval": args.with_eval,
        "load": args.load,
        "jobid": os.environ.get("SLURM_JOB_ID"),
        "checkpoint": modeldict.dir_checkpoint.stem,
    }
    print(experiment_dict)
    experiment.config.update(experiment_dict)

    print(f"Start training at {modeldict.dir_checkpoint}")
    pbar0 = trange(args.epoch + 1)
    for epoch in pbar0:
        (
            train_loss_pot,
            train_loss_ene,
            train_loss_ene_tot,
        ) = modeldict.train_model(database_train)
        if not modeldict.with_eval:
            for key in modeldict.keys:
                modeldict.scheduler_dict[key].step()

        if epoch % args.eval_step == 0:
            (
                eval_loss_pot,
                eval_loss_ene,
                eval_loss_ene_tot,
            ) = modeldict.eval_model(database_eval)
            if modeldict.with_eval:
                if modeldict.output_size == 1:
                    modeldict.scheduler_dict["1"].step(np.mean(eval_loss_pot))
                    modeldict.scheduler_dict["2"].step(
                        np.mean(
                            eval_loss_ene + modeldict.ene_weight * eval_loss_ene_tot
                        )
                    )
                elif modeldict.output_size == 2:
                    modeldict.scheduler_dict["1"].step(
                        np.mean(
                            eval_loss_pot
                            + eval_loss_ene
                            + modeldict.ene_weight * eval_loss_ene_tot
                        )
                    )
                elif modeldict.output_size == -1:
                    modeldict.scheduler_dict["1"].step(
                        np.mean(
                            eval_loss_pot
                            + eval_loss_ene
                            + modeldict.ene_weight * eval_loss_ene_tot
                        )
                    )
                elif modeldict.output_size == -2:
                    modeldict.scheduler_dict["1"].step(
                        np.mean(
                            eval_loss_ene + modeldict.ene_weight * eval_loss_ene_tot
                        )
                    )

            experiment_dict = {
                "epoch": epoch,
                "global_step": epoch,
                "train_loss_pot": np.mean(train_loss_pot),
                "train_loss_ene": np.mean(train_loss_ene),
                "train_loss_ene_tot": np.mean(train_loss_ene_tot),
                "eval_loss_pot": np.mean(eval_loss_pot),
                "eval_loss_ene": np.mean(eval_loss_ene),
                "eval_loss_ene_tot": np.mean(eval_loss_ene_tot),
            }
            lr1_2 = ""
            for key in modeldict.keys:
                experiment_dict[f"lr{key}"] = -np.log10(
                    modeldict.optimizer_dict[key].param_groups[0]["lr"]
                )
                lr1_2 += f"{-np.log10(modeldict.optimizer_dict[key].param_groups[0]['lr']):.2g} "
            experiment.log(experiment_dict)

            pbar0.set_description(
                f"t/e1 {np.mean(train_loss_pot):.2e}/{np.mean(eval_loss_pot):.2e} "
                f"t/e2 {np.mean(train_loss_ene):.2e}/{np.mean(eval_loss_ene):.2e} "
                f"t/e3 {np.mean(train_loss_ene_tot):.2e}/{np.mean(eval_loss_ene_tot):.2e} "
                f"lr1/2 {lr1_2}"
            )

        if (epoch % (args.eval_step * 50) == 0) and (epoch != 0):
            save_csv_loss(
                database_train.name_list,
                modeldict.dir_checkpoint / "loss" / f"train-loss-{epoch}",
                {
                    "loss_rho": train_loss_pot,
                    "loss_tot_rho": train_loss_ene,
                    "loss_ene": train_loss_ene_tot,
                },
            )
            save_csv_loss(
                database_eval.name_list,
                modeldict.dir_checkpoint / "loss" / f"eval-loss-{epoch}",
                {
                    "loss_rho": eval_loss_pot,
                    "loss_tot_rho": eval_loss_ene,
                    "loss_ene": eval_loss_ene_tot,
                },
            )
            modeldict.save_model(epoch)
    pbar0.close()
