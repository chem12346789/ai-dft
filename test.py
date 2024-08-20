"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
import copy
from itertools import product

import pandas as pd
import torch

from cadft.utils import ModelDict
from cadft import add_args, gen_logger
from cadft import test_rks, test_uks
from cadft.utils import Mol


if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init the model
    modeldict = ModelDict(
        args.load,
        args.input_size,
        args.hidden_size,
        args.output_size,
        args.num_layers,
        args.residual,
        device,
        args.precision,
        if_mkdir=False,
        load_epoch=args.load_epoch,
    )
    modeldict.load_model()
    modeldict.eval()

    # 2. Test loop
    df_dict = {
        "name": [],
        "error_scf_ene": [],
        "error_dft_ene": [],
        "abs_scf_ene": [],
        "abs_dft_ene": [],
        "abs_cc_ene": [],
        "error_scf_rho_r": [],
        "error_dft_rho_r": [],
        "dipole_x_diff_scf": [],
        "dipole_y_diff_scf": [],
        "dipole_z_diff_scf": [],
        "dipole_x_diff_dft": [],
        "dipole_y_diff_dft": [],
        "dipole_z_diff_dft": [],
        "time_cc": [],
        "time_dft": [],
    }

    distance_l = gen_logger(args.distance_list)
    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        args.name_mol,
        args.extend_atom,
        args.extend_xyz,
        distance_l,
    ):
        molecular = copy.deepcopy(Mol[name_mol])
        name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        df_dict["name"].append(name)
        print(f"Generate {name_mol}_{distance:.4f}", flush=True)
        print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}", flush=True)

        if abs(distance) < 1e-3:
            if (extend_atom != 0) or extend_xyz != 1:
                print(f"Skip: {name:>40}")
                continue

        if extend_atom >= len(Mol[name_mol]):
            print(f"Skip: {name:>40}")
            continue

        molecular[extend_atom][extend_xyz] += distance

        if "openshell" in name_mol:
            test_uks(args, molecular, name, modeldict, df_dict)
        else:
            test_rks(args, molecular, name, modeldict, df_dict)
