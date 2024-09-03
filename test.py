"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
import copy
from itertools import product

import torch

from cadft import add_args, extend, gen_logger
from cadft import test_rks, test_uks
from cadft.utils import Mol

from cadft.utils import ModelDict_xy as ModelDict
# from cadft.utils import ModelDict as ModelDict


if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init the model
    modeldict = ModelDict(
        load=args.load,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
        residual=args.residual,
        device=device,
        precision=args.precision,
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
    dm1_scf = None
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
        molecular, name = extend(
            name_mol, extend_atom, extend_xyz, distance, args.basis
        )
        if molecular is None:
            print(f"Skip: {name:>40}")
            continue
        df_dict["name"].append(name)

        if abs(distance) > 4:
            N_DIIS = 70
        elif abs(distance) > 3:
            N_DIIS = 50
        else:
            N_DIIS = 20
        if "openshell" in name_mol:
            test_uks(args, molecular, name, modeldict, df_dict)
        else:
            dm1_scf = test_rks(
                args,
                molecular,
                name,
                modeldict,
                df_dict,
                n_diis=N_DIIS,
                dm1_scf=dm1_scf,
            )
