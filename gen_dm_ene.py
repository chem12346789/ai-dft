import argparse
from pathlib import Path
import gc
import copy
from itertools import product

from cadft import CC_DFT_DATA, Mol, add_args, gen_logger
from cadft import MAIN_PATH

parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
args = add_args(parser)

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
    print(f"Generate {name_mol}_{distance:.4f}")
    print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}")

    name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
    if abs(distance) < 1e-3:
        if (extend_atom != 0) or extend_xyz != 1:
            print(f"Skip: {name:>40}")
            continue

    if extend_atom >= len(Mol[name_mol]):
        print(f"Skip: {name:>40}")
        continue

    molecular[extend_atom][extend_xyz] += distance

    dft2cc = CC_DFT_DATA(
        molecular,
        name=name,
        basis=args.basis,
        if_basis_str=args.if_basis_str,
        spin=1,
    )

    if abs(distance) >= 0.5:
        FACTOR = 0.9
    elif abs(distance) >= 0.3:
        FACTOR = 0.85
    else:
        FACTOR = 0.8
    dft2cc.umrks_diis(0, args.load_inv)
    # dft2cc.umrks_diis(FACTOR, args.load_inv)
    # dft2cc.mrks_append(FACTOR, args.load_inv)

    del dft2cc
    gc.collect()
