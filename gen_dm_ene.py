import argparse
import gc
from itertools import product

from cadft import CC_DFT_DATA, add_args, gen_logger
from cadft import MAIN_PATH, extend


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
    molecular, name = extend(name_mol, extend_atom, extend_xyz, distance, args.basis)
    if molecular is None:
        print(f"Skip: {name:>40}")
        continue

    spin = 0
    if "openshell" in name_mol:
        spin = 1

    dft2cc = CC_DFT_DATA(
        molecular,
        name=name,
        basis=args.basis,
        if_basis_str=args.if_basis_str,
        spin=spin,
    )

    if abs(distance) >= 0.5:
        FACTOR = 0.9
    elif abs(distance) >= 0.3:
        FACTOR = 0.85
    else:
        FACTOR = 0.8
    if "openshell" in name_mol:
        dft2cc.umrks_diis(0, args.load_inv)
    else:
        dft2cc.mrks_diis(0, args.load_inv)
    # dft2cc.umrks_diis(FACTOR, args.load_inv)
    # dft2cc.mrks_append(FACTOR, args.load_inv)

    del dft2cc
    gc.collect()
