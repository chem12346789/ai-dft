import argparse
import gc
from itertools import product

from cadft import CC_DFT_DATA, add_args, gen_logger
from cadft import extend


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

    SPIN = 0
    if "openshell" in name_mol:
        SPIN = 1

    dft2cc = CC_DFT_DATA(
        molecular,
        name=name,
        basis=args.basis,
        if_basis_str=args.if_basis_str,
        spin=SPIN,
    )

    if abs(distance) >= 1.5:
        FACTOR = 0.5
        DIIS_N = 50
    else:
        FACTOR = 0
        DIIS_N = 20

    print(f"FACTOR: {FACTOR}, diis_n: {DIIS_N}")

    if "openshell" in name_mol:
        vxc_inv = dft2cc.umrks_diis(
            FACTOR,
            args.load_inv,
            diis_n=DIIS_N,
            vxc_inv=None,
        )
    else:
        vxc_inv = dft2cc.mrks_diis(
            FACTOR,
            args.load_inv,
            diis_n=DIIS_N,
            vxc_inv=None,
        )

    # dft2cc.deepks()
    # if "openshell" in name_mol:
    #     dft2cc.umrks_append()
    # else:
    #     dft2cc.mrks_append()

    del dft2cc
    gc.collect()
