import argparse
from pathlib import Path
import gc
import copy
from itertools import product

from cadft import CC_DFT_DATA, Mol, add_args, gen_logger

path = Path("./data")
parser = argparse.ArgumentParser(
    description="Generate the inversed potential and energy."
)
args = add_args(parser)

distance_l = gen_logger(args.distance_list)

weight_path = Path("data") / "weight"
weight_path.mkdir(parents=True, exist_ok=True)
output_path = Path("data") / "output"
output_path.mkdir(parents=True, exist_ok=True)
input_path = Path("data") / "input"
input_path.mkdir(parents=True, exist_ok=True)

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

    if abs(distance) < 1e-3:
        if (extend_atom != 0) or extend_xyz != 1:
            print(f"Skip {name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}")
            continue

    if extend_atom >= len(Mol[name_mol]):
        print(
            f"\rSkip: {name_mol:>20}_{extend_atom}_{extend_xyz}_{distance:.4f}",
            end="",
        )
        continue

    molecular[extend_atom][extend_xyz] += distance

    dft2cc = CC_DFT_DATA(
        molecular,
        name=f"{name_mol}_{extend_atom}_{extend_xyz}_{distance:.4f}",
        basis=args.basis,
        if_basis_str=args.if_basis_str,
    )
    dft2cc.save_dm1(args.cc_triple)

    del dft2cc
    gc.collect()
