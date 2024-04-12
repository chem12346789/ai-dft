import numpy as np
from pathlib import Path
import argparse
import json


def clean_dir(pth):
    pth = Path(pth)
    for child in pth.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            clean_dir(child)


parser = argparse.ArgumentParser(description="Obtain data from npy files")

parser.add_argument(
    "--name",
    "-n",
    type=str,
    default="First_Run",
    help="Witch directory we save data to.",
)

parser.add_argument(
    "--energy",
    "-e",
    type=bool,
    default=False,
    help="If contain energy.",
)

parser.add_argument(
    "--atom",
    type=str,
    default="H",
    help="Atom name. Default is 'H'",
)

parser.add_argument(
    "--data",
    "-d",
    nargs="+",
    type=str,
    default="data",
    help='Name of data directory. Could be a list. Default is "data".',
)

parser.add_argument(
    "--nclass",
    "-c",
    type=int,
    default=1,
    help="Number of nclass.",
)

args = parser.parse_args()

main_dir = Path(__file__).resolve().parents[1]
imgs_path = main_dir / args.name / "data" / "imgs"
mask_path = main_dir / args.name / "data" / "masks"
weit_path = main_dir / args.name / "data" / "weights"

imgs_path.mkdir(parents=True, exist_ok=True)
mask_path.mkdir(parents=True, exist_ok=True)
weit_path.mkdir(parents=True, exist_ok=True)
clean_dir(imgs_path)
clean_dir(mask_path)
clean_dir(weit_path)
print(f"data: {args.data}")

MESSAGE = ""

for data_i in args.data:
    data_path = Path("./") / data_i
    for child in (data_path).glob("*"):
        if not child.is_dir():
            continue
        print(f"Processing {child.parts[-1]}")

        rho_input_file = list(child.glob("rho_input.npy"))
        e_output_file = list(child.glob("e_output.npy"))
        exc_file = list(child.glob("exc.npy"))
        tau_rho_wf_file = list(child.glob("tau_rho_wf.npy"))
        ene_nuc_file = list(child.glob("ene_nuc.npy"))
        rho_output_file = list(child.glob("rho_output.npy"))
        weight_file = list(child.glob("weight.npy"))

        if (
            (len(rho_input_file) == 1)
            and (len(e_output_file) == 1)
            and (len(rho_output_file) == 1)
            and (len(exc_file) == 1)
            and (len(tau_rho_wf_file) == 1)
            and (len(ene_nuc_file) == 1)
            and (len(weight_file) == 1)
        ):
            rho_input = np.load(rho_input_file[0])
            e_output = np.load(e_output_file[0])
            exc = np.load(exc_file[0])
            tau_rho_wf = np.load(tau_rho_wf_file[0])
            ene_nuc = np.load(ene_nuc_file[0])
            rho_output = np.load(rho_output_file[0])
            weight = np.load(weight_file[0])

            for i in range(rho_input.shape[0]):
                data_name = f"{data_path.parts[-1]}-{child.parts[-1]}-{i}.npy"

                if "weit" in args.name:
                    image = rho_input * weight
                else:
                    image = rho_input.copy()

                masks = np.zeros((2, image.shape[1], image.shape[2]))
                masks_e = (exc + tau_rho_wf) / (rho_output + 1e-14)
                masks_rho = rho_output / (rho_input + 1e-14)
                masks[0, :, :] = masks_rho[i, :, :]
                masks[1, :, :] = masks_e[i, :, :]
                data_shape = (1, image.shape[1], image.shape[2])
                np.save(imgs_path / data_name, rho_input[i, :, :].reshape(data_shape))
                np.save(mask_path / data_name, masks)

                print(
                    f"""{child.parts[-1]} max of imag {np.max(rho_input[i, :, :]):.3e} """
                    f"""min of imag {np.min(rho_input[i, :, :]):.3e}"""
                )
                print(
                    f"""{child.parts[-1]} max of masks_e {np.max(masks_e[i, :, :]):.3e} """
                    f"""min of masks_e {np.min(masks_e[i, :, :]):.3e}"""
                )
                print(
                    f"""{child.parts[-1]} max of masks_rho {np.max(masks_rho[i, :, :]):.3e} """
                    f"""min of masks_rho {np.min(masks_rho[i, :, :]):.3e}"""
                )
        else:
            MESSAGE += f"""{child.parts[-1]} not found\n"""
print("\n")
print(MESSAGE)
