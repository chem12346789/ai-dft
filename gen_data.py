import numpy as np
from pathlib import Path
import argparse


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
    "--method",
    "-m",
    type=str,
    default="mrks",
    choices=["wy", "mrks"],
    help="Witch method we used to obtain the potential.",
)

parser.add_argument(
    "--energy",
    "-e",
    type=bool,
    default=False,
    help="If contain energy.",
)

parser.add_argument(
    "--data",
    "-d",
    nargs="+",
    type=str,
    default="data",
    help='Name of data. Could be a list. Default is "data".',
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
masks_path = main_dir / args.name / "data" / "masks"

imgs_path.mkdir(parents=True, exist_ok=True)
masks_path.mkdir(parents=True, exist_ok=True)
clean_dir(imgs_path)
clean_dir(masks_path)
print(f"data: {args.data}")

error_message = ""
method_str = args.method

for data_i in args.data:
    data_path = Path("./") / data_i
    for child in (data_path).glob("*"):
        if not child.is_dir():
            continue
        print(f"Processing {child.parts[-1]}")
        data_file = list(child.glob(f"rho_t_{method_str}.npy"))
        masks_v_file = list(child.glob(f"{method_str}.npy"))

        if (len(data_file) == 1) and (len(masks_v_file) == 1):
            data = np.load(data_file[0])
            masks_v = np.load(masks_v_file[0])
            for i in range(data.shape[0]):
                data_name = f"{data_path.parts[-1]}-{child.parts[-1]}-{i}.npy"
                data_shape = (1, data.shape[1], data.shape[2])
                masks_ve = np.zeros((1, data.shape[1], data.shape[2]))
                masks_ve[0, :, :] = masks_v[i, :, :].reshape(data_shape)

                if args.energy:
                    if args.nclass == 1:
                        masks_e_file = list(child.glob(f"{method_str}_e_dm.npy"))
                        if len(masks_e_file) == 1:
                            masks_e = np.load(masks_e_file[0])
                            masks_ve[0, :, :] = masks_e[i, :, :].reshape(data_shape)
                    elif args.nclass == 2:
                        masks_ve = np.zeros((2, data.shape[1], data.shape[2]))
                        masks_e_file = list(child.glob(f"{method_str}_e_dm.npy"))
                        if len(masks_e_file) == 1:
                            masks_e = np.load(masks_e_file[0])
                            masks_ve[0, :, :] = masks_v[i, :, :].reshape(data_shape)
                            masks_ve[1, :, :] = masks_e[i, :, :].reshape(data_shape)

                np.save(imgs_path / data_name, data[i, :, :].reshape(data_shape))
                np.save(masks_path / data_name, masks_ve)

            if args.energy:
                print(
                    f"""{child.parts[-1]} max of masks_e {np.max(masks_e):.3e} """
                    f"""min of masks {np.min(masks_e):.3e}"""
                )
            print(
                f"""{child.parts[-1]} max of masks_v {np.max(masks_v):.3e} """
                f"""min of masks {np.min(masks_v):.3e}"""
            )
        else:
            error_message += f"""{child.parts[-1]} not found\n"""
print("\n")
print(error_message)
