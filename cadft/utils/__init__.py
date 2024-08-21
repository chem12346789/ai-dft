import pandas as pd
import copy

from cadft.utils.basis import gen_basis
from cadft.utils.rotate import rotate
from cadft.utils.parser import add_args
from cadft.utils.mrks import mrks, mrks_append
from cadft.utils.umrks import umrks
from cadft.utils.mrks_diis import mrks_diis
from cadft.utils.umrks_diis import umrks_diis
from cadft.utils.gmrks_diis import gmrks_diis
from cadft.utils.DataBase import gen_logger, process_input

from cadft.utils.DataBase import DataBase
from cadft.utils.model.fc_net import FCNet
from cadft.utils.model.transformer import Transformer
from cadft.utils.Grids import Grid
from cadft.utils.ModelDict import ModelDict
from cadft.utils.ModelDict_xy import ModelDict_xy
from cadft.utils.diis import DIIS

from cadft.utils.mol import Mol
from cadft.utils.env_var import MAIN_PATH, DATA_PATH, DATA_SAVE_PATH, DATA_CC_PATH


def save_csv_loss(
    name_list,
    path,
    loss_rho=0.0,
    loss_tot_rho=0.0,
    loss_ene=0.0,
    loss_tot_ene=0.0,
):
    """
    save the loss to a csv file
    """
    df = pd.DataFrame(
        {
            "name": name_list,
            "loss_rho": loss_rho,
            "loss_ene": loss_ene,
            "loss_tot_rho": loss_tot_rho,
            "loss_tot_ene": loss_tot_ene,
        }
    )
    df.to_csv(path, index=False)


NAO = {
    "H": 5,
    "C": 14,
}


def extend(
    name_mol: str,
    extend_atom: str,
    extend_xyz: int,
    distance: float,
    basis: str,
) -> tuple:
    molecular = copy.deepcopy(Mol[name_mol])
    print(f"Generate {name_mol}_{distance:.4f}")

    print(f"Extend {extend_atom} {extend_xyz} {distance:.4f}")
    name = f"{name_mol}_{basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"

    if "-" in extend_atom:
        extend_atom_1, extend_atom_2 = map(int, extend_atom.split("-"))
        if extend_atom_1 >= len(Mol[name_mol]) or extend_atom_2 >= len(Mol[name_mol]):
            print(f"Skip: {name:>40}")
            return None, name
        if abs(distance) < 1e-3:
            if (extend_atom_1 != 0) and (extend_atom_2 != 1):
                print(f"Skip: {name:>40}")
                return None, name
        distance_1_2_array = [
            molecular[extend_atom_2][1] - molecular[extend_atom_1][1],
            molecular[extend_atom_2][2] - molecular[extend_atom_1][2],
            molecular[extend_atom_2][3] - molecular[extend_atom_1][3],
        ]
        distance_1_2 = sum(map(lambda x: x**2, distance_1_2_array)) ** 0.5
        for i in range(1, 4):
            molecular[extend_atom_2][i] += (
                distance * distance_1_2_array[i - 1] / distance_1_2
            )
    else:
        extend_atom = int(extend_atom)
        if abs(distance) < 1e-3:
            if (extend_atom != 0) or extend_xyz != 1:
                print(f"Skip: {name:>40}")
                return None, name
        if extend_atom >= len(Mol[name_mol]):
            print(f"Skip: {name:>40}")
            return None, name
        molecular[extend_atom][extend_xyz] += distance

    return molecular, name
