import pandas as pd
import numpy as np

from cadft.utils.basis import gen_basis
from cadft.utils.rotate import rotate
from cadft.utils.parser import add_args

from cadft.utils.BasicDataset import BasicDataset
from cadft.utils.DataBase import DataBase
from cadft.utils.model.fc_net import FCNet
from cadft.utils.model.transformer import Transformer
from cadft.utils.Grids import Grid
from cadft.utils.aux_train import ModelDict

from cadft.utils.mol import Mol
from cadft.utils.DataBase import MIDDLE_SCALE, OUTPUT_SCALE


def save_csv_loss(name_list, loss_rho, loss_ene, path):
    """
    save the loss to a csv file
    """
    df = pd.DataFrame(
        {
            "name": name_list,
            "loss_rho": loss_rho,
            "loss_ene": loss_ene,
        }
    )
    df.to_csv(path, index=False)


def gen_logger(distance_list):
    """
    Function to distance list and generate logger
    """
    if len(distance_list) == 3:
        distance_l = np.linspace(
            distance_list[0], distance_list[1], int(distance_list[2])
        )
    else:
        distance_l = distance_list
    return distance_l


NAO = {
    "H": 5,
    "C": 14,
}

MASS = {
    "H": 1.00782503207,
    "He": 4.00260325415,
    "Li": 6.938,
    "Be": 9.012183065,
    "B": 10.806,
    "C": 12.0096,
    "N": 14.006855,
    "O": 15.9994,
    "F": 18.998403163,
    "Ne": 20.1797,
}
