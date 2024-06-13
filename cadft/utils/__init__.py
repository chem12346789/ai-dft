import pandas as pd
import numpy as np

from cadft.utils.basis import gen_basis
from cadft.utils.rotate import rotate
from cadft.utils.parser import add_args
from cadft.utils.gen_tau import gen_taup_rho, gen_tau_rho

from cadft.utils.BasicDataset import BasicDataset
from cadft.utils.DataBase import DataBase, gen_logger
from cadft.utils.model.fc_net import FCNet
from cadft.utils.model.transformer import Transformer
from cadft.utils.Grids import Grid
from cadft.utils.ModelDict import ModelDict

from cadft.utils.mol import Mol
from cadft.utils.DataBase import MIDDLE_SCALE, OUTPUT_SCALE


def save_csv_loss(name_list, path, loss_rho, loss_ene, loss_ene_train):
    """
    save the loss to a csv file
    """
    df = pd.DataFrame(
        {
            "name": name_list,
            "loss_rho": loss_rho,
            "loss_ene": loss_ene,
            "loss_ene_train": loss_ene_train,
        }
    )
    df.to_csv(path, index=False)


NAO = {
    "H": 5,
    "C": 14,
}
