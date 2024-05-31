import pandas as pd

from cadft.utils.basis import gen_basis
from cadft.utils.rotate import rotate
from cadft.utils.logger import gen_logger
from cadft.utils.parser import add_args

from cadft.utils.BasicDataset import BasicDataset
from cadft.utils.DataBase import DataBase
from cadft.utils.model.fc_net import FCNet
from cadft.utils.model.transformer import Transformer
from cadft.utils.Grids import Grid
from cadft.utils.aux_train import ModelDict

from cadft.utils.mass import MASS
from cadft.utils.mol import Mol
from cadft.utils.nao import NAO
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
