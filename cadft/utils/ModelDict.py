"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import pyscf.grad
import torch
import torch.optim as optim
import pyscf

from cadft.utils.model.unet import UNet
from cadft.utils.model.cnn3d import CNN3D
from cadft.utils.env_var import CHECKPOINTS_PATH
from cadft.utils.DataBase import process_input
from cadft.utils.Grids import Grid

# from cadft.utils.model.fc_net import FCNet as Model
# from cadft.utils.model.transformer import Transformer as Model


def get_input_mat(
    ks: pyscf.dft.rks.RKS,
    grids: Grid,
    dms: np.ndarray = None,
    xctype: str = "lda",
):
    """
    Get the input matrix for the model.
    Input:
    ks: the dft instance, RKS/UKS object; See https://pyscf.org/_modules/pyscf/dft/rks.html
    grids: the grids instance, Grids object; See https://pyscf.org/_modules/pyscf/dft/numint.html and the modified version in dft2cc/utils/Grids.py
    """
    if isinstance(ks, pyscf.dft.rks.RKS) or isinstance(ks, pyscf.grad.rks.Gradients):
        if xctype.lower() == "lda":
            if not hasattr(ks, "ao_value"):
                ks.ao_value = pyscf.dft.numint.eval_ao(ks.mol, grids.coords, deriv=1)
            scf_rho_r = pyscf.dft.numint.eval_rho(
                ks.mol, ks.ao_value[0, :], dms, xctype="LDA"
            )
            input_mat = grids.vector_to_matrix(scf_rho_r)
            input_mat = input_mat[:, np.newaxis, :, :]
            return scf_rho_r, input_mat
        elif xctype.lower() == "gga":
            if not hasattr(ks, "ao_value"):
                ks.ao_value = pyscf.dft.numint.eval_ao(ks.mol, grids.coords, deriv=1)
            scf_rho_r_3 = pyscf.dft.numint.eval_rho(
                ks.mol, ks.ao_value, dms, xctype="GGA"
            )
            input_mat = process_input(scf_rho_r_3, grids)
            input_mat = np.transpose(input_mat, (1, 0, 2, 3))
            return scf_rho_r_3[0, :], input_mat


class ModelDict:
    """
    Model_Dict
    """

    def __init__(
        self,
        load,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        residual,
        device,
        precision,
        with_eval=True,
        ene_weight=0.0,
        pot_weight=0.1,
        if_mkdir=True,
        load_epoch=-1,
    ):
        """
        input:
            hidden_size: number of hidden units
            num_layers: number of layers in the fully connected network
            residual: whether to use residual connection, "0", "1" or "2".
                "0" for no residual connection
                "1" for residual connection in the last layer
                "2" for residual connection in all the last layers
            device: device to run the model
        output:
            model_dict: dictionary of models
        """
        self.load = load
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.residual = residual
        self.ene_weight = ene_weight
        self.pot_weight = pot_weight
        self.with_eval = with_eval
        self.load_epoch = load_epoch

        self.device = device
        if precision == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

        self.dir_checkpoint = Path(
            CHECKPOINTS_PATH
            / f"checkpoint-ccdft_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}_{self.input_size}_{self.hidden_size}_{self.output_size}_{self.num_layers}_{self.residual}/"
        ).resolve()
        if if_mkdir:
            print(f"Create checkpoint directory: {self.dir_checkpoint}")
            self.dir_checkpoint.mkdir(parents=True, exist_ok=True)
            (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

        self.keys = []
        self.model_dict = {}
        self.model_dict["size"] = {}
        self.optimizer_dict = {}
        self.scheduler_dict = {}

        self.loss_multiplier = 1.0
        # self.loss_fn1 = torch.nn.MSELoss()
        # self.loss_fn2 = torch.nn.MSELoss()
        # self.loss_fn3 = torch.nn.MSELoss(reduction="sum")

        self.loss_fn1 = torch.nn.L1Loss()
        self.loss_fn2 = torch.nn.L1Loss()
        self.loss_fn3 = torch.nn.L1Loss(reduction="sum")

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        if self.load not in ["", "None", "NEW", "new"]:
            load_checkpoint = Path(
                CHECKPOINTS_PATH
                / f"checkpoint-ccdft_{self.load}_{self.input_size}_{self.hidden_size}_{self.output_size}_{self.num_layers}_{self.residual}/"
            ).resolve()
            if load_checkpoint.exists():
                print(f"Loading from {load_checkpoint}")
                for key in self.keys:
                    list_of_path = list(load_checkpoint.glob(f"{key}-*.pth"))
                    if len(list_of_path) == 0:
                        print(f"No model found for {key}, use random initialization.")
                        continue
                    load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                    if self.load_epoch != -1:
                        load_path = load_checkpoint / f"{key}-{self.load_epoch}.pth"
                    state_dict = torch.load(
                        load_path, map_location=self.device, weights_only=True
                    )
                    self.model_dict[key].load_state_dict(state_dict)
                    print(f"Model loaded from {load_path}")
            else:
                print(f"Load checkpoint directory {load_checkpoint} not found.")

    def train(self):
        """
        Set the model to train mode.
        """
        for key in self.keys:
            self.model_dict[key].train(True)
            self.optimizer_dict[key].zero_grad(set_to_none=True)

    def zero_grad(self):
        """
        Set the model to train mode.
        """
        for key in self.keys:
            self.optimizer_dict[key].zero_grad(set_to_none=True)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        for key in self.keys:
            self.model_dict[key].eval()
            self.optimizer_dict[key].zero_grad(set_to_none=True)

    def step(self):
        """
        Step the optimizer.
        """
        for key in self.keys:
            self.optimizer_dict[key].step()

    def save_model(self, epoch):
        """
        Save the model to the checkpoint.
        """
        for key in self.keys:
            state_dict = self.model_dict[key].state_dict()
            torch.save(
                state_dict,
                self.dir_checkpoint / f"{key}-{epoch}.pth",
            )


class ModelDictUnet(ModelDict):
    """
    Model_Dict for unet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.output_size == 1:
            self.keys = ["1", "2"]
        elif self.output_size == 2:
            self.keys = ["1"]
        elif self.output_size == -1:
            self.keys = ["1"]
        elif self.output_size == -2:
            self.keys = ["1"]

        for i_key, key in enumerate(self.keys):
            self.model_dict[key] = UNet(
                self.input_size,
                self.hidden_size,
                self.output_size if self.output_size > 0 else 1,
                (
                    int(self.residual)
                    if "." not in self.residual
                    else int(self.residual.split(".")[i_key])
                ),
                self.num_layers,
            ).to(self.device)

        for key in self.keys:
            if self.dtype is torch.float64:
                self.model_dict[key].double()

            self.optimizer_dict[key] = optim.Adam(
                self.model_dict[key].parameters(),
                lr=1e-4,
            )

        for key in self.keys:
            if self.with_eval:
                self.scheduler_dict[key] = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_dict[key],
                    mode="min",
                    factor=0.5,
                    patience=10,
                )
            else:
                self.scheduler_dict[key] = optim.lr_scheduler.ExponentialLR(
                    self.optimizer_dict[key],
                    gamma=1.0,
                )

    def loss(self, batch):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        middle_mat_real = batch["middle"]
        output_mat_real = batch["output"]
        weight = batch["weight"]
        tot_correct_energy = batch["tot_correct_energy"]

        if self.output_size == 1:
            middle_mat = self.model_dict["1"](input_mat)
            loss_pot_i = self.loss_multiplier * self.loss_fn1(
                middle_mat,
                middle_mat_real,
            )

            output_mat = self.model_dict["2"](input_mat)
            loss_ene_i = self.loss_multiplier * self.loss_fn2(
                output_mat,
                output_mat_real,
            )
            loss_ene_tot_i = self.loss_multiplier * self.loss_fn3(
                tot_correct_energy,
                torch.sum(output_mat * input_mat[:, [0], :, :] * weight),
            )
        elif self.output_size == 2:
            output_mat = self.model_dict["1"](input_mat)
            loss_pot_i = self.loss_multiplier * self.loss_fn1(
                output_mat[:, [0], :, :],
                output_mat_real[:, [0], :, :],
            )

            loss_ene_i = self.loss_multiplier * self.loss_fn2(
                output_mat[:, [1], :, :],
                output_mat_real[:, [1], :, :],
            )
            loss_ene_tot_i = self.loss_multiplier * self.loss_fn3(
                tot_correct_energy,
                torch.sum(output_mat[:, [1], :, :] * input_mat[:, [0], :, :] * weight),
            )
        elif self.output_size == -1:
            input_mat = input_mat.requires_grad_(True)
            output_mat = self.model_dict["1"](input_mat)
            loss_ene_i = self.loss_multiplier * self.loss_fn2(
                output_mat,
                output_mat_real,
            )
            loss_ene_tot_i = self.loss_multiplier * self.loss_fn3(
                tot_correct_energy,
                torch.sum(output_mat * input_mat[:, [0], :, :] * weight),
            )

            middle_mat = torch.autograd.grad(
                torch.sum(input_mat[:, [0], :, :] * output_mat),
                input_mat,
                create_graph=True,
            )[0]
            loss_pot_i = self.loss_multiplier * self.loss_fn1(
                middle_mat,
                middle_mat_real,
            )
        elif self.output_size == -2:
            output_mat = self.model_dict["1"](input_mat)
            loss_pot_i = torch.tensor([0.0], device=self.device)
            loss_ene_i = self.loss_multiplier * self.loss_fn2(
                output_mat,
                output_mat_real,
            )
            loss_ene_tot_i = self.loss_multiplier * self.loss_fn3(
                tot_correct_energy,
                torch.sum(output_mat * input_mat[:, [0], :, :] * weight),
            )
        return loss_pot_i, loss_ene_i, loss_ene_tot_i

    def train_model(self, database_train):
        """
        Train the model, one epoch.
        """
        train_loss_pot, train_loss_ene, train_loss_ene_tot = [], [], []
        database_train.rng.shuffle(database_train.name_list)
        self.train()

        for name in database_train.name_list:
            for batch in database_train.data_gpu[name]:
                self.zero_grad()
                loss_pot_i, loss_ene_i, loss_ene_tot_i = self.loss(batch)

                train_loss_pot.append(loss_pot_i.item())
                train_loss_ene.append(loss_ene_i.item())
                train_loss_ene_tot.append(loss_ene_tot_i.item())

                if self.output_size == 1:
                    loss_pot_i.backward()
                    (loss_ene_i + self.ene_weight * loss_ene_tot_i).backward()
                elif self.output_size == 2:
                    (
                        loss_pot_i + loss_ene_i + self.ene_weight * loss_ene_tot_i
                    ).backward()
                elif self.output_size == -1:
                    loss_pot_i.backward(retain_graph=True)
                    (loss_ene_i + self.ene_weight * loss_ene_tot_i).backward()
                elif self.output_size == -2:
                    (loss_ene_i + self.ene_weight * loss_ene_tot_i).backward()

                self.step()

        return (
            np.array(train_loss_pot),
            np.array(train_loss_ene),
            np.array(train_loss_ene_tot),
        )

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()
        eval_loss_pot, eval_loss_ene, eval_loss_ene_tot = [], [], []

        for name in database_eval.name_list:
            for batch in database_eval.data_gpu[name]:
                loss_pot_i, loss_ene_i, loss_ene_tot_i = self.loss(batch)

                eval_loss_pot.append(loss_pot_i.item())
                eval_loss_ene.append(loss_ene_i.item())
                eval_loss_ene_tot.append(loss_ene_tot_i.item())

        return (
            np.array(eval_loss_pot),
            np.array(eval_loss_ene),
            np.array(eval_loss_ene_tot),
        )

    def get_v(
        self,
        ks: pyscf.dft.rks.RKS,
        grids: Grid,
        dms: np.ndarray = None,
    ):
        """
        Obtain the potential.
        Input: [rho, nabla rho] (4, ngrids),
        Output: the potential (ngrids).
        """
        if dms is None:
            dms = ks.make_rdm1()

        if self.input_size == 1:
            _, input_mat = get_input_mat(ks, grids, dms, "lda")
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        elif self.input_size == 4:
            _, input_mat = get_input_mat(ks, grids, dms, "gga")
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        else:
            raise ValueError("input_size must be 1 or 4")

        if self.output_size == 1 or self.output_size == 2:
            with torch.no_grad():
                middle_mat = self.model_dict["1"](input_mat).detach().cpu().numpy()
            middle_mat = middle_mat[:, 0, :, :]
        elif self.output_size == -1 or self.output_size == -2:
            input_mat = input_mat.requires_grad_(True)
            with torch.no_grad():
                output_mat = self.model_dict["1"](input_mat)
            middle_mat = torch.autograd.grad(
                torch.sum(input_mat[:, [0], :, :] * output_mat),
                input_mat,
                create_graph=True,
            )[0]
            middle_mat = middle_mat.detach().cpu().numpy()[:, 0, :, :]

        return grids.matrix_to_vector(middle_mat)

    def get_e(
        self,
        ks: pyscf.dft.rks.RKS,
        grids: Grid,
        dms: np.ndarray = None,
    ):
        """
        Obtain the energy density.
        Input: [rho, nabla rho] (4, ngrids),
        Output: the potential (ngrids).
        """
        if dms is None:
            dms = ks.make_rdm1()

        if self.input_size == 1:
            scf_rho_r, input_mat = get_input_mat(ks, grids, dms, "lda")
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        elif self.input_size == 4:
            scf_rho_r, input_mat = get_input_mat(ks, grids, dms, "gga")
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        else:
            raise ValueError("input_size must be 1 or 4")

        if self.output_size == 1:
            with torch.no_grad():
                output_mat = self.model_dict["2"](input_mat).detach().cpu().numpy()
            output_mat = output_mat[:, 0, :, :]
        elif self.output_size == 2:
            with torch.no_grad():
                output_mat = self.model_dict["1"](input_mat).detach().cpu().numpy()
            output_mat = output_mat[:, 1, :, :]
        elif self.output_size == -1 or self.output_size == -2:
            input_mat = input_mat.requires_grad_(True)
            with torch.no_grad():
                output_mat = self.model_dict["1"](input_mat).detach().cpu().numpy()
            output_mat = output_mat[:, 0, :, :]

        exc_scf = grids.matrix_to_vector(output_mat)
        return np.sum(exc_scf * scf_rho_r * grids.weights)


class ModelDict3DCNN(ModelDict):
    """
    Model_Dict for unet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.output_size == -1:
            self.keys = ["1"]

        for i_key, key in enumerate(self.keys):
            self.model_dict[key] = CNN3D(
                self.input_size,
                self.hidden_size,
                self.output_size if self.output_size > 0 else 1,
                (
                    int(self.residual)
                    if "." not in self.residual
                    else int(self.residual.split(".")[i_key])
                ),
                self.num_layers,
            ).to(self.device)

        for key in self.keys:
            if self.dtype is torch.float64:
                self.model_dict[key].double()

            self.optimizer_dict[key] = optim.Adam(
                self.model_dict[key].parameters(),
                lr=1e-4,
            )

        for key in self.keys:
            if self.with_eval:
                self.scheduler_dict[key] = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer_dict[key],
                    mode="min",
                    factor=0.5,
                    patience=10,
                )
            else:
                self.scheduler_dict[key] = optim.lr_scheduler.ExponentialLR(
                    self.optimizer_dict[key],
                    gamma=1.0,
                )

    def loss(self, batch):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        middle_mat_real = batch["middle"]
        output_mat_real = batch["output"]
        weight = batch["weight"]
        tot_correct_energy = batch["tot_correct_energy"]

        if self.output_size == -1:
            input_mat = input_mat.requires_grad_(True)
            output_mat = self.model_dict["1"](input_mat)
            loss_ene_i = self.loss_multiplier * self.loss_fn2(
                output_mat,
                output_mat_real,
            )
            loss_ene_tot_i = self.loss_multiplier * self.loss_fn3(
                tot_correct_energy,
                torch.sum(output_mat),
            )

            middle_mat = torch.autograd.grad(
                torch.sum(output_mat),
                input_mat,
                create_graph=True,
            )[0]
            loss_pot_i = self.loss_multiplier * self.loss_fn1(
                middle_mat,
                middle_mat_real,
            )
        return loss_pot_i, loss_ene_i, loss_ene_tot_i

    def train_model(self, database_train):
        """
        Train the model, one epoch.
        """
        train_loss_pot, train_loss_ene, train_loss_ene_tot = [], [], []
        database_train.rng.shuffle(database_train.name_list)
        self.train()

        for name in database_train.name_list:
            for batch in database_train.data_gpu[name]:
                self.zero_grad()
                loss_pot_i, loss_ene_i, loss_ene_tot_i = self.loss(batch)

                train_loss_pot.append(loss_pot_i.item())
                train_loss_ene.append(loss_ene_i.item())
                train_loss_ene_tot.append(loss_ene_tot_i.item())

                if self.output_size == 1:
                    loss_pot_i.backward()
                    (loss_ene_i + self.ene_weight * loss_ene_tot_i).backward()
                elif self.output_size == 2:
                    (
                        loss_pot_i + loss_ene_i + self.ene_weight * loss_ene_tot_i
                    ).backward()
                elif self.output_size == -1:
                    loss_pot_i.backward(retain_graph=True)
                    (loss_ene_i + self.ene_weight * loss_ene_tot_i).backward()
                elif self.output_size == -2:
                    (loss_ene_i + self.ene_weight * loss_ene_tot_i).backward()

                self.step()

        return (
            np.array(train_loss_pot),
            np.array(train_loss_ene),
            np.array(train_loss_ene_tot),
        )

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()
        eval_loss_pot, eval_loss_ene, eval_loss_ene_tot = [], [], []

        for name in database_eval.name_list:
            for batch in database_eval.data_gpu[name]:
                loss_pot_i, loss_ene_i, loss_ene_tot_i = self.loss(batch)

                eval_loss_pot.append(loss_pot_i.item())
                eval_loss_ene.append(loss_ene_i.item())
                eval_loss_ene_tot.append(loss_ene_tot_i.item())

        return (
            np.array(eval_loss_pot),
            np.array(eval_loss_ene),
            np.array(eval_loss_ene_tot),
        )

    def get_v(
        self,
        ks: pyscf.dft.rks.RKS,
        grids: Grid,
        dms: np.ndarray = None,
    ):
        """
        Obtain the potential.
        Input:
            ks: the dft instance, RKS/UKS object; See https://pyscf.org/_modules/pyscf/dft/rks.html
            grids: the grids instance, Grids object; See https://pyscf.org/_modules/pyscf/dft/numint.html and the modified version in dft2cc/utils/Grids.py
            dms: the density matrix (nspin, nao, nao), np.ndarray
        Output: the potential (ngrids).
        """
        if dms is None:
            dms = ks.make_rdm1()

        if self.input_size == 1:
            _, input_mat = get_input_mat(ks, grids, dms, "LDA")
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        elif self.input_size == 4:
            _, input_mat = get_input_mat(ks, grids, dms, "GGA")
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        else:
            raise ValueError("input_size must be 1 or 4")

        if self.output_size == -1:
            input_mat = input_mat.requires_grad_(True)
            with torch.no_grad():
                output_mat = self.model_dict["1"](input_mat)
            middle_mat = torch.autograd.grad(
                torch.sum(output_mat),
                input_mat,
                create_graph=True,
            )[0]
            middle_mat = middle_mat.detach().cpu().numpy()[:, 0, :, :]

        return grids.matrix_to_vector(middle_mat)

    def get_e(
        self,
        ks: pyscf.dft.rks.RKS,
        grids: Grid,
        dms: np.ndarray = None,
    ):
        """
        Obtain the energy density.
        Input:
            ks: the dft instance, RKS/UKS object; See https://pyscf.org/_modules/pyscf/dft/rks.html
            grids: the grids instance, Grids object; See https://pyscf.org/_modules/pyscf/dft/numint.html and the modified version in dft2cc/utils/Grids.py
            dms: the density matrix (nspin, nao, nao), np.ndarray
        Output: the potential (ngrids).
        """
        if dms is None:
            dms = ks.make_rdm1()

        if self.input_size == 1:
            _, input_mat = get_input_mat(ks, grids, dms, "LDA")
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        elif self.input_size == 4:
            _, input_mat = get_input_mat(ks, grids, dms, "GGA")
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        else:
            raise ValueError("input_size must be 1 or 4")

        if self.output_size == 1:
            with torch.no_grad():
                output_mat = self.model_dict["1"](input_mat).detach().cpu().numpy()

        return np.sum(output_mat)
