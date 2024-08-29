"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import torch
import torch.optim as optim

from cadft.utils.env_var import CHECKPOINTS_PATH
from cadft.utils_deepks.DataBase import process_input

from cadft.utils_deepks.model.fc_net import FCNet as Model

# from cadft.utils.model.transformer import Transformer as Model


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

        self.model = Model(
            302,
            self.hidden_size,
            self.output_size,
            self.residual,
            self.num_layers,
        ).to(device)

        if precision == "float64":
            self.model.double()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-4,
        )

        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=1,
        )

        self.loss_multiplier = 1.0
        # self.loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.L1Loss()

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
                list_of_path = list(load_checkpoint.glob("*.pth"))
                if len(list_of_path) == 0:
                    print("No model found, use random initialization.")
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                if self.load_epoch != -1:
                    load_path = load_checkpoint / f"{self.load_epoch}.pth"
                state_dict = torch.load(
                    load_path, map_location=self.device, weights_only=True
                )
                self.model.load_state_dict(state_dict)
                print(f"Model loaded from {load_path}")
            else:
                print(f"Load checkpoint directory {load_checkpoint} not found.")

    def train(self):
        """
        Set the model to train mode.
        """
        self.model.train(True)

    def zero_grad(self):
        """
        Set the model to train mode.
        """
        self.optimizer.zero_grad(set_to_none=True)

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()

    def step(self):
        """
        Step the optimizer.
        """
        self.optimizer.step()

    def loss(self, batch):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        output_real = batch["output"]

        output_mat = self.model(input_mat)
        loss = self.loss_multiplier * self.loss_fn(
            torch.sum(output_mat),
            torch.sum(output_real),
        )
        return loss

    def save_model(self, epoch):
        """
        Save the model to the checkpoint.
        """
        state_dict = self.model.state_dict()
        torch.save(state_dict, self.dir_checkpoint / f"{epoch}.pth")

    def train_model(self, database_train):
        """
        Train the model, one epoch.
        """
        train_loss = []
        # database_train.rng.shuffle(database_train.name_list)
        self.train()

        for name in database_train.name_list:
            loss = torch.tensor([0.0], device=self.device)
            self.zero_grad()

            for batch in database_train.data_gpu[name]:
                loss_i = self.loss(batch)
                loss += loss_i

            train_loss.append(loss.item())
            loss.backward()
            self.step()

        return train_loss

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()

        eval_loss = []

        for name in database_eval.name_list:
            loss = torch.tensor([0.0], device=self.device)

            for batch in database_eval.data_gpu[name]:
                with torch.no_grad():
                    loss_i = self.loss(batch)
                    loss += loss_i

            eval_loss.append(loss.item())

        return eval_loss

    def get_v(self, scf_r_3, grids):
        """
        Obtain the potential.
        Input: [rho, nabla rho] (4, ngrids),
        Output: the potential (ngrids).
        """
        if self.input_size == 1:
            input_mat = grids.vector_to_matrix(scf_r_3[0, :])
            input_mat = torch.tensor(
                input_mat[:, np.newaxis, :, :], dtype=self.dtype
            ).to("cuda")
        elif self.input_size == 4:
            input_mat = process_input(scf_r_3, grids)
            input_mat = np.transpose(input_mat, (1, 0, 2, 3))
            input_mat = torch.tensor(input_mat, dtype=self.dtype).to("cuda")
        else:
            raise ValueError("input_size must be 1 or 4")
        input_mat = input_mat.requires_grad_(True)
        with torch.no_grad():
            output_mat = self.model(input_mat)
        middle_mat = (
            torch.autograd.grad(
                output_mat,
                input_mat,
                create_graph=True,
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        middle_mat = middle_mat[:, 0, :, :]
        vxc_scf = grids.matrix_to_vector(middle_mat)
        return vxc_scf

    def get_e(self, scf_r_3, grids):
        """
        Obtain the energy density.
        Input: [rho, nabla rho] (4, ngrids),
        Output: the potential (ngrids).
        """
        if self.input_size == 1:
            input_mat = grids.vector_to_matrix(scf_r_3[0, :])
            input_mat = torch.tensor(
                input_mat[:, np.newaxis, :, :], dtype=self.dtype
            ).to("cuda")
        elif self.input_size == 4:
            input_mat = process_input(scf_r_3, grids)
            input_mat = np.transpose(input_mat, (1, 0, 2, 3))
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
        elif self.output_size == -1:
            input_mat = input_mat.requires_grad_(True)
            with torch.no_grad():
                output_mat = self.model_dict["1"](input_mat).detach().cpu().numpy()
            output_mat = output_mat[:, 0, :, :]

        exc_scf = grids.matrix_to_vector(output_mat)
        return exc_scf
