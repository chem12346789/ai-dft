"""
Generate list of model.
"""

from pathlib import Path
import datetime

import torch
import torch.optim as optim

from cadft.utils.model.unet import UNet as Model
from cadft.utils.env_var import CHECKPOINTS_PATH

# from cadft.utils.model.fc_net import FCNet as Model
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
        if_mkdir=True,
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
        self.with_eval = with_eval

        self.device = device
        if precision == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

        self.dir_checkpoint = Path(
            CHECKPOINTS_PATH
            / f"checkpoint-ccdft_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}_{self.hidden_size}_{self.num_layers}_{self.residual}/"
        ).resolve()
        if if_mkdir:
            print(f"Create checkpoint directory: {self.dir_checkpoint}")
            self.dir_checkpoint.mkdir(parents=True, exist_ok=True)
            (self.dir_checkpoint / "loss").mkdir(parents=True, exist_ok=True)

        if self.output_size == 1:
            self.keys = ["1", "2"]
        else:
            self.keys = ["1"]

        self.model_dict = {}
        self.model_dict["size"] = {}
        self.optimizer_dict = {}
        self.scheduler_dict = {}

        for key in self.keys:
            self.model_dict[key] = Model(
                self.input_size,
                self.hidden_size,
                self.output_size,
                self.residual,
                self.num_layers,
            ).to(device)

        for key in self.keys:
            if precision == "float64":
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
                    patience=5,
                    factor=0.5,
                )
            else:
                self.scheduler_dict[key] = optim.lr_scheduler.ExponentialLR(
                    self.optimizer_dict[key],
                    gamma=0.9999,
                )

        self.loss_multiplier = 1.0
        self.loss_fn1 = torch.nn.MSELoss()
        self.loss_fn2 = torch.nn.MSELoss()
        self.loss_fn3 = torch.nn.MSELoss(reduction="sum")

        # self.loss_multiplier = 1.0
        # self.loss_fn1 = torch.nn.L1Loss()
        # self.loss_fn2 = torch.nn.L1Loss()
        # self.loss_fn3 = torch.nn.L1Loss(reduction="sum")

        # self.loss_multiplier = 2.0 * 0.1
        # self.loss_fn1 = torch.nn.SmoothL1Loss(beta=0.1)
        # self.loss_fn2 = torch.nn.SmoothL1Loss(beta=0.1)
        # self.loss_fn3 = torch.nn.SmoothL1Loss(beta=0.1, reduction="sum")

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        if self.load not in ["", "None", "NEW", "new"]:
            load_checkpoint = Path(
                CHECKPOINTS_PATH
                / f"checkpoint-ccdft_{self.load}_{self.hidden_size}_{self.num_layers}_{self.residual}/"
            ).resolve()
            if load_checkpoint.exists():
                print(f"Loading from {load_checkpoint}")
                for key in self.keys:
                    list_of_path = list(load_checkpoint.glob(f"{key}-*.pth"))
                    if len(list_of_path) == 0:
                        print(f"No model found for {key}, use random initialization.")
                        continue
                    load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                    state_dict = torch.load(load_path, map_location=self.device)
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

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        for key in self.keys:
            self.model_dict[key].eval()

    def step(self):
        """
        Step the optimizer.
        """
        for key in self.keys:
            self.optimizer_dict[key].step()

    def loss(self, batch):
        """
        Calculate the loss.
        """
        input_mat = batch["input"]
        middle_mat_real = batch["middle"]
        output_mat_real = batch["output"]
        weight = batch["weight"]

        if self.output_size == 1:
            middle_mat = self.model_dict["1"](input_mat)
            loss_1_i = self.loss_multiplier * self.loss_fn1(middle_mat, middle_mat_real)

            output_mat = self.model_dict["2"](input_mat)
            loss_2_i = self.loss_multiplier * self.loss_fn2(
                output_mat,
                output_mat_real,
            )

            loss_3_i = self.loss_multiplier * self.loss_fn3(
                torch.sum(output_mat_real * input_mat[:, 0, :, :] * weight),
                torch.sum(output_mat * input_mat[:, 0, :, :] * weight),
            )
            return loss_1_i, loss_2_i, loss_3_i
        else:
            output_mat = self.model_dict["1"](input_mat)
            loss_1_i = self.loss_multiplier * self.loss_fn1(
                output_mat,
                output_mat_real,
            )

            loss_3_i = self.loss_multiplier * self.loss_fn2(
                torch.sum(output_mat_real * input_mat[:, 0, :, :] * weight),
                torch.sum(output_mat * input_mat[:, 0, :, :] * weight),
            )
            return loss_1_i, torch.tensor([0.0], device=self.device), loss_3_i

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

    def train_model(self, database_train):
        """
        Train the model, one epoch.
        """
        train_loss_1, train_loss_2, train_loss_3 = [], [], []
        database_train.rng.shuffle(database_train.name_list)
        self.train()

        for name in database_train.name_list:
            (
                loss_1,
                loss_2,
                loss_3,
            ) = (
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
            )

            for batch in database_train.data_gpu[name]:
                loss_1_i, loss_2_i, loss_3_i = self.loss(batch)
                loss_1 += loss_1_i
                loss_2 += loss_2_i
                loss_3 += loss_3_i

            train_loss_1.append(loss_1.item())
            train_loss_2.append(loss_2.item())
            train_loss_3.append(loss_3.item())

            if self.output_size == 1:
                loss_2 += loss_3 * self.ene_weight
                loss_1.backward()
                loss_2.backward()
            else:
                loss_1 += loss_3 * self.ene_weight
                loss_1.backward()

            self.step()

        return train_loss_1, train_loss_2, train_loss_3

    def eval_model(self, database_eval):
        """
        Evaluate the model.
        """
        self.eval()

        eval_loss_1, eval_loss_2, eval_loss_3 = [], [], []

        for name in database_eval.name_list:
            (
                loss_1,
                loss_2,
                loss_3,
            ) = (
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
            )

            for batch in database_eval.data_gpu[name]:
                with torch.no_grad():
                    loss_1_i, loss_2_i, loss_3_i = self.loss(batch)
                    loss_1 += loss_1_i
                    loss_2 += loss_2_i
                    loss_3 += loss_3_i

            eval_loss_1.append(loss_1.item())
            eval_loss_2.append(loss_2.item())
            eval_loss_3.append(loss_3.item())
        return eval_loss_1, eval_loss_2, eval_loss_3
