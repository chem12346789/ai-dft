"""
Generate list of model.
"""

from pathlib import Path
import datetime

import torch
import torch.optim as optim

from cadft.utils.model.fc_net import FCNet as Model

# from cadft.utils.model.transformer import Transformer as Model


class ModelDict:
    """
    Model_Dict
    """

    def __init__(self, hidden_size, num_layers, residual, device):
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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.residual = residual
        self.device = device

        self.model_dict = {}
        self.model_dict["size"] = {}
        self.optimizer_dict = {}
        self.scheduler_dict = {}
        self.dir_checkpoint = Path(
            f"checkpoints/checkpoint-ccdft-{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}-{self.hidden_size}/"
        ).resolve()
        self.dir_checkpoint.mkdir(parents=True, exist_ok=True)

        self.model_dict["1"] = Model(
            302, self.hidden_size, 302, self.residual, self.num_layers
        ).to(device)
        self.model_dict["1"].double()
        self.model_dict["2"] = Model(
            302, self.hidden_size, 302, self.residual, self.num_layers
        ).to(device)
        self.model_dict["2"].double()

        self.optimizer_dict["1"] = optim.Adam(
            self.model_dict["1"].parameters(),
            lr=1e-5,
        )
        self.scheduler_dict["1"] = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dict["1"],
            T_max=2500,
        )

        self.optimizer_dict["2"] = optim.Adam(
            self.model_dict["2"].parameters(),
            lr=1e-6,
        )
        self.scheduler_dict["2"] = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_dict["2"],
            T_max=2500,
        )

        self.loss_fn = torch.nn.L1Loss(reduction="sum")

    def load_model(self, load):
        """
        Load the model from the checkpoint.
        """
        if load != "":
            dir_load = Path(f"checkpoints/checkpoint-ccdft-{load}-{self.hidden_size}/")
            for i_str in ["1", "2"]:
                list_of_path = list(dir_load.glob(f"{i_str}*.pth"))
                if len(list_of_path) == 0:
                    print(f"No model found for {i_str}, use random initialization.")
                    continue
                load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
                state_dict = torch.load(load_path, map_location=self.device)
                self.model_dict[i_str].load_state_dict(state_dict)
                print(f"Model loaded from {load_path}")

    def save_model(self, epoch):
        """
        Save the model to the checkpoint.
        """
        for i_str in ["1", "2"]:
            state_dict = self.model_dict[i_str].state_dict()
            torch.save(
                state_dict,
                self.dir_checkpoint / f"{i_str}-{epoch}.pth",
            )

    def train_model(self, database_train):
        """
        Train the model, one epoch.
        """
        train_loss_1 = []
        train_loss_2 = []

        for key in ["1", "2"]:
            self.model_dict[key].train(True)
            self.optimizer_dict[key].zero_grad(set_to_none=True)

        for name in database_train.name_list:
            loss_1, loss_2 = 0, database_train.ene[name]
            for batch in database_train.data_gpu[name]:
                input_mat = batch["input"]
                middle_mat_real = batch["middle"]
                weight = batch["weight"]

                middle_mat = self.model_dict["1"](input_mat)
                loss_1 += self.loss_fn(middle_mat * weight, middle_mat_real * weight)
                out_mat = self.model_dict["2"](input_mat)
                loss_2 -= torch.sum(out_mat * weight)

            loss_2 = torch.abs(loss_2)
            loss_1.backward()
            loss_2.backward()
            self.optimizer_dict["1"].step()
            self.optimizer_dict["2"].step()

            train_loss_1.append(loss_1.item())
            train_loss_2.append(loss_2.item())

        self.scheduler_dict["1"].step()
        self.scheduler_dict["2"].step()
        return train_loss_1, train_loss_2

    def eval_model(self, database_train):
        """
        Evaluate the model.
        """
        eval_loss_1 = []
        eval_loss_2 = []

        for key in ["1", "2"]:
            self.model_dict[key].eval(True)

        for name in database_train.name_list:
            loss_1, loss_2 = 0, database_train.ene[name]
            for batch in database_train.data_gpu[name]:
                input_mat = batch["input"]
                middle_mat_real = batch["middle"]

                with torch.no_grad():
                    middle_mat = self.model_dict["1"](input_mat)
                    loss_1 += self.loss_fn(middle_mat, middle_mat_real)
                    out_mat = self.model_dict["2"](input_mat)
                    loss_2 -= torch.sum(out_mat * weight)

            loss_2 = torch.abs(loss_2)

            eval_loss_1.append(loss_1.item())
            eval_loss_2.append(loss_2.item())

        return eval_loss_1, eval_loss_2
