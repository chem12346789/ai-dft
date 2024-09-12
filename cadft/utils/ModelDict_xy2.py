"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from cadft.utils.env_var import CHECKPOINTS_PATH
from cadft.utils.DataBase import process_input


class MLPMixer(nn.Module):
    def __init__(self, **kwargs):
        super(MLPMixer, self).__init__()
        self.hidden_dim = kwargs.get("hidden_dim", 768)
        self.num_blocks = kwargs.get("num_blocks", 12)
        self.tokens_mlp_dim = kwargs.get("tokens_mlp_dim", 384)
        self.channels_mlp_dim = kwargs.get("channels_mlp_dim", 3072)
        self.drop_rate = kwargs.get("drop_rate", 0.1)

        self.layernorm1 = nn.LayerNorm((75, 302, 1))
        self.dense = nn.Linear(1, self.hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(self.drop_rate)
        layers = dict()
        for i in range(self.num_blocks):
            layers.update(
                {
                    "layernorm1_%d" % i: nn.LayerNorm((self.hidden_dim, 302)),
                    "linear1_%d" % i: nn.Linear(302, self.tokens_mlp_dim),
                    "gelu1_%d" % i: nn.GELU(),
                    "linear2_%d" % i: nn.Linear(self.tokens_mlp_dim, 302),
                    "layernorm2_%d" % i: nn.LayerNorm((302, self.hidden_dim)),
                    "linear3_%d" % i: nn.Linear(self.hidden_dim, self.channels_mlp_dim),
                    "gelu2_%d" % i: nn.GELU(),
                    "linear4_%d" % i: nn.Linear(self.channels_mlp_dim, self.hidden_dim),
                    "layernorm3_%d" % i: nn.LayerNorm((self.hidden_dim, 75)),
                    "linear5_%d" % i: nn.Linear(75, self.tokens_mlp_dim),
                    "gelu3_%d" % i: nn.GELU(),
                    "linear6_%d" % i: nn.Linear(self.tokens_mlp_dim, 75),
                    "layernorm4_%d" % i: nn.LayerNorm((75, self.hidden_dim)),
                    "linear7_%d" % i: nn.Linear(self.hidden_dim, self.channels_mlp_dim),
                    "gelu4_%d" % i: nn.GELU(),
                    "linear8_%d" % i: nn.Linear(self.channels_mlp_dim, self.hidden_dim),
                }
            )
        self.layers = nn.ModuleDict(layers)
        self.layernorm2 = nn.LayerNorm((75, 302, self.hidden_dim))
        self.head = nn.Linear(self.hidden_dim, 1)

    def forward(self, inputs):
        batch = inputs.shape[0]
        # inputs.shape = (batch, 75, 302, 1)
        # results = self.layernorm1(inputs)
        results = inputs
        results = self.dense(results)  # results.shape = (batch, 75, 302, hidden_dim)
        results = self.gelu(results)
        results = self.dropout(results)

        for i in range(self.num_blocks):
            # merge dimension
            results = torch.reshape(results, (batch * 75, 302, self.hidden_dim))
            # 1) spatial mixing
            skip = results
            results = torch.permute(
                results, (0, 2, 1)
            )  # results.shape = (batch, channel, 302)
            results = self.layers["layernorm1_%d" % i](results)
            results = self.layers["linear1_%d" % i](
                results
            )  # results.shape = (batch, channel, token_mlp_dim)
            results = self.layers["gelu1_%d" % i](results)
            results = self.layers["linear2_%d" % i](
                results
            )  # results.shape = (batch, channel, 302)
            results = torch.permute(
                results, (0, 2, 1)
            )  # resutls.shape = (batch, 302, channel)
            results = results + skip
            # 2) channel mixing
            skip = results
            results = self.layers["layernorm2_%d" % i](results)
            results = self.layers["linear3_%d" % i](
                results
            )  # results.shape = (batch, 302, channels_mlp_dim)
            results = self.layers["gelu2_%d" % i](results)
            results = self.layers["linear4_%d" % i](
                results
            )  # results.shape = (batch, 302, channel)
            results = results + skip
            # remerge dimension
            results = torch.reshape(results, (batch, 75, 302, self.hidden_dim))
            results = torch.permute(results, (0, 2, 1, 3))
            results = torch.reshape(
                results, (batch * 302, 75, self.hidden_dim)
            )  # results.shape = (batch * 302, 75, channel)
            # 1) spatial mixing
            skip = results
            results = torch.permute(
                results, (0, 2, 1)
            )  # results.shape = (batch, channel, 75)
            results = self.layers["layernorm3_%d" % i](results)
            results = self.layers["linear5_%d" % i](
                results
            )  # results.shape = (batch, channel, token_mlp_dim)
            results = self.layers["gelu3_%d" % i](results)
            results = self.layers["linear6_%d" % i](
                results
            )  # results.shape = (batch, channel, 75)
            results = torch.permute(
                results, (0, 2, 1)
            )  # results.shape = (batch, 75, channel)
            results = results + skip
            # 2) channel mixing
            skip = results
            results = self.layers["layernorm4_%d" % i](results)
            results = self.layers["linear7_%d" % i](
                results
            )  # results.shape = (batch, 75, channels_mlp_dim)
            results = self.layers["gelu4_%d" % i](results)
            results = self.layers["linear8_%d" % i](
                results
            )  # results.shape = (batch, 75, channel)
            results = results + skip
            # reshape dimension
            results = torch.reshape(results, (batch, 302, 75, self.hidden_dim))
            results = torch.permute(
                results, (0, 2, 1, 3)
            )  # results.shape = (batch, 75, 302, channel)

        results = self.layernorm2(results)  # results.shape = (batch, 75, 302, channel)
        results = self.head(results)  # results.shape = (batch, 75, 302, 1)
        return results


class Predictor(nn.Module):
    def __init__(self, **kwargs):
        super(Predictor, self).__init__()
        self.predictor = MLPMixer(**kwargs)

    def forward(self, inputs):
        results = self.predictor(inputs)
        results = torch.squeeze(results, dim=-1)  # results.shape = (batch, 75, 302)
        return results


class PredictorSmall(nn.Module):
    def __init__(self):
        super(PredictorSmall, self).__init__()
        kwargs = {
            "hidden_dim": 256,
            "num_blocks": 3,
            "tokens_mlp_dim": 384,
            "channels_mlp_dim": 256 * 4,
            "drop_rate": 0.1,
        }
        self.predictor = Predictor(**kwargs)

    def forward(self, inputs):
        return self.predictor(inputs)


class PredictorBase(nn.Module):
    def __init__(self):
        super(PredictorBase, self).__init__()
        kwargs = {
            "hidden_dim": 768,
            "num_blocks": 12,
            "tokens_mlp_dim": 384,
            "channels_mlp_dim": 3072,
            "drop_rate": 0.1,
        }
        self.predictor = Predictor(**kwargs)

    def forward(self, inputs):
        return self.predictor(inputs)


class ModelDict:
    """
    Model_Dict
    """

    def __init__(self, **kwargs):
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
        self.load = kwargs.get("load")
        self.input_size = kwargs.get("input_size")
        self.dir_checkpoint = Path(CHECKPOINTS_PATH / self.load).resolve()

        self.device = kwargs.get("device")
        if kwargs.get("precision", "float32") == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float64

        self.keys = ["1"]
        self.model = PredictorSmall().to(torch.device("cuda"))

    def load_model(self):
        """
        Load the model from the checkpoint.
        """
        print(f"Load model from {self.dir_checkpoint}")
        list_of_path = list(self.dir_checkpoint.glob("*.pth"))
        load_path = max(list_of_path, key=lambda p: p.stat().st_ctime)
        state_dict = torch.load(load_path, map_location=self.device, weights_only=False)
        state_dict = {
            (key.replace("module.", "") if key.startswith("module.") else key): value
            for key, value in state_dict["state_dict"].items()
        }
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {load_path}")

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()

    def get_v(self, scf_r, grids):
        rho = grids.vector_to_matrix(scf_r)  # rho.shape = (natom, 75, 302)
        rho = torch.tensor(rho, dtype=self.dtype).to(self.device)
        inputs = torch.unsqueeze(rho, dim=-1)  # inputs.shape = (natom, 75, 302, 1)
        rho.requires_grad = True
        pred_exc = self.model(inputs)  # pred_exc.shape = (natom, 75, 302)
        pred_vxc = (
            torch.autograd.grad(torch.sum(rho * pred_exc), rho, create_graph=True)[0]
            + pred_exc
        )  # pred_exc.shape = (natom, 75, 302)
        pred_vxc = pred_vxc.detach().cpu().numpy()
        vxc_scf = grids.matrix_to_vector(pred_vxc)
        return vxc_scf

    def get_e(self, scf_r, grids):
        rho = grids.vector_to_matrix(scf_r)  # rho.shape = (natom, 75, 302)
        rho = torch.tensor(rho, dtype=self.dtype).to(self.device)
        inputs = torch.unsqueeze(rho, dim=-1)  # inputs.shape = (natom, 75, 302, 1)
        inputs.requires_grad = True
        pred_exc = self.model(inputs)  # pred_exc.shape = (natom, 75, 302)
        exc_scf = grids.matrix_to_vector(pred_exc)
        return exc_scf

    # def get_v(self, scf_r_3, grids):
    #     rho = grids.vector_to_matrix(scf_r_3[0, :])  # rho.shape = (natom, 75, 302)
    #     rho = torch.tensor(rho, dtype=self.dtype).to(self.device)
    #     inputs = torch.unsqueeze(rho, dim=-1)  # inputs.shape = (natom, 75, 302, 1)
    #     _, pred_vxc = self.model_dict["1"](inputs)  # pred_exc.shape = (natom, 75, 302)
    #     pred_vxc = pred_vxc.detach().cpu().numpy()
    #     vxc_scf = grids.matrix_to_vector(pred_vxc)
    #     return vxc_scf

    # def get_e(self, scf_r_3, grids):
    #     rho = grids.vector_to_matrix(scf_r_3[0, :])  # rho.shape = (natom, 75, 302)
    #     rho = torch.tensor(rho, dtype=self.dtype).to(self.device)
    #     inputs = torch.unsqueeze(rho, dim=-1)  # inputs.shape = (natom, 75, 302, 1)
    #     pred_exc, _ = self.model_dict["1"](inputs)  # pred_exc.shape = (natom, 75, 302)
    #     exc_scf = grids.matrix_to_vector(pred_exc)
    #     return exc_scf
