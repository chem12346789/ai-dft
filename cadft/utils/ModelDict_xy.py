"""
Generate list of model.
"""

from pathlib import Path
import datetime

import numpy as np
import torch
import torch.optim as optim
from torch import nn

from cadft.utils.model.unet import UNet as Model
from cadft.utils.env_var import CHECKPOINTS_PATH
from cadft.utils.DataBase import process_input


class Attention(nn.Module):
    """
    Attention
    """

    def __init__(self, **kwargs):
        super(Attention, self).__init__()
        self.channel = kwargs.get("channel", 768)
        self.num_heads = kwargs.get("num_heads", 8)
        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.drop_rate = kwargs.get("drop_rate", 0.1)

        self.dense1 = nn.Linear(self.channel, self.channel * 3, bias=self.qkv_bias)
        self.dense2 = nn.Linear(self.channel, self.channel, bias=self.qkv_bias)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)

    def forward(self, inputs):
        # inputs.shape = (batch, seq_len, channel)
        results = self.dense1(inputs)  # results.shape = (batch, 302, 3 * channel)
        b, s, _ = results.shape
        results = torch.reshape(
            results, (b, s, 3, self.num_heads, self.channel // self.num_heads)
        )  # results.shape = (batch, seq_len, 3, head, channel // head)
        results = torch.permute(
            results, (0, 2, 3, 1, 4)
        )  # results.shape = (batch, 3, head, seq_len, channel // head)
        q, k, v = (
            results[:, 0, ...],
            results[:, 1, ...],
            results[:, 2, ...],
        )  # shape = (batch, head, seq_len, channel // head)
        qk = torch.matmul(
            q, torch.transpose(k, 2, 3)
        )  # qk.shape = (batch, head, seq_len, seq_len)
        attn = torch.softmax(qk, dim=-1)  # attn.shape = (batch, head, seq_len, seq_len)
        attn = self.dropout1(attn)
        qkv = torch.permute(
            torch.matmul(attn, v), (0, 2, 1, 3)
        )  # qkv.shape = (batch, seq_len, head, channel // head)
        qkv = torch.reshape(
            qkv, (b, s, self.channel)
        )  # qkv.shape = (batch, seq_len, channel)
        results = self.dense2(qkv)  # results.shape = (batch, seq_len, channel)
        results = self.dropout2(results)
        return results


class ABlock(nn.Module):
    def __init__(self, length, **kwargs):
        super(ABlock, self).__init__()
        self.channel = kwargs.get("channel", 768)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4)
        self.drop_rate = kwargs.get("drop_rate", 0.1)
        self.num_heads = kwargs.get("num_heads", 8)
        self.qkv_bias = kwargs.get("qkv_bias", False)

        self.dense1 = nn.Linear(self.channel, self.channel * self.mlp_ratio)
        self.dense2 = nn.Linear(self.channel * self.mlp_ratio, self.channel)
        self.gelu = nn.GELU()
        self.layernorm1 = nn.LayerNorm(
            [
                length,
                self.channel,
            ]
        )
        self.layernorm2 = nn.LayerNorm(
            [
                length,
                self.channel,
            ]
        )
        self.dropout0 = nn.Dropout(self.drop_rate)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.atten = Attention(**kwargs)

    def forward(self, inputs):
        # inputs.shape = (batch, length, channel)
        # attention
        skip = inputs
        results = self.layernorm1(inputs)  # results.shape = (batch, length, channel)
        results = self.atten(results)  # results.shape = (batch, length, channel)
        results = self.dropout0(results)
        results = skip + results
        # mlp
        skip = results
        results = self.layernorm2(results)
        results = self.dense1(
            results
        )  # results.shape = (batch, length, channel * mlp_ratio)
        results = self.gelu(results)
        results = self.dropout1(results)
        results = self.dense2(results)  # results.shape = (batch, length, channel)
        results = self.dropout2(results)
        results = skip + results
        return results


class Extractor(nn.Module):
    def __init__(self, **kwargs):
        super(Extractor, self).__init__()
        self.in_channel = kwargs.get("in_channel", 1)
        self.hidden_channels = kwargs.get("hidden_channels", 512)
        self.depth = kwargs.get("depth", 12)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        self.drop_rate = kwargs.get("drop_rate", 0.1)
        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.num_heads = kwargs.get("num_heads", 8)

        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.dense1 = nn.Linear(self.in_channel, self.hidden_channels)
        self.dense2 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.layernorm1 = nn.LayerNorm(
            [
                75,
                302,
                1,
            ]
        )
        self.layernorm2 = nn.LayerNorm([75, 302, self.hidden_channels])
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.layer_blocks = nn.ModuleList(
            [
                ABlock(
                    channel=self.hidden_channels,
                    qkv_bias=self.qkv_bias,
                    num_heads=self.num_heads,
                    length=75,
                    **kwargs,
                )
                for i in range(self.depth)
            ]
        )
        self.spatial_blocks = nn.ModuleList(
            [
                ABlock(
                    channel=self.hidden_channels,
                    qkv_bias=self.qkv_bias,
                    num_heads=self.num_heads,
                    length=302,
                    **kwargs,
                )
                for i in range(self.depth)
            ]
        )
        self.head = nn.Linear(self.hidden_channels, 1, bias=False)

    def forward(self, inputs):
        batch = inputs.shape[0]
        # inputs.shape = (batch, 75, 302, 1)
        # results = self.layernorm1(inputs)
        results = inputs
        results = self.dense1(
            results
        )  # results.shape = (batch, 75, 302, hidden_channels)
        results = self.gelu(results)
        results = self.dropout1(results)
        # do attention only when the feature shape is small enough
        for i in range(self.depth):
            # results.shape = (batch, 75, 302, hidden_channels)
            results = torch.reshape(
                results, (batch * 75, 302, self.hidden_channels)
            )  # results.shape = (batch * 75, 302, hidden_channels)
            results = self.spatial_blocks[i](
                results
            )  # results.shape = (batch * 75, 302, hidden_channels)
            results = torch.reshape(
                results, (batch, 75, 302, self.hidden_channels)
            )  # results.shape = (batch, 75, 302, hidden_channels)
            results = torch.permute(
                results, (0, 2, 1, 3)
            )  # result.shape = (batch, 302, 75, hidden_channels)
            results = torch.reshape(
                results, (batch * 302, 75, self.hidden_channels)
            )  # results.shape = (batch * 302, 75, hidden_channels)
            results = self.layer_blocks[i](
                results
            )  # results.shape = (batch * 302, 75, hidden_channels)
            results = torch.reshape(
                results, (batch, 302, 75, self.hidden_channels)
            )  # results.shape = (batch, 302, 75, hidden_channels)
            results = torch.permute(
                results, (0, 2, 1, 3)
            )  # results.shape = (batch, 75, 302, hidden_channel)
        results = self.layernorm2(results)
        results = self.dense2(
            results
        )  # results.shape = (batch, 75, 302, hidden_channels)
        results = self.tanh(
            results
        )  # results.shape = (batch, 75, 302, hidden_channels)
        results = self.head(results)  # results.shape = (batch, 75, 302, 1)
        return results


class PredictorSmall(nn.Module):
    def __init__(self, **kwargs):
        super(PredictorSmall, self).__init__()
        hidden_channels = kwargs.get("hidden_channels", 256)
        depth = kwargs.get("depth", 3)
        self.predictor = Extractor(
            hidden_channels=hidden_channels, depth=depth, **kwargs
        )

    def forward(self, inputs):
        return torch.squeeze(self.predictor(inputs), dim=-1)


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
        self.model_dict = {}
        self.model_dict["1"] = PredictorSmall(in_channel=1).to(self.device)

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
        self.model_dict["1"].load_state_dict(state_dict)
        print(f"Model loaded from {load_path}")

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        for key in self.keys:
            self.model_dict[key].eval()

    def get_v(self, scf_r_3, grids):
        rho = grids.vector_to_matrix(scf_r_3[0, :])  # rho.shape = (natom, 75, 302)
        rho = torch.tensor(rho, dtype=self.dtype).to(self.device)
        inputs = torch.unsqueeze(rho, dim=-1)  # inputs.shape = (natom, 75, 302, 1)
        rho.requires_grad = True
        pred_exc = self.model_dict["1"](inputs)  # pred_exc.shape = (natom, 75, 302)
        pred_vxc = (
            torch.autograd.grad(torch.sum(rho * pred_exc), rho, create_graph=True)[0]
            + pred_exc
        )  # pred_exc.shape = (natom, 75, 302)
        pred_vxc = pred_vxc.detach().cpu().numpy()
        vxc_scf = grids.matrix_to_vector(pred_vxc)
        return vxc_scf

    def get_e(self, scf_r_3, grids):
        rho = grids.vector_to_matrix(scf_r_3[0, :])  # rho.shape = (natom, 75, 302)
        rho = torch.tensor(rho, dtype=self.dtype).to(self.device)
        inputs = torch.unsqueeze(rho, dim=-1)  # inputs.shape = (natom, 75, 302, 1)
        inputs.requires_grad = True
        pred_exc = self.model_dict["1"](inputs)  # pred_exc.shape = (natom, 75, 302)
        exc_scf = grids.matrix_to_vector(pred_exc)
        return exc_scf
