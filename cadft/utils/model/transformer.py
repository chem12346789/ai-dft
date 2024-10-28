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

        self.dense1 = nn.Linear(self.channel, self.channel * 3)
        self.dense2 = nn.Linear(self.channel, self.channel)

    def forward(self, inputs):
        # inputs.shape = (batch, seq_len, channel)
        results = self.dense1(inputs)
        # results.shape = (batch, 302, 3 * channel)
        b, s, _ = results.shape
        results = torch.reshape(
            results, (b, s, 3, self.num_heads, self.channel // self.num_heads)
        )
        # results.shape = (batch, seq_len, 3, head, channel // head)
        results = torch.permute(results, (0, 2, 3, 1, 4))
        # results.shape = (batch, 3, head, seq_len, channel // head)
        q, k, v = (
            results[:, 0, ...],
            results[:, 1, ...],
            results[:, 2, ...],
        )
        # shape = (batch, head, seq_len, channel // head)
        qk = torch.matmul(q, torch.transpose(k, 2, 3))
        # qk.shape = (batch, head, seq_len, seq_len)
        attn = torch.softmax(qk, dim=-1)
        # attn.shape = (batch, head, seq_len, seq_len)
        qkv = torch.permute(torch.matmul(attn, v), (0, 2, 1, 3))
        # qkv.shape = (batch, seq_len, head, channel // head)
        qkv = torch.reshape(qkv, (b, s, self.channel))
        # qkv.shape = (batch, seq_len, channel)
        results = self.dense2(qkv)
        # results.shape = (batch, seq_len, channel)
        return results


class ABlock(nn.Module):
    def __init__(self, length, **kwargs):
        super(ABlock, self).__init__()
        self.channel = kwargs.get("channel", 768)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4)

        self.dense1 = nn.Linear(self.channel, self.channel * self.mlp_ratio)
        self.dense2 = nn.Linear(self.channel * self.mlp_ratio, self.channel)
        self.gelu = nn.GELU()
        self.atten = Attention(**kwargs)

    def forward(self, inputs):
        # inputs.shape = (batch, length, channel)
        # attention
        skip = inputs
        results = self.atten(inputs)
        results = skip + results
        # results.shape = (batch, length, channel)
        # mlp with skip connection
        skip = results
        results = self.dense1(results)
        # results.shape = (batch, length, channel * mlp_ratio)
        results = self.dense2(results)
        # results.shape = (batch, length, channel)
        results = skip + results
        return results


class Extractor(nn.Module):
    def __init__(self, **kwargs):
        super(Extractor, self).__init__()
        self.in_channel = kwargs.get("in_channel", 1)
        self.hidden_channels = kwargs.get("hidden_channels", 512)
        self.depth = kwargs.get("depth", 12)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.num_heads = kwargs.get("num_heads", 8)

        self.gelu = nn.GELU()
        self.dense1 = nn.Linear(self.in_channel, self.hidden_channels)
        self.dense2 = nn.Linear(self.hidden_channels, self.hidden_channels)

        self.layer_blocks = nn.ModuleList(
            [
                ABlock(
                    channel=self.hidden_channels,
                    qkv_bias=self.qkv_bias,
                    num_heads=self.num_heads,
                    length=75,
                    **kwargs,
                )
                for _ in range(self.depth)
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
                for _ in range(self.depth)
            ]
        )
        self.head = nn.Linear(self.hidden_channels, 1, bias=False)

    def forward(self, inputs):
        batch = inputs.shape[0]
        # inputs.shape = (batch, 75, 302, 1)
        results = inputs
        results = self.dense1(inputs)
        # results.shape = (batch, 75, 302, hidden_channels)
        results = self.gelu(results)
        # do attention only when the feature shape is small enough
        for i in range(self.depth):
            # results.shape = (batch, 75, 302, hidden_channels)
            results = torch.reshape(results, (batch * 75, 302, self.hidden_channels))
            # results.shape = (batch * 75, 302, hidden_channels)
            results = self.spatial_blocks[i](results)
            # results.shape = (batch * 75, 302, hidden_channels)
            results = torch.reshape(results, (batch, 75, 302, self.hidden_channels))
            # results.shape = (batch, 75, 302, hidden_channels)
            results = torch.permute(results, (0, 2, 1, 3))
            # result.shape = (batch, 302, 75, hidden_channels)
            results = torch.reshape(results, (batch * 302, 75, self.hidden_channels))
            # results.shape = (batch * 302, 75, hidden_channels)
            results = self.layer_blocks[i](results)
            # results.shape = (batch * 302, 75, hidden_channels)
            results = torch.reshape(results, (batch, 302, 75, self.hidden_channels))
            # results.shape = (batch, 302, 75, hidden_channels)
            results = torch.permute(results, (0, 2, 1, 3))
            # results.shape = (batch, 75, 302, hidden_channel)
        results = self.dense2(results)
        # results.shape = (batch, 75, 302, hidden_channels)
        results = self.head(results)
        # results.shape = (batch, 75, 302, 1)
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
