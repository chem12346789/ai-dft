#!/usr/bin/python3

import torch
from torch import nn


class Attention(nn.Module):
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
        # inputs.shape = (batch, channel, seq_len)
        results = self.dense1(
            torch.transpose(inputs, 1, 2)
        )  # results.shape = (batch, seq_len, 3 * channel)
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
        results = torch.transpose(
            results, 1, 2
        )  # results.shape = (batch, channel, seq_len)
        return results


class ABlock(nn.Module):
    def __init__(self, input_size, **kwargs):
        super(ABlock, self).__init__()
        self.input_size = input_size
        self.channel = kwargs.get("channel", 768)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4)
        self.drop_rate = kwargs.get("drop_rate", 0.1)
        self.num_heads = kwargs.get("num_heads", 8)
        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.groups = kwargs.get("groups", 1)

        self.dense1 = nn.Linear(self.channel, self.channel * self.mlp_ratio)
        self.dense2 = nn.Linear(self.channel * self.mlp_ratio, self.channel)
        self.gelu = nn.GELU()
        self.layernorm1 = nn.LayerNorm([self.channel, 302])
        self.layernorm2 = nn.LayerNorm([self.channel, 302])
        self.dropout0 = nn.Dropout(self.drop_rate)
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.dropout2 = nn.Dropout(self.drop_rate)
        self.atten = Attention(**kwargs)

    def forward(self, inputs):
        # inputs.shape = (batch, c, seq_len)
        # attention
        skip = inputs
        results = self.layernorm1(inputs)
        results = self.atten(results)  # results.shape = (batch, channel, seq_len)
        results = self.dropout0(results)
        results = skip + results
        # mlp
        skip = results
        results = self.layernorm2(results)
        results = torch.permute(results, (0, 2, 1))
        results = self.dense1(
            results
        )  # results.shape = (batch, channel * mlp_ratio, seq_len)
        results = torch.permute(results, (0, 2, 1))
        results = self.gelu(results)
        results = self.dropout1(results)
        results = torch.permute(results, (0, 2, 1))
        results = self.dense2(results)  # results.shape = (batch, channel, seq_len)
        results = torch.permute(results, (0, 2, 1))
        results = self.dropout2(results)
        results = skip + results
        return results


class Extractor(nn.Module):
    def __init__(self, **kwargs):
        super(Extractor, self).__init__()
        self.in_channel = kwargs.get("in_channel", 75)
        self.hidden_channels = kwargs.get("hidden_channels", 512)
        self.depth = kwargs.get("depth", 12)
        self.mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        self.drop_rate = kwargs.get("drop_rate", 0.1)
        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.num_heads = kwargs.get("num_heads", 8)
        self.groups = kwargs.get("groups", 1)

        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.dense1 = nn.Linear(75, self.hidden_channels)
        self.dense2 = nn.Linear(self.hidden_channels, self.in_channel * 2)
        self.layernorm1 = nn.LayerNorm([75, 302])
        self.layernorm2 = nn.LayerNorm([self.hidden_channels, 302])
        self.dropout1 = nn.Dropout(self.drop_rate)
        self.blocks = nn.ModuleList(
            [
                ABlock(
                    input_size=9,
                    channel=self.hidden_channels,
                    qkv_bias=self.qkv_bias,
                    num_heads=self.num_heads,
                    **kwargs
                )
                for i in range(self.depth)
            ]
        )

    def forward(self, inputs):
        # inputs.shape = (batch, 75, seq_len)
        results = self.layernorm1(inputs)
        results = torch.permute(results, (0, 2, 1))
        results = self.dense1(
            results
        )  # results.shape = (batch, hidden_channels[0], seq_len)
        results = torch.permute(results, (0, 2, 1))
        results = self.gelu(results)
        results = self.dropout1(results)
        # do attention only when the feature shape is small enough
        for i in range(self.depth):
            results = self.blocks[i](results)
        results = self.layernorm2(results)
        results = torch.permute(results, (0, 2, 1))
        results = self.dense2(
            results
        )  # results.shape = (batch, hidden_channels, seq_len)
        results = torch.permute(results, (0, 2, 1))
        return results


if __name__ == "__main__":
    extractor = Extractor()
    inputs = torch.randn(2, 75, 302)
    outputs = extractor(inputs)
    print(outputs.shape)
