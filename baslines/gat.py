import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dgl.nn.pytorch.conv.gatconv import GATConv



class SpatioConvLayer(nn.Module):
    """
    parameter:
    c : 通道数
    Lk: dgl图
    NATree: 空间树矩阵
    num_heads: 多头注意力机制数
    """
    def __init__(self, c, Lk, num_heads):  # c : hidden dimension Lk: graph matrix
        super(SpatioConvLayer, self).__init__()
        self.num_heads = num_heads
        self.g = Lk
        self.gc = GATConv(c, c, num_heads, feat_drop=0.1, attn_drop=0.1)
    def forward(self, x):
        b, n, t, c = x.shape
        x = x.permute(1, 0, 2, 3) #change shape from (b, n, t, c) into (n, b, t, c)
        output = self.gc(self.g, x).reshape(n, b, t * self.num_heads, c)
        output = output.permute(1, 0, 2, 3)
        return torch.relu(output)


class FullyConvLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (1, T), 1, dilation=1, padding=(0, 0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation=1, padding=(0, 0))
        self.fc = nn.Linear(c * 12, 12)

    def forward(self, x):
        b, n, t, c = x.shape
        x_t1 = self.tconv1(x.permute(0, 3, 1, 2))
        x_ln = self.ln(x_t1.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2.reshape(b, n, c * 12))

class GAT_WAVE(nn.Module):
    def __init__(
        self, c, T, n, Lk, num_heads, control_str="SNSN"
    ):
        super(GAT_WAVE, self).__init__()
        self.control_str = control_str  # model structure controller
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList([])
        cnt = 0
        diapower = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == "F":
                self.layers.append(FullyConvLayer(c[cnt], c[cnt + 1]))
                diapower += 1
                cnt += 1
            if i_layer == "S":  # Spatio Layer
                self.layers.append(SpatioConvLayer(c[cnt], Lk, num_heads))
            if i_layer == "N":  # Norm Layer
                self.layers.append(nn.LayerNorm([n, c[cnt]]))
        self.output = OutputLayer(c[cnt], (num_heads-1) * T + 1, n)
        for layer in self.layers:
            layer = layer

    def forward(self, x):
        # input x shape is (b, n, t, c)
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == "N":
                x = self.layers[i](x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            else:
                x = self.layers[i](x)
        return self.output(x)