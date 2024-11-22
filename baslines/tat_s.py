import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from dgl.nn.pytorch.conv.treeat import TreeAt

class SpatioConvLayer(nn.Module):
    """
    parameter:
    c : 通道数
    Lk: dgl图
    NATree: 空间树矩阵
    num_heads: 多头注意力机制数
    """
    def __init__(self, c, Lk, NATree, num_heads, input_szie=12, output_size=12):  # c : hidden dimension Lk: graph matrix
        super(SpatioConvLayer, self).__init__()
        self.g = Lk
        self.num_heads = num_heads
        self.gc = TreeAt(c[0], c[1], num_heads, NATree)
        self.fc = FullyConvLayer(c[1])
    def forward(self, x):
        """
         TreeAt input (b, n, t, c)
        """
        output = self.gc(self.g, x)
        output_head = output.view(output.shape[0], output.shape[1], output.shape[2], -1, self.num_heads)
        output_avg = torch.mean(output_head, dim=4).squeeze()
        fc_output = self.fc(output_avg.permute(0, 3, 1, 2))
        return torch.relu(fc_output).permute(0, 2, 3, 1)


class FullyConvLayer(nn.Module):
    def __init__(self, c):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, c, (1, 1))

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        # padding = (T - 1) * 1
        self.tconv1 = nn.Conv2d(c, c, (1, T), 1, dilation=1, padding=(0, 0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation=1, padding=(0, 0))
        # self.fc = FullyConvLayer(c)
        tmp = int((12-T)/1) + 1
        self.fc = nn.Linear(c * tmp, 12)

    def forward(self, x):
        b, n, t, c = x.shape
        x_t1 = self.tconv1(x.permute(0, 3, 1, 2))
        x_ln = self.ln(x_t1.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2.reshape(b, n, -1))

class TreeAt_TCN(nn.Module):
    def __init__(self, c, n, Lk, STree, nheads, time_input, time_output, control_str="TNTSTNTST"):
        super(TreeAt_TCN, self).__init__()
        self.control_str = control_str  # model structure controller
        self.num_layers = len(control_str)
        self.c = c
        self.fc = nn.Conv2d(in_channels=c[0], out_channels=c[1], kernel_size=(1, 1), stride=1)
        self.layers = nn.ModuleList([])
        cnt = 0
        diapower = 0
        for i in range(self.num_layers):
            i_layer = control_str[i]
            if i_layer == "S":  # Spatio Layer
                self.layers.append(SpatioConvLayer([c[cnt], c[cnt+1]], Lk, STree, nheads, time_input, time_output))
                diapower += 1
                cnt += 1
            if i_layer == "N":  # Norm Layer
                self.layers.append(nn.LayerNorm([n, c[cnt]]))
        self.output = OutputLayer(c[cnt], time_input, n)
        # self.output = OutputLayer(c[cnt], time_input + 1 - 2 ** (diapower), n)
        for layer in self.layers:
            layer = layer

    def forward(self, x):
        # x = self.fc(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        for i in range(self.num_layers):
            i_layer = self.control_str[i]
            if i_layer == "N":
                # input (b, t, n, c)
                x = self.layers[i](x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            else:
                x = self.layers[i](x)
        return self.output(x)