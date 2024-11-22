import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from dgl.nn.pytorch.conv.treeat import TreeAt


class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """

    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size)
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size[i] - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size[i]), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            if kernel_size[i] != 1:
                self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            if kernel_size[i] != 1:
                layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
            else:
                layers += [nn.Sequential(self.conv, self.relu, self.dropout)]
        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        like ResNet
        Args:
            X : input data of shape (B, N, T, F)
        """
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        return y.permute(0, 2, 3, 1)

class SpatioConvLayer(nn.Module):
    """
    parameter:
    c : 通道数
    Lk: dgl图
    NATree: 空间树矩阵
    num_heads: 多头注意力机制数
    """
    def __init__(self, c, Lk, NATree, num_heads=1, input_szie=12, output_size=12):  # c : hidden dimension Lk: graph matrix
        super(SpatioConvLayer, self).__init__()
        self.g = Lk
        self.gc = TreeAt(c, c, num_heads, NATree)
        self.fc = FullyConvLayer(c)
        self.ln = nn.Linear(input_szie, output_size)
    def forward(self, x):
        """
         TreeAt input (b, n, t, c)
        """
        output = self.gc(self.g, x)
        fc_output = self.fc(output.permute(0, 3, 1, 2))
        # ln_output = self.ln(fc_output)
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
            if i_layer == "T":  # Temporal Layer
                num_levels = 3
                num_channels = num_levels * [c[cnt+1]]
                kernel = [3, 5, 1]
                self.layers.append(
                    TemporalConvNet(c[cnt], num_channels, kernel)
                )
                diapower += 1
                cnt += 1
            if i_layer == "S":  # Spatio Layer
                self.layers.append(SpatioConvLayer(c[cnt], Lk, STree, nheads, time_input, time_output))
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