import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, c, num_nodes, features, timesteps_input, timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(LSTM, self).__init__()
        self.num_node = num_nodes
        self.timeinput = timesteps_input
        self.hidden_size = c
        self.input_size = features
        # input size (b, input_size)
        self.lstm = torch.nn.LSTMCell(input_size=timesteps_input * features, hidden_size=timesteps_input)
        # nn.LSTM input x(b, n, t) output (b, n,hidden)
        # self.lstm = torch.nn.LSTM(input_size= timesteps_input * features, hidden_size=c, num_layers=1)
        self.lin = nn.Linear(c, timesteps_output)
        self.node = num_nodes
    def forward(self, x):
        """
        :param X: Input data of shape (b, n, t, c).
        """

        '''50:150,100:300,150:450'''
        b, n, t, c = x.shape
        x_em = x.reshape(b, n, t*c).permute(1, 0, 2)
        # hidden init
        h = torch.zeros(b, t)
        c = torch.zeros(b, t)
        output = []
        for i in range(n):
            h, c = self.lstm(x_em[i, :, :], (h, c))
            output.append(h)
        output = torch.stack(output, dim=0).permute(1, 0, 2)
        # output, (h, c) = self.lstm(x_em)
        # output = self.lin(output)

        return output


