import numpy as np
import pandas as pd
import torch
import os
import csv
from tqdm import tqdm
from fastdtw import fastdtw
from utils import *

files = {
    'pems03': ['PEMS03/PEMS03.npz', 'PEMS03/PEMS03.csv'],
    'pems04': ['PEMS04/PEMS04.npz', 'PEMS04/PEMS04.csv'],
    'pems07': ['PEMS07/PEMS07.npz', 'PEMS07/PEMS07.csv'],
    'pems08': ['PEMS08/PEMS08.npz', 'PEMS08/PEMS08.csv', '170'],
    'pemsbay': ['PEMSBAY/pems_bay.npz', 'PEMSBAY/distance.csv'],
    'pemsD7M': ['PeMSD7M/PeMSD7M.npz', 'PeMSD7M/distance.csv'],
    'pemsD7L': ['PeMSD7L/PeMSD7L.npz', 'PeMSD7L/distance.csv'],
    'metr-la': ['Metr-la/metr-la.npz', 'Metr-la/distances_la_2012.csv'],
    'randomuniformity': ['RandomUniformity/V_flow_50.npz', 'RandomUniformity/V_flow_50.csv'],
    'smallscaleaggregation': ['SmallScaleAggregation/V_flow_50.npz', 'SmallScaleAggregation/V_flow_50.csv']
}



def generate_adj_mx(args):
    filename = args.filename
    timesteps_input = args.window
    timesteps_output = args.pred_len
    file = files[filename]
    filepath = "./data/"
    data = np.load(filepath+file[0])['data'][:8640].astype(np.float32)
    if len(data.shape) == 2:
        timeslice = data.shape[0]
        num_nodes = data.shape[1]
        data = np.reshape(data, (timeslice, num_nodes, 1))
    data, data_mean, data_std = Z_Score(data)
    num_node = data.shape[1]
    if not os.path.exists(f'data/{filename}_dtw_distance.npy'):
        data_mean = np.mean([data[:, :, 0][24*12*i: 24*12*(i+1)] for i in range(data.shape[0]//(24*12))], axis=0)
        data_mean = data_mean.squeeze().T
        dtw_distance = np.zeros((num_node, num_node))
        for i in tqdm(range(num_node)):
            for j in range(i, num_node):
                dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
        for i in range(num_node):
            for j in range(i):
                dtw_distance[i][j] = dtw_distance[j][i]
        np.save(f'data/{filename}_dtw_distance.npy', dtw_distance)

    dist_matrix = np.load(f'data/{filename}_dtw_distance.npy')

    mean = np.mean(dist_matrix)
    std = np.std(dist_matrix)
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma1
    dist_matrix = np.exp(-dist_matrix ** 2 / sigma ** 2)
    dtw_matrix = np.zeros_like(dist_matrix)
    dtw_matrix[dist_matrix > args.thres1] = 1
    np.save(f'data/{filename}_se_c_matrix.npy', dtw_matrix)

    # use continuous spatial matrix
    # if not os.path.exists(f'data/{filename}_spatial_distance.npy'):
    #     with open(filepath + file[1], 'r') as fp:
    #         dist_matrix = np.zeros((num_node, num_node)) + np.float('inf')
    #         file = csv.reader(fp)
    #         for line in file:
    #             break
    #         for line in file:
    #             start = int(line[0])
    #             end = int(line[1])
    #             dist_matrix[start][end] = float(line[2])
    #             dist_matrix[end][start] = float(line[2])
    #         np.save(f'data/{filename}_spatial_distance.npy', dist_matrix)
    # use 0/1 spatial matrix
    if not os.path.exists(f'data/{filename}_sp_matrix.npy'):
        dist_matrix = np.load(f'data/{filename}_spatial_distance.npy')
        sp_matrix = np.zeros((num_node, num_node))
        sp_matrix[dist_matrix != np.float('inf')] = 1
        np.save(f'data/{filename}_sp_matrix.npy', sp_matrix)
    sp_matrix = np.load(f'data/{filename}_sp_matrix.npy')

    # normalization
    std = np.std(dist_matrix[dist_matrix != np.float('inf')])
    mean = np.mean(dist_matrix[dist_matrix != np.float('inf')])
    dist_matrix = (dist_matrix - mean) / std
    sigma = args.sigma2
    sp_matrix = np.exp(- dist_matrix**2 / sigma**2)
    sp_matrix[sp_matrix < args.thres2] = 0
    np.save(f'data/{filename}_sp_c_matrix.npy', sp_matrix)
    # sp_matrix = np.load(f'data/{filename}_sp_c_matrix.npy')

    return dtw_matrix, sp_matrix

def Data_load(args):
    filename = args.filename
    timesteps_input = args.window
    timesteps_output = args.pred_len
    file = files[filename]
    filepath = "./data/"
    data = np.load(filepath+file[0])['data'][:8640].astype(np.float32)
    if len(data.shape) == 2:
        timeslice = data.shape[0]
        num_nodes = data.shape[1]
        data = np.reshape(data, (timeslice, num_nodes, 1))
    data_cp = data
    input_features = data_cp.shape[2]
    num_nodes = data_cp.shape[1]
    data_length = data.shape[0]
    data, data_mean, data_std = Z_Score(data)
    index_1 = int(data_length * 0.8)
    index_2 = int(data_length * 0.9)
    train_original_data = data[:index_1]
    val_original_data = data[index_1: index_2]
    test_original_data = data[index_2:]

    train_input, train_target = generate_dataset(train_original_data, timesteps_input, timesteps_output)
    evaluate_input, evaluate_target = generate_dataset(val_original_data, timesteps_input, timesteps_output)
    test_input, test_target = generate_dataset(test_original_data, timesteps_input, timesteps_output)

    data_set = {}
    data_set['train_input'], data_set['train_target'], data_set['eval_input'], data_set['eval_target'], \
    data_set['test_input'], data_set['test_target'], data_set['data_mean'], data_set['data_std'],\
    data_set['input_features'], data_set['num_nodes'] = train_input, train_target, evaluate_input, evaluate_target, test_input, test_target,\
                                 data_mean, data_std, input_features, num_nodes
    return data_set


# def load_data(file_path, len_train, len_val):
#     df = pd.read_csv(file_path, header=None).values.astype(float)
#     train = df[:len_train]
#     val = df[len_train : len_train + len_val]
#     test = df[len_train + len_val :]
#     return train, val, test


# def data_transform(data, n_his, n_pred, device):
#     # produce data slices for training and testing
#     n_route = data.shape[1]
#     l = len(data)
#     num = l - n_his - n_pred
#     x = np.zeros([num, 1, n_his, n_route])
#     y = np.zeros([num, n_route])
#
#     cnt = 0
#     for i in range(l - n_his - n_pred):
#         head = i
#         tail = i + n_his
#         x[cnt, :, :, :] = data[head:tail].reshape(1, n_his, n_route)
#         y[cnt] = data[tail + n_pred - 1]
#         cnt += 1
#     return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
