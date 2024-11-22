import argparse
import random
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from load_data import *
from model import *
from sensors2graph import *
from utils import *
from tqdm import tqdm
from logger import Logger
from torch.utils.data import DataLoader
from sklearn.svm import SVR
import dgl
import pickle

parser = argparse.ArgumentParser(description="STGCN_WAVE")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--disablecuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="batch size for training and validation (default: 64)",
)
parser.add_argument(
    "--epochs", type=int, default=200, help="epochs for training  (default: 200)"
)
parser.add_argument(
    "--num_layers", type=int, default=1, help="number of layers"
)
parser.add_argument(
    "--nheads", type=int, default=1, help="number of att heads"
)
parser.add_argument(
    "--tree_num_layers", type=int, default=10, help="number of tree layers"
)
parser.add_argument("--window", type=int, default=12, help="window length")

parser.add_argument('--filename', type=str, default='metr-la')
parser.add_argument(
    "--sensorsfilepath",
    type=str,
    default="./data/sensor_graph/graph_sensor_ids.txt",
    help="sensors file path",
)
parser.add_argument(
    "--disfilepath",
    type=str,
    default="./data/sensor_graph/distances_la_2012.csv",
    help="distance file path",
)
parser.add_argument(
    "--tsfilepath", type=str, default="./data/metr-la.h5", help="ts file path"
)



parser.add_argument(
    "--savemodelpath",
    type=str,
    default="./best_model/svm.pt",
    help="save model path",
)
parser.add_argument(
    "--pred_len",
    type=int,
    default=12,
    help="how many steps away we want to predict",
)
parser.add_argument(
    "--control_str",
    type=str,
    default="TSTN",
    help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
)
parser.add_argument(
    "--channels",
    type=int,
    nargs="+",
    default=[1, 32, 32, 64, 64, 128],
    help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
)
args = parser.parse_args()

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.disablecuda
    else torch.device("cpu")
)

with open(args.sensorsfilepath) as f:
    sensor_ids = f.read().strip().split(",")

distance_df = pd.read_csv(args.disfilepath, dtype={"from": "str", "to": "str"})

adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
sp_mx = sp.coo_matrix(adj_mx)
T = dgl.from_scipy_tree(sp_mx, True)
G = dgl.from_scipy(sp_mx, True)


# read data
# df = pd.read_hdf(args.tsfilepath)
# num_samples, num_nodes = df.shape
# tsdata = df.to_numpy()


data_set = Data_load(args)

save_path = args.savemodelpath
n_his = args.window
n_pred = args.pred_len
blocks = args.channels
drop_prob = 0
num_layers = args.num_layers
tree_num_layers = args.tree_num_layers

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
W = adj_mx


# generate data_loader

train_data = torch.utils.data.TensorDataset(data_set['train_input'], data_set['train_target'])
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(data_set['eval_input'], data_set['eval_target'])
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(data_set['test_input'], data_set['test_target'])
test_iter = torch.utils.data.DataLoader(test_data, batch_size)


num_nodes = data_set['num_nodes']
input_features = data_set['input_features']


loss = nn.MSELoss()
T = T.to(device)
G = G.to(device)
STree = T._tree.get_tree_matrixw().to(device)
# svm
svr_rbf = SVR(kernel='rbf', C=10, gamma=1)
# x_train (b, n, t, c) y_train(b, n, t)
x_train, y_train = data_set['train_input'], data_set['train_target']
x_val, y_val = data_set['eval_input'], data_set['eval_target']
x_test, y_test = data_set['test_input'],  data_set['test_target']

time, timesteps = y_test.shape[0], y_test.shape[2]

# x shape is (t_all, n, t_12)
x_train = x_train[:, :, :, 0].view(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
y_train = y_train.view(y_train.shape[0], y_train.shape[1] * y_train.shape[2])
x_val = x_val[:, :, :, 0].view(x_val.shape[0], x_val.shape[1] * x_val.shape[2])
y_val = y_val.view(y_val.shape[0], y_val.shape[1] * y_val.shape[2])
x_test = x_test[:, :, :, 0].view(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
y_test = y_test.view(y_test.shape[0], y_test.shape[1] * y_test.shape[2])



len = x_train.shape[1]
y_pred_all = []

result_path = "./result/svm1/"
if not os.path.exists(result_path):
    os.makedirs(result_path)
svr_rbf.fit(x_train[:, 0:1], y_train[:, 0])
with open(result_path + f"svr.pkl", 'wb') as f:
    pickle.dump(svr_rbf, f)
for i in range(len):
    # save model
    # if (i+1) % 207 == 0:
    #     with open(result_path+f"svr{i}.pkl", 'wb') as f:
    #         pickle.dump(svr_rbf, f)
    y_rbf = svr_rbf.predict(x_test[:, i:i+1])
    y_pred_all.append(y_rbf)

y_pred_all = torch.from_numpy(np.array(y_pred_all)).transpose(1, 0)

# y_pred_all = y_pred_all.repeat(1, 2484)

# get MAE, RMSE, MAPE

mae = []
mape = []
mse = []

std = data_set['data_std']
mean = data_set['data_mean']

y_test = y_test.cpu().numpy().reshape(-1)
y_pred_all = y_pred_all.cpu().numpy().reshape(-1)

# inverse normalization
y_test, y_pred_all = Un_Z_Score(y_test, mean, std), Un_Z_Score(y_pred_all, mean, std)

d = np.abs(y_test-y_pred_all)

mae += d.tolist()
mape += (d / y_test).tolist()
mse += (d**2).tolist()
MAE = np.array(mae).mean()
MAPE = np.array(mape).mean()
RMSE = np.sqrt(np.array(mse).mean())

elogger = Logger('run_log_svm_1')

print("test result:", "MAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
elogger.log(f"test result: MAE:{MAE}, MAPE:{MAPE}, RMSE:{RMSE}")
elogger.log("######################")


# save result
y_pred = y_pred_all.reshape(time, num_nodes, timesteps)
y_target = y_test.reshape(time, num_nodes, timesteps)

save_result_path = "./result/svm1/"
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)
np.savetxt(save_result_path + "pred_svm" + ".csv", y_pred[:, :, 0], delimiter=',')
np.savetxt(save_result_path + "true_svm" + ".csv", y_target[:, :, 0], delimiter=',')



#
# std = torch.tensor(data_set['data_std']).to(device)
# mean = torch.tensor(data_set['data_mean']).to(device)
# # best_model = STGCN_WAVE(c=blocks,
# #                    n=num_nodes,
# #                    Lk=G,
# #                    STree=STree,
# #                    nheads=1,
# #                    time_input=n_his,
# #                    time_output=n_pred,
# #                    control_str=args.control_str).to(device)
#
# best_model = GAT_WAVE(c=blocks,
#                       T=n_his,
#                       n=num_nodes,
#                       Lk=G,
#                       num_heads=1,
#                       control_str='SN').to(device)
# best_model.load_state_dict(torch.load(save_path))
# l = evaluate_model(best_model, loss, test_iter, mean, std)
# MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, mean, std)
# print("final test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
