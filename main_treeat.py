import argparse
import random
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from load_data import *
from baslines.tat_s import *
from sensors2graph import *
from sklearn.preprocessing import StandardScaler
from utils import *
from tqdm import tqdm
from logger import Logger
from torch.utils.data import DataLoader
import dgl

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
    "--epochs", type=int, default=100, help="epochs for training  (default: 200)"
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
    default="./data/Metr-la/graph_sensor_ids.txt",
    help="sensors file path",
)
parser.add_argument(
    "--disfilepath",
    type=str,
    default="./data/Metr-la/distances_la_2012.csv",
    help="distance file path",
)
parser.add_argument(
    "--savemodelpath",
    type=str,
    default="./best_model/Metr-la/tat_2h.pt",
    help="save model path",
)
parser.add_argument(
    "--pred_len",
    type=int,
    default=12,
    help="how many steps away we want to predict",
)
parser.add_argument(
    "--channels",
    type=int,
    nargs="+",
    default=[1, 32, 32, 32, 64, 64],
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
STree = T._tree.get_tree_matrixw()
if STree.shape[0] != num_nodes:
    if STree.shape[0] > num_nodes:
        STree = STree[:num_nodes, :, :]
    else:
        pad_zero = torch.zeros(1, STree.shape[1], STree.shape[2])
        for i in range(num_nodes - STree.shape[0]):
            STree = torch.cat((STree, pad_zero), dim=0)
STree = STree.to(device)
controlstr = 'SN'
model = TreeAt_TCN(c=blocks,
                   n=num_nodes,
                   Lk=G,
                   STree=STree,
                   nheads=12,
                   time_input=n_his,
                   time_output=n_pred,
                   control_str=controlstr).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
min_val_mae = np.inf
min_test_mae = np.inf
if_save_true = False
# file_path = "result.txt"
# with open(file_path, 'w') as file:
#     file.write("Tree Train Process: \n")
elogger = Logger('tat_s_12h')

for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    std = torch.tensor(data_set['data_std']).to(device)
    mean = torch.tensor(data_set['data_mean']).to(device)
    for x, y in tqdm(train_iter):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x)
        y, y_pred = Un_Z_Score(y, mean, std), Un_Z_Score(y_pred, mean, std)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()

    #
    torch.cuda.empty_cache()
    # val data
    val_loss, val_mae = evaluate_model(model, loss, val_iter, mean, std)
    # if val_loss < min_val_loss:
    # if val_mae < min_val_mae:
    #     min_val_mae = val_mae
    #     torch.save(model.state_dict(), save_path)
    #     # 保存文件
    #     val_train = data_set['eval_input']
    #     val_target = data_set['eval_target']
    #     val_pred = model(val_train)
    #     val_pred, val_target = Un_Z_Score(val_pred, mean, std), Un_Z_Score(val_target, mean, std)
    #     save_result_path = "./result/treeat1/"
    #     if not os.path.exists(save_result_path):
    #         os.makedirs(save_result_path)
    #     val_pred = val_pred.detach().numpy()
    #     val_target = val_target.detach().numpy()
    #     np.savetxt(save_result_path + "pred_" + str(epoch) + ".csv", val_pred[:, :, 0], delimiter=',')
    #     if if_save_true == False:
    #         if_save_true = True
    #         np.savetxt(save_result_path + "true_" + ".csv", val_test[:, :, 0], delimiter=',')


    # test data
    MAE, MAPE, RMSE = evaluate_metric(model, test_iter, mean, std)
    test_loss, test_mae = evaluate_model(model, loss, test_iter, mean, std)

    # if MAE < min_test_mae:
    #     min_test_mae = MAE
    #     test_train = data_set['test_input']
    #     test_target = data_set['test_target']
    #     test_pred = model(test_train)
    #     test_pred, test_target = Un_Z_Score(test_pred, mean, std), Un_Z_Score(test_target, mean, std)
    #     if not os.path.exists("./best_model/"):
    #         os.makedirs("./best_model/")
    #     torch.save(model.state_dict(), save_path)
    #     save_result_path = "./result/Metr-la/tat_s_2h/"
    #     if not os.path.exists(save_result_path):
    #         os.makedirs(save_result_path)
    #     test_pred = test_pred.detach().numpy()
    #     test_target = test_target.detach().numpy()
    #     np.savetxt(save_result_path + "pred_" + str(epoch) + ".csv", test_pred[:, :, 0], delimiter=',')
    #     if if_save_true == False:
    #         if_save_true = True
    #         np.savetxt(save_result_path + "true_" + ".csv", test_target[:, :, 0], delimiter=',')


    print("epoch: {}, train loss: {}, validation loss: {}\n".format(epoch, l_sum/n, val_loss))
    elogger.log("epoch: {}, train loss: {}, validation loss: {}\n".format(epoch, l_sum/n, val_loss))

    print("test loss:", test_loss, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
    elogger.log("test loss:{} \n MAE: {}, MAPE:{}, RMSE:{}".format(test_loss, MAE, MAPE, RMSE))
    elogger.log("######################")


std = torch.tensor(data_set['data_std']).to(device)
mean = torch.tensor(data_set['data_mean']).to(device)
# best_model = TreeAt_TCN(c=blocks,
#                    n=num_nodes,
#                    Lk=G,
#                    STree=STree,
#                    nheads=1,
#                    time_input=n_his,
#                    time_output=n_pred,
#                    control_str=controlstr).to(device)
# best_model.load_state_dict(torch.load(save_path))
# l = evaluate_model(best_model, loss, test_iter, mean, std)
# MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, mean, std)
# print("final test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
