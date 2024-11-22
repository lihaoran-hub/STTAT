import numpy as np
import torch
from torch.utils.data import Dataset
class DatasetPEMS(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, index):
        sample = self.data[0][index]
        label = self.data[1][index]

        return sample, label

def Z_Score(matrix):
    mean, std = np.mean(matrix), np.std(matrix)
    return (matrix - mean) / (std+0.001), mean, std


def Un_Z_Score(matrix, mean, std):
    return (matrix * std) + mean

def RMSE(v, v_):
    return torch.sqrt(torch.mean((v_ - v) ** 2))


def MAE(v, v_):
    return torch.mean(torch.abs(v_ - v))

def SMAPE(v, v_):
    """
    Mean absolute percentage error.
    :param v: np.ndarray or int, ground truth.
    :param v_: np.ndarray or int, prediction.
    :return: int, MAPE averages on all elements of input.
    """
    return torch.mean(torch.abs((v_ - v) / ((torch.abs(v) + torch.abs(v_)) / 2 + 1e-5)))


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[0] - (num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []

    for i, j in indices:
        features.append(
            X[i: i + num_timesteps_input, :, :].transpose((1, 0, 2)))
        target.append(X[i + num_timesteps_input: j, :, 0].transpose((1, 0)))
    return torch.from_numpy(np.array(features)), torch.from_numpy(np.array(target))

def evaluate_model(model, loss, data_iter, mean, std):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        mae = []
        for x, y in data_iter:
            # astgcn
            # y_pred = model([x.permute(0, 1, 3, 2), x.permute(0, 1, 3, 2), x.permute(0, 1, 3, 2)])
            y_pred = model(x)
            y = y
            y, y_pred = Un_Z_Score(y, mean, std), Un_Z_Score(y_pred, mean, std)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        MAE = np.array(mae).mean()
        return l_sum / n, MAE


def evaluate_metric(model, data_iter, mean, std):
    model.eval()
    mean = mean.cpu().numpy()
    std = std.cpu().numpy()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = y.cpu().numpy().reshape(-1)
            # astgcn
            # y_pred = model([x.permute(0, 1, 3, 2), x.permute(0, 1, 3, 2), x.permute(0, 1, 3, 2)]).contiguous().view(len(x), -1).cpu().numpy().reshape(-1)
            y_pred = model(x).contiguous().view(len(x), -1).cpu().numpy().reshape(-1)
            y, y_pred = Un_Z_Score(y, mean, std), Un_Z_Score(y_pred, mean, std)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d**2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE

def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (np.eye(A.shape[0]) + A_wave)
    return torch.from_numpy(A_reg.astype(np.float32))
