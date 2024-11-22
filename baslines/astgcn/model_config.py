# -*- coding:utf-8 -*-

import configparser
import numpy as np
import torch
import pandas as pd
from scipy.sparse.linalg import eigs


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    distance_matrix = pd.read_csv(distance_df_filename, sep=',', header=None, index_col=None)
    sigma = pow(1000000, 2)
    var = 0.8
    distance_exp = np.exp(-((distance_matrix*distance_matrix)/sigma))
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)
    for i in range(distance_exp.shape[0]):
        for j in range(i, distance_exp.shape[1]):
            if distance_exp[i][j] > var:
                A[i][j] = 1
                A[j][i] = 1
    return A
def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

def get_backbones(K, num_of_weeks, num_of_days, num_of_hours, num_of_vertices, adj_mx):

    K = K
    num_of_weeks = int(num_of_weeks)
    num_of_days = int(num_of_days)
    num_of_hours = int(num_of_hours)
    num_of_vertices = int(num_of_vertices)

    # adj_mx = get_adjacency_matrix(adj_filename, num_of_vertices)
    adj_mx = adj_mx
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = torch.tensor(cheb_polynomial(L_tilde, K), dtype=torch.float32)

    backbones1 = [
        {
            "K": K,
            "num_of_chev_filters": 32,
            "num_of_time_filters": 32,
            "time_conv_strides": num_of_weeks,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 32,
            "num_of_time_filters": 32,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones2 = [
        {
            "K": K,
            "num_of_chev_filters": 32,
            "num_of_time_filters": 32,
            "time_conv_strides": num_of_days,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 32,
            "num_of_time_filters": 32,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    backbones3 = [
        {
            "K": K,
            "num_of_chev_filters": 32,
            "num_of_time_filters": 32,
            "time_conv_strides": num_of_hours,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 32,
            "num_of_time_filters": 32,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]

    all_backbones = [
        backbones1,
        backbones2,
        backbones3
    ]

    return all_backbones
