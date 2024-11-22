import numpy as np
import pandas as pd
import torch
from utils import *

def search_data(sequence_length, num_of_batches, label_strat_idx, timesteps_input,
                units, points_per_hour):
    '''
    :param sequence_length: 历史数据长度 int
    :param num_of_batches: 用于训练的batch大小，int
    :param label_strat_idx:
    :param num_of_predict:
    :param units:
    :param points_per_hour:
    :return:
    '''
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    if label_strat_idx + timesteps_input > sequence_length:
        return None
    x_idx = []
    for i in range(1, num_of_batches+1):
        start_idx = label_strat_idx - points_per_hour * units * i
        end_idx = start_idx + timesteps_input
        if start_idx >=0:
            x_idx.append((start_idx,end_idx))
        else:
            return None
    if len(x_idx) != num_of_batches:
        return None

    return x_idx[::-1]

#生成样本序列函数
def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours, label_start_idx,
                       timesteps_input, timesteps_output, points_per_hour=12):
    '''
    :param data_sequence:数据序列，(sequence_length,num_of_vertices,num_of_features)
    :param num_of_weeks:周周期数 int
    :param num_of_days:天周期数 int
    :param num_of_hours:近期周期数 int
    :param label_start_idx:预测目标的开始下标 int
    :param num_of_predict: 每个样本的预测点数 int
    :param points_per_hour: 每小时的点数 int 默认为12
    :return:
    '''
    week_indices = search_data(data_sequence.shape[0], num_of_weeks, label_start_idx,
                               timesteps_input, 7*24, points_per_hour)
    if not week_indices:
        return None

    day_indices = search_data(data_sequence.shape[0], num_of_days, label_start_idx,
                              timesteps_input, 24, points_per_hour)
    if not day_indices:
        return None

    hour_indices = search_data(data_sequence.shape[0], num_of_hours, label_start_idx,
                               timesteps_input, 1, points_per_hour)
    if not hour_indices:
        return None

    week_sample = np.concatenate([data_sequence[i:j]
                                  for i, j in week_indices], axis=0)
    day_sample = np.concatenate([data_sequence[i:j]
                                 for i, j in day_indices], axis=0)
    hour_sample = np.concatenate([data_sequence[i:j]
                                 for i, j in hour_indices], axis=0)
    target = data_sequence[label_start_idx:label_start_idx+timesteps_output]

    return week_sample, day_sample, hour_sample, target


def read_and_generate_dataset(filename, num_of_weeks, num_of_days, num_of_hours,
                               timesteps_input, timesteps_output, points_per_hour=12):
    """
    Parameters
    ----------
    filename: str, path of graph signal matrix file

    num_of_weeks, num_of_days, num_of_hours: int

    num_for_predict: int

    points_per_hour: int, default 12, depends on data

    merge: boolean, default False,
           whether to merge training set and validation set to train model

    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)
         wd: shape is (num_of_samples, num_of_vertices, num_of_features,
                       num_of_weeks/days/hours * points_per_hour)??

    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)

    """

    data_seq = np.load(filename)['data'][:8640].astype(np.float32)
    if len(data_seq.shape) == 2:
        timeslice = data_seq.shape[0]
        num_nodes = data_seq.shape[1]
        data_seq = np.reshape(data_seq, (timeslice, num_nodes, 1))
    # 归一化
    data_seq, data_mean, data_std = Z_Score(data_seq)

    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days, num_of_hours,
                                    idx, timesteps_input, timesteps_output, points_per_hour)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample
        '''周周期数据，天周期数据，时周期数据按增加的维度进行拼接'''
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 1, 3)),
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 1, 3)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 1, 3)),
            np.expand_dims(target, axis=0).transpose((0, 2, 1, 3))[:, :, :, 0]  # wd: first feature is the traffic flow
        ))
    # 数据划分线
    split_line1 = int(len(all_samples) * 0.8)
    split_line2 = int(len(all_samples) * 0.9)
    # 生成训练集、验证集、测试集
    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    #生成周、天、近期的训练数据以及预测目标数据
    train_week, train_day, train_hour, train_target = training_set
    #生成周、天、近期的验证数据以及预测目标数据
    val_week, val_day, val_hour, val_target = validation_set
    #生成周、天、近期的测试数据以及预测目标数据
    test_week, test_day, test_hour, test_target = testing_set

    # print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
    #     train_week.shape, train_day.shape,train_hour.shape, train_target.shape))
    # print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
    #     val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    # print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
    #     test_week.shape, test_day.shape, test_hour.shape, test_target.shape))


    all_data = {
        'train': {
            'week': train_week,
            'day': train_day,
            'recent': train_hour,
            'target': train_target,  # wd: target does not need to be normalized?
        },
        'val':{
            'week': val_week,
            'day': val_day,
            'recent': val_hour,
            'target': val_target,
        },
        'test': {
            'week': test_week,
            'day': test_day,
            'recent': test_hour,
            'target': test_target
        },

        'data_mean': data_mean,
        'data_std': data_std
    }

    return all_data