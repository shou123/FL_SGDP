#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import argparse
import pickle
import time
import os
import torch
import pandas as pd
import numpy as np


from utils import Data, build_graph
from model import *
from tqdm import tqdm
from collections import Counter
from cache import *
import matplotlib.pyplot as plt
from run_modes import run_distributed, run_federated


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int,
                    default=128, help='input batch size')
parser.add_argument('--hiddenSize', type=int,
                    default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=1,
                    help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1,
                    help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay')
# [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--step', type=int, default=1,
                    help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3,
                    help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true',
                    help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--draw_graph', action='store_true')
parser.add_argument('--see_ori_dataset', action='store_true')
parser.add_argument('--topn', type=int, default=20, help='top n')
parser.add_argument('--dataset_percent', type=float, default=1.0,
                    help='datasets percent for training and testing')
parser.add_argument('--window', type=int, default=32, help='window')
parser.add_argument('--topnum', type=int, default=1000, help='top n')

parser.add_argument('--NUM_CLIENTS', type=int, default=10, help='number of client')
parser.add_argument('--run_type', type=str, default="distributed", help='distributed or federated')

opt = parser.parse_args()
print(opt)


def dict_generate(train_trace, top_num=1000):
    train_trace['ByteOffset_Delta'] = train_trace['ByteOffset'] - \
        train_trace['ByteOffset'].shift(-1)
    train_trace['ByteOffset_Delta'] = train_trace['ByteOffset_Delta'].fillna(0)

    a = train_trace['ByteOffset_Delta'].astype(int).unique().tolist()

    operation_id_map = {}
    for i, id in enumerate(a):
        operation_id_map[id] = i
    train_trace['ByteOffset_Delta_class'] = train_trace['ByteOffset_Delta'].map(
        lambda x: operation_id_map[x])

    x = Counter(train_trace['ByteOffset_Delta_class'])
    vals = {}
    vals = x.most_common(top_num)
    bo_list = []

    for x in vals:
        bo_list.append(x[0])

    count = 0
    label_list = []
    while (count < len(train_trace)):
        x = train_trace['ByteOffset_Delta_class'].iloc[count]
        if x in bo_list:
            label_list.append(x)
        else:
            label_list.append(999999)  # no Prefetch class
        count = count + 1

    train_trace['ByteOffset_Delta_class'] = label_list
    a = train_trace['ByteOffset_Delta_class'].unique().tolist()
    bo_map = {}
    for i, id in enumerate(a):
        bo_map[id] = i
    operation_id_map_div = {v: k for k, v in operation_id_map.items()}
    operation_id_map_div[999999] = 0
    bo_map_div = {v: k for k, v in bo_map.items()}

    return bo_map, bo_map_div, operation_id_map, operation_id_map_div


def trace2input(dicts, trace, window_size=32):
    bo_map, _, operation_id_map, _ = dicts
    # print(len(trace))

    keys = bo_map.keys()
    inputs = []
    targets = []
    for i in tqdm(range(len(trace)-window_size-1)):

        input_single = []
        for j in range(i, i+window_size+1):
            diff = int(trace[j]-trace[j+1])
            if operation_id_map[diff] in keys:
                input_single.append(bo_map[operation_id_map[diff]]+1)###
            else:
                input_single.append(bo_map[999999]+1)###
        inputs.append(input_single[:-1])
        targets.append(input_single[-1])
    return inputs, targets




def dataset2input(dataset, window_size=32, method='top', top_num=1000):
    if method == 'top':
        print('\ntrain dataset, trace name:\t', dataset, '\n')
        lba_trace = 'dataset/8k_lba_traces/' + dataset + '.csv'
        print('\n', lba_trace, '\n')
        names = ['ByteOffset', 'TimeStamp']
        df = pd.read_csv(lba_trace, engine='python', skiprows=1, header=None, na_values=[
                         '-1'], usecols=[0, 1], names=names)
        df = df.sort_values(by=['TimeStamp'])
        df.reset_index(inplace=True, drop=True)

        train_trace = df[:int(len(df)*-opt.valid_portion)
                         ]['ByteOffset'].tolist()
        test_trace = df[int(len(df)*-opt.valid_portion) +
                        1:]['ByteOffset'].tolist()

        dicts = dict_generate(df, top_num=top_num)

        train_data = tuple(trace2input(
            dicts, train_trace, window_size=window_size))
        graph = build_graph(train_data[0])
        
        test_data = tuple(trace2input(
            dicts, test_trace, window_size=window_size))
        train_data = Data(train_data, shuffle=True,graph = graph)
        test_data = Data(test_data, shuffle=False)

        train_silces = train_data.generate_batch(opt.batchSize)
        train_data_list = []

        for i in tqdm(train_silces):
            alias_inputs, A, items, mask, targets = train_data.get_slice(i)
            train_data_list.append((alias_inputs, A, items, mask, targets))

        test_silces = test_data.generate_batch(opt.batchSize)
        test_data_list = []

        for i in tqdm(test_silces):
            alias_inputs, A, items, mask, targets = test_data.get_slice(i)
            test_data_list.append((alias_inputs, A, items, mask, targets))

        n_node = top_num + 3

        return train_data_list, train_silces, test_data_list, test_silces, dicts, n_node, train_trace, test_trace



def dataset2input_cached(dataset, window_size=32, top_num=1000, cache_dir="cache"):
    """
    Wrapper for dataset2input to cache the result to a file.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Define the cache file path
    cache_file = os.path.join(cache_dir, f"{dataset}_window{window_size}_top{top_num}.pkl")

    # Check if the cache file exists
    if os.path.exists(cache_file):
        print(f"Loading cached dataset2input result from {cache_file}...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # If cache doesn't exist, compute the result
    print(f"Cache not found. Computing dataset2input for {dataset}...")
    result = dataset2input(dataset, window_size=window_size, top_num=top_num)

    # Save the result to the cache file
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
        print(f"Saved dataset2input result to {cache_file}.")

    return result

    
def main():
    dataset_col = ['src1_2']#'ftds_0802_1021_trace','ftds_0805_17_trace',
    deviceID = 1
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(deviceID)
    device = torch.device('cuda:'+str(deviceID))

    # Check the run type and execute the corresponding logic
    if opt.run_type == "distributed":
        print("Running in Distributed Learning Mode...")
        run_distributed(opt, dataset_col, dataset2input_cached)
    elif opt.run_type == "federated":
        print("Running in Federated Learning Mode...")
        run_federated(opt, dataset_col, dataset2input_cached)
    else:
        raise ValueError("Invalid run type. Choose 'distributed' or 'federated'.")

    torch.cuda.empty_cache()
    print('-------------------------------------------------------')




if __name__ == '__main__':
    main()
