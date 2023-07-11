#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   main.py
@Desc         :   data process & random walk & embedding & train & test
'''


import sys
import os
import pickle
import utils
from DataProcess.process_api_sequence import load_api_sequence
from DataProcess.process_corpus import load_corpus
from DataProcess.api2vec import api2vec
from DataProcess.process_dataloader import data_split
from train_ml import train_ml
CONFIG = utils.get_config(sys, )


def _process(process_name, process_func, process_params=tuple()):
    save_path = os.path.join(CONFIG.data_output_dir, process_name)
    if not os.path.exists(save_path):
        utils.prostart(save_path)
        datas = process_func(CONFIG, *process_params)
        with open(save_path, 'wb') as f:
            pickle.dump(datas, f)
        utils.proend()
    else:
        utils.loadstart(save_path)
        with open(save_path, 'rb') as f:
            datas = pickle.load(f)
        utils.loadend()
    return datas


def init_api_sequence(save_name):
    datas = _process(save_name, load_api_sequence, ())
    return datas


def generate_corpus(save_name, pid_api_sequences, tpg_datas, tag_datas, labels):
    datas = _process(save_name, load_corpus,
                     (pid_api_sequences, tpg_datas, tag_datas, labels))
    return datas


def tovec(save_name, corpus, labels, datanames, idx2api):
    datas = _process(save_name, api2vec, (corpus, labels, datanames, idx2api))
    return datas


def generate_dataloader(save_name, inputs, datanames, labels, models):
    datas = _process(save_name, data_split,
                     (inputs, datanames, labels, models))
    return datas


def data_process_main():
    datas = init_api_sequence(CONFIG.init_data_name)
    if(CONFIG.pid_flag == 'nopid'):
        pid_api_sequences, base_sequences, status, datanames, labels, api2idx, idx2api, train_cut_index = datas
        tovec_model = tovec(CONFIG.tovec_name, base_sequences,
                            labels, datanames, idx2api)
        data_loaders = generate_dataloader(
            CONFIG.todata_name, base_sequences, datanames, labels, tovec_model)
    elif(CONFIG.pid_flag == 'pid'):
        pid_api_sequences, base_sequences, status, datanames, tpg_datas, tag_datas, labels, api2idx, idx2api, train_cut_index = datas
        corpus = generate_corpus(
            CONFIG.random_walk_name, pid_api_sequences, tpg_datas, tag_datas, labels)
        tovec_model = tovec(CONFIG.tovec_name, corpus,
                            labels, datanames, idx2api)
        data_loaders = generate_dataloader(
            CONFIG.todata_name, corpus, datanames, labels, tovec_model)
    elif(CONFIG.pid_flag == 'node2vec'):
        pid_api_sequences, base_sequences, status, datanames, graph_datas, labels, api2idx, idx2api, train_cut_index = datas
        corpus = generate_corpus(
            CONFIG.random_walk_name, pid_api_sequences, graph_datas, None, None)
        tovec_model = tovec(CONFIG.tovec_name, corpus,
                            labels, datanames, idx2api)
        data_loaders = generate_dataloader(
            CONFIG.todata_name, base_sequences, datanames, labels, tovec_model)

    return data_loaders


def data_load_main():

    data_loaders = generate_dataloader(
        CONFIG.todata_name, None, None, None, None)

    return data_loaders


def malware_detection(data_loaders):
    if(CONFIG.model_type == 'ml'):
        train_ml(CONFIG, data_loaders)


def main():
    # 如果完成了数据的预处理则直接执行检测任务
    flag_path = os.path.join(CONFIG.data_output_dir, CONFIG.todata_name)

    if(not os.path.exists(flag_path)):
        # 进行数据的预处理
        data_loaders = data_process_main()
    else:
        # 加载处理完成的数据
        data_loaders = data_load_main()

    ((train_X, train_y, train_names), (test_X, test_y, test_names)) = data_loaders
    train_count = len(train_X)
    test_count = len(test_y)
    print("Train_Count", train_count)
    print("Train_Black_Count", train_count - sum(train_y))
    print("Train White Count", sum(train_y))
    print("Test Count", test_count)
    print("Test Black Count", test_count - sum(test_y))
    print("Test White Count", sum(test_y))

    malware_detection(data_loaders)


if __name__ == '__main__':
    main()

# Attack - PID
# python main.py 2009_2018_2019_2019_cuck_pid_doc2vec_ml_knn_mean_status+count
# python main.py 2009_2018_2020_2020_cuck_pid_doc2vec_ml_knn_mean_status+count
# Attack - Basic
# python main.py 2009_2018_2019_2019_cuck_nopid_w2v_ml_knn_mean_status+None
# python main.py 2009_2018_2020_2020_cuck_nopid_w2v_ml_knn_mean_status+None
# Attack - node2vec
# python main.py 2009_2018_2019_2019_cuck_node2vec_w2v_ml_knn_mean_status+None
# python main.py 2009_2018_2020_2020_cuck_node2vec_w2v_ml_knn_mean_status+None

#############################################
# Attack - Basic

# python main.py 2009_2018_2019_2019_cuck_nopid_w2v_ml_knn_mean_status+None
# python main.py 2009_2018_2020_2020_cuck_nopid_w2v_ml_knn_mean_status+None

# Attack - Node2Vec

# python main.py 2009_2018_2019_2019_cuck_node2vec_w2v_ml_knn_mean_status+None
# python main.py 2009_2018_2020_2020_cuck_node2vec_w2v_ml_knn_mean_status+None

# Attack - PID
# python main.py 2009_2018_2019_2019_cuck_pid_doc2vec_ml_knn_mean_status+count
# python main.py 2009_2018_2020_2020_cuck_pid_doc2vec_ml_knn_mean_status+count
