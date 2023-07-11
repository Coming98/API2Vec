#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   main.py
@Desc         :   data process & random walk & embedding & train & test
'''


import sys
import os
import pickle
import time
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


def word2vec(save_name, corpus, labels, datanames, idx2api):
    datas = _process(save_name, api2vec, (corpus, labels, datanames, idx2api))
    return datas


def generate_dataloader(save_name, inputs, datanames, labels, vectors, api2idx, train_cut_index):
    datas = _process(save_name, data_split, (inputs, datanames,
                     labels, vectors, api2idx, train_cut_index))
    return datas


def data_process_main():
    datas = init_api_sequence(CONFIG.init_data_name)
    if(CONFIG.pid_flag == 'nopid'):
        pid_api_sequences, base_sequences, status, datanames, labels, api2idx, idx2api, train_cut_index = datas
        if(CONFIG.model_type == 'ml'):
            vectors, names = word2vec(
                CONFIG.tovec_name, base_sequences, labels, datanames, idx2api)
        data_loaders, datanames, vectors, api2idx, train_cut_index = generate_dataloader(
            CONFIG.todata_name, base_sequences, datanames, labels, vectors, api2idx, train_cut_index)
    elif(CONFIG.pid_flag == 'pid'):
        start_time = time.time()
        pid_api_sequences, base_sequences, status, datanames, tpg_datas, tag_datas, labels, api2idx, idx2api, train_cut_index = datas
        cost_time = time.time() - start_time
        print("TOTAL COUNT = ", len(pid_api_sequences))
        print(f'{cost_time:=}')
        if(CONFIG.model_type == 'ml'):
            start_time = time.time()
            corpus = generate_corpus(
                CONFIG.random_walk_name, pid_api_sequences, tpg_datas, tag_datas, labels)
            cost_time = time.time() - start_time
            print(f'{cost_time:=}')
            start_time = time.time()
            vectors, _ = word2vec(CONFIG.tovec_name, corpus,
                                  labels, datanames, idx2api)
            cost_time = time.time() - start_time
            print(f'{cost_time:=}')
            data_loaders, datanames, vectors, api2idx, train_cut_index = generate_dataloader(
                CONFIG.todata_name, corpus, datanames, labels, vectors, api2idx, train_cut_index)
            
        # data_loaders, datanames, vectors, api2idx, train_cut_index = generate_dataloader(CONFIG.todata_name, base_sequences, datanames, labels, vectors, api2idx, train_cut_index)
    elif(CONFIG.pid_flag == 'node2vec'):
        pid_api_sequences, base_sequences, status, datanames, graph_datas, labels, api2idx, idx2api, train_cut_index = datas
        corpus = generate_corpus(
            CONFIG.random_walk_name, pid_api_sequences, graph_datas, None, None)
        vectors, names = word2vec(
            CONFIG.tovec_name, corpus, labels, datanames, idx2api)
        # data_loaders, datanames, vectors, api2idx, train_cut_index = generate_dataloader(CONFIG.todata_name, corpus, datanames, labels, vectors, api2idx, train_cut_index)
        data_loaders, datanames, vectors, api2idx, train_cut_index = generate_dataloader(
            CONFIG.todata_name, base_sequences, datanames, labels, vectors, api2idx, train_cut_index)

    return data_loaders, datanames, vectors, api2idx, train_cut_index


def data_load_main():

    data_loaders, datanames, vectors, api2idx, train_cut_index = generate_dataloader(
        CONFIG.todata_name, None, None, None, None, None, None)

    return data_loaders, datanames, vectors, api2idx, train_cut_index


def malware_detection(data_loaders, datanames, vectors, api2idx, train_cut_index):
    if(CONFIG.model_type == 'ml'):
        train_ml(CONFIG, data_loaders, datanames, train_cut_index)


def main():

    flag_path = os.path.join(CONFIG.data_output_dir, CONFIG.todata_name)

    if(not os.path.exists(flag_path)):
        data_loaders, datanames, vectors, api2idx, train_cut_index = data_process_main()
    else:
        data_loaders, datanames, vectors, api2idx, train_cut_index = data_load_main()

    print(len(datanames))

    malware_detection(data_loaders, datanames, vectors,
                      api2idx, train_cut_index)


if __name__ == '__main__':
    main()

# API2Vec
#√ python main.py normal_cuck_pid_doc2vec_ml_knn_mean_status+count 
# Node2Vec
#√ 1,2      python main.py normal_cuck_node2vec_w2v_ml_knn_mean_status+None
#√ 1,1      python main.py normal_cuck_node2vec_w2v_ml_knn_mean_status+None
#√ 1,0.5    python main.py normal_cuck_node2vec_w2v_ml_knn_mean_status+None
# Basic
#√ python main.py normal_cuck_nopid_w2v_ml_knn_mean_status+None

##################### Attack
# python main.py attack_cuck_pid_doc2vec_ml_knn_mean_status+count
# python main.py attack_cuck_node2vec_w2v_ml_knn_mean_status+None
# python main.py attack_cuck_nopid_w2v_ml_knn_mean_status+None


#################### Target Type
# API2Vec
# python main.py target_cuck_pid_doc2vec_ml_knn_mean_status+count+virus
# python main.py target_cuck_pid_doc2vec_ml_knn_mean_status+count+backdoor
# python main.py target_cuck_pid_doc2vec_ml_knn_mean_status+count+worm
# python main.py target_cuck_pid_doc2vec_ml_knn_mean_status+count+grayware
# python main.py target_cuck_pid_doc2vec_ml_knn_mean_status+count+downloader

# Node2Vec
# python main.py target_cuck_node2vec_w2v_ml_knn_mean_status+None+virus
# python main.py target_cuck_node2vec_w2v_ml_knn_mean_status+None+backdoor
# python main.py target_cuck_node2vec_w2v_ml_knn_mean_status+None+worm
# python main.py target_cuck_node2vec_w2v_ml_knn_mean_status+None+grayware
# python main.py target_cuck_node2vec_w2v_ml_knn_mean_status+None+downloader

# Basic
# python main.py target_cuck_nopid_w2v_ml_knn_mean_status+None+virus
# python main.py target_cuck_nopid_w2v_ml_knn_mean_status+None+backdoor
# python main.py target_cuck_nopid_w2v_ml_knn_mean_status+None+worm
# python main.py target_cuck_nopid_w2v_ml_knn_mean_status+None+grayware
# python main.py target_cuck_nopid_w2v_ml_knn_mean_status+None+downloader
