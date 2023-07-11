#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   process_dataloader.py
@Desc         :   data to dataloader
'''
from tqdm import tqdm
import numpy as np
import utils
from itertools import chain

def generate_embedding_nopid(config, datas, vectors, show=True):
    embeded_datas = []
    if(config.embedding_model_name == 'doc2vec'):
        # Raw Calls + doc2vec
        index = 0
        for data in tqdm(datas):
            embeded_datas.append(list(vectors.docvecs[index]))
            index += 1
    elif(config.embedding_model_name == 'w2v'):
        # Raw Calls + w2v
        if(show):
            for data in tqdm(datas):
                embeded_data = [list(vectors.wv[api_node])
                                for api_node in data]  # 对应索引
                if(config.data_type == 'mean'):
                    embeded_data = np.mean(
                        np.array(embeded_data, dtype=np.float32), axis=0)
                elif(config.data_type == 'max'):
                    embeded_data = np.max(
                        np.array(embeded_data, dtype=np.float32), axis=0)
                elif(config.data_type == 'sum'):
                    embeded_data = np.sum(
                        np.array(embeded_data, dtype=np.float32), axis=0)
                embeded_datas.append(embeded_data)
        else:
            for data in datas:
                embeded_data = [list(vectors.wv[api_node])
                                for api_node in data]  # 对应索引
                if(config.data_type == 'mean'):
                    embeded_data = np.mean(
                        np.array(embeded_data, dtype=np.float32), axis=0)
                elif(config.data_type == 'max'):
                    embeded_data = np.max(
                        np.array(embeded_data, dtype=np.float32), axis=0)
                elif(config.data_type == 'sum'):
                    embeded_data = np.sum(
                        np.array(embeded_data, dtype=np.float32), axis=0)
                embeded_datas.append(embeded_data)
    return embeded_datas


def generate_embedding_pid(config, datas, vectors):

    embeded_datas = []
    if(config.embedding_model_name == 'doc2vec'):
        # Paths + doc2vec
        index = 0
        for data in datas:
            embeded_data = []
            for _ in data:
                embeded_data.append(list(vectors.docvecs[index]))
                index += 1
            embeded_datas.append(
                np.mean(np.array(embeded_data, dtype=np.float32), axis=0))
    elif(config.embedding_model_name == 'w2v'):
        for data in tqdm(datas):
            embeded_paths = generate_embedding_nopid(config, data, vectors, show=False)
            embeded_datas.append(np.mean(np.array(embeded_paths, dtype=np.float32), axis=0))
        # embeded_datas = generate_embedding_nopid(config, datas, vectors, show=True)  # N * Dim

    return embeded_datas


def generate_embedding(config, datas, vectors):
    if(config.pid_flag == 'pid' or config.pid_flag == 'node2vec'):
        return generate_embedding_pid(config, datas, vectors)
    else:
        return generate_embedding_nopid(config, datas, vectors)

from torch.utils.data import Dataset, DataLoader, dataset
import torch
class SingleDataset(Dataset):
    def __init__(self, datas, labels):
        assert len(datas) == len(labels)
        self.datas = np.array(datas)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        inputs = torch.tensor(self.datas[index]).long()
        labels = torch.tensor(self.labels[index]).long()

        return {'inputs': inputs, 'labels' : labels}

def _split_datas(datas, labels, ratios=[0.7, 0.1, 0.2]):
    datas = np.array(datas)
    labels = np.array(labels)

    length = len(datas)
    train_len = int(length * ratios[0])
    val_len = int(length * ratios[1])

    indexes = np.random.permutation(np.arange(length))
    train_indexes = indexes[:train_len]
    val_indexes = indexes[train_len: train_len + val_len]
    test_indexes = indexes[train_len + val_len: ]

    return ((datas[item_indexes], labels[item_indexes]) for item_indexes in (train_indexes, val_indexes, test_indexes))


def convey_to_dataloader(config, datas, labels):
    datas = _split_datas(datas, labels, [0.6, 0.2, 0.2])

    dataset = SingleDataset

    train_dataset, val_dataset, test_dataset = [ dataset(data, label) for data, label in datas ]

    dataloaders = {name : torch.utils.data.DataLoader(dataset,batch_size=config.batch_size,shuffle=True) for dataset, name in
                                                        [(train_dataset, 'train'),(val_dataset, 'val'), (test_dataset, 'test')]}
    print([next(iter(dataset)) for dataset in [train_dataset, val_dataset, test_dataset]])

    return dataloaders

def data_split(config, datas, datanames, labels, vectors, api2idx, train_cut_index):

    if(config.model_type == 'dl'):
        dl_datas = []
        for data in datas:
            # 拼接
            data = list(chain.from_iterable(data))
            if(len(data) > config.sen_len):
                index = np.random.randint(0, len(data) - config.sen_len)
                data = data[index: index + config.sen_len]
            else:
                data = data + [0] * (config.sen_len - len(data))
            dl_datas.append(data)
        dataloaders = convey_to_dataloader(config, dl_datas, labels)
        return dataloaders, datanames, vectors, api2idx, train_cut_index

    if(config.task_name == 'multi'):
        datas, datanames, labels = utils.filter_multi(
            config, datas, datanames, labels)
    
    if(config.embedding_model_name == 'bert'):
        return (vectors, labels), datanames, vectors, api2idx, train_cut_index 
            
    embeded_datas = generate_embedding(config, datas, vectors)

    if(config.model_type == 'ml'):
        return (embeded_datas, labels), datanames, vectors, api2idx, train_cut_index
