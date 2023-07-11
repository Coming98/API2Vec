#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   utils.py
@Desc         :   Tool case of 'main process'
'''

from config import Config
import pickle
import matplotlib.pyplot as plt
from itertools import chain


def get_config(sys, logger=None):

    if(len(sys.argv) <= 1):
        print("Input a configuration name: ", end='')
        sys.argv.append(input())

    config = Config(sys.argv[1])

    config.print_config()

    return config


def show_plot(x, y, x_label, y_label, save=False):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if(save):
        plt.savefig(save)
    else:
        plt.show()


# PRINT

def prostart(path):
    print(f"Processing {path} ......")


def proend():
    print(f"Processing Success!")


def loadstart(path):
    print(f"Loading {path} ......")


def loadend():
    print(f"Loading Success!")


def pickle_load(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def pickle_dump(data, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print("Dump Success")


def data2multiclass(config, datanames):
    sha256s = [name.split('\\')[-1][:-5] for name in datanames]
    # Load namemap
    name_map = pickle_load('./Analysis/VT_NameMap/resource/nameMap.pkl')
    # Load fam2idx
    fam2idx, idx2fam = pickle_load(
        './Analysis/VT_NameMap/resource/fam2idx.pkl')
    labels = []
    for sha256 in sha256s:
        class_ = name_map[sha256]['class_']
        if(class_ not in fam2idx):
            class_ = 'others'
        label = fam2idx[class_]
        labels.append(label)

    accept_label = [fam2idx[item] for item in config.multi_classes]
    return labels, accept_label


def filter_mvs(config, datas, datanames, labels):
    datas_ = []
    datanames_ = []
    labels_ = []
    for i, name in enumerate(datanames):
        brief_name = name.split('\\')[-1].split('.')[0]
        if(brief_name not in config.ignored_names):
            datas_.append(datas[i])
            datanames_.append(name)
            labels_.append(labels[i])
    return datas_, datanames_, labels_


def path2name(path):
    return path.split('\\')[-1].split('.')[0]
