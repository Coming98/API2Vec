#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   utils.py
@Desc         :   Tool case of 'main process'
'''

from config import Config
import pickle
import matplotlib.pyplot as plt

def get_config(sys, logger=None):
    
    if(len(sys.argv) <= 1):
        print("请输入配置目录名: ", end='')
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


################# PRINT

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
    # 加载 namemap
    name_map = pickle_load('./Analysis/VT_NameMap/resource/nameMap.pkl')
    # 加载 fam2idx
    fam2idx, idx2fam = pickle_load('./Analysis/VT_NameMap/resource/fam2idx.pkl')
    labels = []
    for sha256 in sha256s:
        class_ = name_map[sha256]['class_']
        if(class_ not in fam2idx): class_ = 'others'
        label = fam2idx[class_]
        labels.append(label)

    accept_label = [fam2idx[item] for item in config.multi_classes]
    return labels, accept_label

def filter_multi(config, datas, datanames, labels):
    datas = [datas[i] for i in range(len(datas)) if labels[i] == 0]
    datanames = [datanames[i] for i in range(len(datanames)) if labels[i] == 0]
    labels, accept_labels = data2multiclass(config, datanames)
    datas = [datas[i] for i in range(len(datas)) if labels[i] in accept_labels]
    datanames = [datanames[i] for i in range(len(datanames)) if labels[i] in accept_labels]
    labels = [labels[i] for i in range(len(labels)) if labels[i] in accept_labels]
    return datas, datanames, labels

def filter_time(config, corpus, labels, datanames):
    train_start_year, train_end_year = config.train_start_year, config.train_end_year
    test_start_year, test_end_year = config.test_start_year, config.test_end_year
    accept_train_years = list(range(train_start_year, train_end_year + 1))
    accept_test_years = list(range(test_start_year, test_end_year + 1))
    accept_years = accept_train_years + accept_test_years

    filtered_corpus, filtered_labels, filtered_datanames = [], [], []
    train_data_names, test_data_names = [], []
    # accept_2021 = 2000 - 154
    for i, name in enumerate(datanames):
        name = name.split('\\')[-1].split('.')[0]
        if(config.name2time[name]['year'] in accept_years): 
            filtered_corpus.append(corpus[i])
            filtered_labels.append(labels[i])
            filtered_datanames.append(datanames[i])
            if(config.name2time[name]['year'] in accept_test_years):
                test_data_names.append(name)
            else:
                train_data_names.append(name)
            
    pickle_dump((train_data_names, test_data_names), f'./outputs/{train_start_year}_{train_end_year}+{test_start_year}_{test_end_year}.names.pkl')
    return filtered_corpus, filtered_labels, filtered_datanames
