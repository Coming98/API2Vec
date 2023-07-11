#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   process_api_sequence.py
@Desc         :   process api sequence
'''

import os
import pandas as pd
from tqdm import tqdm
import numpy as np

def init_data(config, filepath):
    if(config.data_name == 'cuck'):
        data = pd.read_excel(filepath)
        data = data.sort_values('timestamp')
        api_names = data[config.api_header]
        pids = data[config.pid_header]
        status = data[config.status_header]
        status = [True if state == 'SUCCESS' else False for state in list(status)]

    return api_names, pids, status

def load_api_sequence_pid(config, filepath, api2idx):

    pid_order_list = [] 

    base_sequences = [] # 记录原始序列

    tpg_apisequence_dict = {} # pid: api_sequence
    pid_apit_dict = {} # pid: apit_sequence
    tag_extime_dict = {} # tag: extime

    api_names, pids, status = init_data(config, filepath)

    for t, (api_name, pid, state) in enumerate(zip(api_names, pids, status), 1):
        if('status' in config.params):
            api_name += ('_S' if state else '_F')

        if(api_name not in api2idx): api2idx[api_name] = len(api2idx)
        api_index = api2idx[api_name]
        base_sequences.append(api_index)

        # pid - 原始序列
        if(pid not in pid_order_list): pid_order_list.append(pid)

        # tag_node - min(extime)
        if(api_index not in tag_extime_dict): tag_extime_dict[api_index] = t

        # pid - api sequence
        # pid - time sequence
        if(pid not in tpg_apisequence_dict):
            tpg_apisequence_dict[pid] = [api_index, ]
            pid_apit_dict[pid] = [t, ]
        else:
            tpg_apisequence_dict[pid].append(api_index) # 按照 PID 划分 API calls
            pid_apit_dict[pid].append(t) # 同时记录每个 API call 的执行时间
        

    tag_edges = {} # pid - (api_s, api_e)
    tag_edget_dict = {} # pid - {(api_s, api_e): time sequence}
    for pid, api_sequence in tpg_apisequence_dict.items():
        edges = list(zip(api_sequence, api_sequence[1:]))
        tag_edges[pid] = list(set(edges))

        if(pid not in tag_edget_dict.keys()): tag_edget_dict[pid] = {}

        for edge, t in zip(edges, pid_apit_dict[pid][1:]):
            if(edge not in tag_edget_dict[pid].keys()):
                tag_edget_dict[pid][edge] = [t, ]
            else:
                tag_edget_dict[pid][edge].append(t)

    pid_api_sequences = [tpg_apisequence_dict[pid] for pid in pid_order_list]
    return pid_api_sequences, base_sequences, status, pid_apit_dict, (tag_edges, tag_edget_dict, tag_extime_dict), len(api_names), len(set(api_names))

def load_api_sequence_cuck(config):

    api2idx = {'<UNK>' : 0}
    idx2api = {0 : '<UNK>'}

    pid_api_sequences = []
    base_sequences = []
    status = []
    datanames = []
    labels = []
    
    tag_datas = [] # edges, edge_time_dict, node_extime_min
    tpg_datas = [] # edges, edge_time_dict

    for rootpath, _, filenames in os.walk(config.data_dir): 
        
        file_type = os.path.split(rootpath)[-1] # 黑白样本的标识通过 文件夹名称来识别
        if(file_type not in ('black', 'white')): continue

        for filename in tqdm(filenames):

            pid_api_sequence_xml, base_sequence_xml, status_xml, tpg_data, tag_data, length, count = load_api_sequence_pid(config, os.path.join(rootpath, filename), api2idx)
            if(length >= config.sequence_length_min and count >= config.api_count_min):
                pid_api_sequences.append(pid_api_sequence_xml)
                base_sequences.append(base_sequence_xml)
                status.append(status_xml)
                tpg_datas.append(tpg_data)
                tag_datas.append(tag_data)
                datanames.append(str(os.path.join(rootpath, filename)))
                labels.append(0 if file_type == 'black' else 1)
            
    idx2api = {value: key for key, value in api2idx.items()}

    return pid_api_sequences, base_sequences, status, datanames, tpg_datas, tag_datas, labels, api2idx, idx2api, None

def load_api_sequence(config):
    return load_api_sequence_cuck(config)


    
