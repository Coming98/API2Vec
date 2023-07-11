#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   process_api_sequence.py
@Desc         :   process api sequence
'''

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
import random

def attack_insert(config, api_names, pids, status, insert_times):
    attack_patterns = random.choices(config.attack_patterns, k=insert_times)
    insert_indexs = sorted(random.choices(list(range(len(api_names))), k=insert_times))
    insert_indexs.append(len(api_names))

    pieces = []

    insert_pids = []

    pre_index = 0
    for idx in insert_indexs:
        pieces.append([api_names[pre_index:idx], pids[pre_index:idx], status[pre_index:idx]])
        if(idx == len(pids)): idx = idx - 1
        insert_pids.append(pids[idx])
        pre_index = idx

    assert len(attack_patterns) == len(pieces) - 1, "Wrong split pieces"

    ret_api_names, ret_pids, ret_status = [], [], []
    for piece, insert_pid, attack_pattern in zip(pieces[:-1], insert_pids[:-1], attack_patterns):
        api_names_piece, pids_piece, status_piece = piece

        ret_api_names.extend(api_names_piece)
        ret_pids.extend(pids_piece)
        ret_status.extend(status_piece)

        attack_api_names_piece = [name[:-2] for name in attack_pattern]
        attack_pids_piece = [insert_pid] * len(attack_api_names_piece)
        attack_status_piece = [('SUCCESS' if name[-2:] == '_S' else 'FAILURE') for name in attack_pattern]

        ret_api_names.extend(attack_api_names_piece)
        ret_pids.extend(attack_pids_piece)
        ret_status.extend(attack_status_piece)
    ret_api_names.extend(pieces[-1][0])
    ret_pids.extend(pieces[-1][1])
    ret_status.extend(pieces[-1][2])

    return ret_api_names, ret_pids, ret_status
    
def init_data(config, filepath, attack=False):
    if(config.data_name == 'cuck'):
        data = pd.read_excel(filepath)
        data = data.sort_values('timestamp')
        api_names = data[config.api_header]
        pids = data[config.pid_header]
        status = data[config.status_header]
        status = [True if item == 'SUCCESS' else False for item in list(status)]

    if(attack):
        length = len(api_names)
        insert_times = min(5, max(1, length // 10))
        api_names, pids, status = attack_insert(config, api_names, pids, status, insert_times)

    return api_names, pids, status

def load_api_sequence_nopid(config, filepath, api2idx, attack=False):

    pid_api_sequences = []
    pid_sequences = []
    pid_apisequences = {}

    base_sequences = []
    
    api_names, pids, status = init_data(config, filepath, attack=attack)
        
    for api_name, pid, state in zip(api_names, pids, status):
        if('status' in config.params):
            api_name += ('_S' if state else '_F')
        if(api_name not in api2idx):
            api2idx[api_name] = len(api2idx)
        api_idx = api2idx[api_name]
        base_sequences.append(api_idx)

        if(pid not in pid_sequences):
            pid_sequences.append(pid)
            pid_apisequences[pid] = [api_idx, ]
        else:
            pid_apisequences[pid].append(api_idx)

    pid_api_sequences = [pid_apisequences[pid] for pid in pid_sequences]
    return pid_api_sequences, base_sequences, status, len(api_names), len(set(api_names))

def load_api_sequence_pid(config, filepath, api2idx, attack=False):

    pid_api_sequences = []
    pid_sequences = [] # 按照 pid 记录原始序列
    pid_apisequences = {}

    base_sequences = []

    tpg_apisequence_dict = {} # pid: api_sequence
    pid_apit_dict = {} # pid: apit_sequence
    tag_extime_dict = {} # tag: extime

    api_names, pids, status = init_data(config, filepath, attack=attack)

    for t, (api_name, pid, state) in enumerate(zip(api_names, pids, status), 1):
        if('status' in config.params):
            api_name += ('_S' if state else '_F')
        if(api_name not in api2idx): api2idx[api_name] = len(api2idx)
        api_index = api2idx[api_name]
        base_sequences.append(api_index)

        # pid - 原始序列
        if(pid not in pid_sequences):
            pid_sequences.append(pid)
            pid_apisequences[pid] = [api_index, ]
        else:
            pid_apisequences[pid].append(api_index)

        # tag_node - min(extime)
        if(api_index not in tag_extime_dict): tag_extime_dict[api_index] = t

        # pid - api sequence
        # pid - time sequence
        if(pid in tpg_apisequence_dict):
            tpg_apisequence_dict[pid].append(api_index) # 按照 PID 划分 API calls
            pid_apit_dict[pid].append(t) # 同时记录每个 API call 的执行时间
        else:
            tpg_apisequence_dict[pid] = [api_index, ]
            pid_apit_dict[pid] = [t, ]
        

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

    pid_api_sequences = [pid_apisequences[pid] for pid in pid_sequences]
    return pid_api_sequences, base_sequences, status, pid_apit_dict, (tag_edges, tag_edget_dict, tag_extime_dict), len(api_names), len(set(api_names))

def load_api_sequence_node2vec(config, filepath, api2idx, attack=False):

    pid_api_sequences = []
    pid_sequences = [] # 按照 pid 记录原始序列
    pid_apisequences = {}

    base_sequences = []

    graph_data = {} # nodes, edges

    api_names, pids, status = init_data(config, filepath, attack=attack)

    for api_name, pid, state in zip(api_names, pids, status):
        if('status' in config.params):
            api_name += ('_S' if state else '_F')
        if(api_name not in api2idx): api2idx[api_name] = len(api2idx)
        api_index = api2idx[api_name]
        base_sequences.append(api_index)

        # pid - 原始序列
        if(pid not in pid_sequences):
            pid_sequences.append(pid)
            pid_apisequences[pid] = [api_index, ]
        else:
            pid_apisequences[pid].append(api_index)

    graph_data = (list(set(base_sequences)), list(set(zip(base_sequences, base_sequences[1:]))))

    pid_api_sequences = [pid_apisequences[pid] for pid in pid_sequences]
    return pid_api_sequences, base_sequences, status, graph_data, len(api_names), len(set(api_names))

def load_api_sequence_cuck(config):
    api2idx = {'<UNK>': 0}
    idx2api = {}

    pid_api_sequences = []
    base_sequences = []
    status = []
    datanames = []
    labels = []
    
    tag_datas = [] # edges, edge_time_dict, node_extime_min
    tpg_datas = [] # edges, edge_time_dict

    graph_datas = [] # node2vec + deepwalk

    for rootpath, _, filenames in os.walk(config.data_dir): 
        
        file_type = os.path.split(rootpath)[-1] 
        if(file_type not in ('black', 'white')): continue

        for filename in tqdm(filenames):

            if(config.pid_flag == 'nopid'):
                pid_api_sequence_xml, base_sequence_xml, status_xml, length, count = load_api_sequence_nopid(config, os.path.join(rootpath, filename), api2idx)
                if(length >= config.sequence_length_min and count >= config.api_count_min):
                    pid_api_sequences.append(pid_api_sequence_xml)
                    base_sequences.append(base_sequence_xml)
                    status.append(status_xml)
                    datanames.append(str(os.path.join(rootpath, filename)))
                    labels.append(1 if file_type == 'black' else 0)
                if(config.task_name == 'attack' and filename[:-5] in config.against_sample_names and file_type == 'black'):
                    pid_api_sequence_xml, base_sequence_xml, status_xml, length, count = load_api_sequence_nopid(config, os.path.join(rootpath, filename), api2idx, attack=True)
                    if(length >= config.sequence_length_min and count >= config.api_count_min):
                        pid_api_sequences.append(pid_api_sequence_xml)
                        base_sequences.append(base_sequence_xml)
                        status.append(status_xml)
                        datanames.append(str(os.path.join(rootpath, filename[:-5] + '_A.xlsx')))
                        labels.append(1 if file_type == 'black' else 0)
            elif(config.pid_flag == 'pid'):
                # (tpg_edges, tpg_edget_dict), (tag_edges, tag_edget_dict, tag_extime_dict)
                pid_api_sequence_xml, base_sequence_xml, status_xml, tpg_data, tag_data, length, count = load_api_sequence_pid(config, os.path.join(rootpath, filename), api2idx)
                if(length >= config.sequence_length_min and count >= config.api_count_min):
                    pid_api_sequences.append(pid_api_sequence_xml)
                    base_sequences.append(base_sequence_xml)
                    status.append(status_xml)
                    tpg_datas.append(tpg_data)
                    tag_datas.append(tag_data)
                    datanames.append(str(os.path.join(rootpath, filename)))
                    labels.append(1 if file_type == 'black' else 0)
                if(config.task_name == 'attack' and filename[:-5] in config.against_sample_names):
                    pid_api_sequence_xml, base_sequence_xml, status_xml, tpg_data, tag_data, length, count = load_api_sequence_pid(config, os.path.join(rootpath, filename), api2idx, attack=True)
                    if(length >= config.sequence_length_min and count >= config.api_count_min):
                        pid_api_sequences.append(pid_api_sequence_xml)
                        base_sequences.append(base_sequence_xml)
                        status.append(status_xml)
                        tpg_datas.append(tpg_data)
                        tag_datas.append(tag_data)
                        datanames.append(str(os.path.join(rootpath, filename[:-5] + '_A.xlsx')))
                        labels.append(0 if file_type == 'back' else 1)
            elif(config.pid_flag == 'node2vec'):
                pid_api_sequence_xml, base_sequence_xml, status_xml, graph_data, length, count = load_api_sequence_node2vec(config, os.path.join(rootpath, filename), api2idx)
                if(length >= config.sequence_length_min and count >= config.api_count_min):
                    pid_api_sequences.append(pid_api_sequence_xml)
                    base_sequences.append(base_sequence_xml)
                    status.append(status_xml)
                    graph_datas.append(graph_data)
                    datanames.append(str(os.path.join(rootpath, filename)))
                    labels.append(0 if file_type == 'black' else 1)
                if(config.task_name == 'attack' and filename[:-5] in config.against_sample_names and file_type == 'black'):
                    pid_api_sequence_xml, base_sequence_xml, status_xml, graph_data, length, count = load_api_sequence_node2vec(config, os.path.join(rootpath, filename), api2idx, attack=True)
                    if(length >= config.sequence_length_min and count >= config.api_count_min):
                        pid_api_sequences.append(pid_api_sequence_xml)
                        base_sequences.append(base_sequence_xml)
                        status.append(status_xml)
                        graph_datas.append(graph_data)
                        datanames.append(str(os.path.join(rootpath, filename[:-5] + '_A.xlsx')))
                        labels.append(0 if file_type == 'black' else 1)
            else:
                print("Wrong pid flag!")
                exit(0)
            


    idx2api = {value: key for key, value in api2idx.items()}

    if(config.pid_flag == 'nopid'):
        return pid_api_sequences, base_sequences, status, datanames, labels, api2idx, idx2api, None
    elif(config.pid_flag == 'pid'):
        return pid_api_sequences, base_sequences, status, datanames, tpg_datas, tag_datas, labels, api2idx, idx2api, None
    elif(config.pid_flag == 'node2vec'):
        return pid_api_sequences, base_sequences, status, datanames, graph_datas, labels, api2idx, idx2api, None

def load_api_sequence(config):

    if(config.data_name == 'cuck'):
        return load_api_sequence_cuck(config)


    
