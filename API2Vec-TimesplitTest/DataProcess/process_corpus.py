#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File         :   process_corpus.py
@Desc         :   process corpus
'''

import bisect
from cmath import inf
import math
import random
import numpy as np
from tqdm import tqdm
import networkx as nx
import multiprocessing
import pandas as pd

def conduct_graph(edges, params=True):
    graph = nx.DiGraph()

    for edge in edges:
        api_b, api_e = edge
        if(params):
            graph.add_edge(api_b, api_e, walkcount=1)
        else:
            graph.add_edge(api_b, api_e)

    return graph

def tag_random_walk(graph, timedict, sequence, time, step=49, count_limit_flag=True):

    if(step == 0):
        return sequence, time
    if(count_limit_flag):
        walk_counts = [params['walkcount'] for edge, params in graph.edges.items()]
        max_walk_count = max(walk_counts)
        min_walk_count = min(walk_counts)

    try:
        neighbors = graph[sequence[-1]].items()
        nxt_nodes = [] # 下一个节点集合
        weights = [] # 目标节点的权重

        for node, params in neighbors:
            edge = (sequence[-1], node)
            walk_count = params['walkcount'] if count_limit_flag else 0
            time_sequence = timedict[edge]

            # TIME Filter
            if(time_sequence[-1] <= time): continue
            
            filter_timelist = time_sequence[bisect.bisect(time_sequence, time):]
            freq = len(filter_timelist)
            time_dis = filter_timelist[0] - time

            # 根据时间跨度筛选节点，时间跨度过大的节点以较高的概率不游走
            choice_flag = random.choices([0, 1], weights=[time_dis / step, 2.])[0]
            if(choice_flag == 1 and max_walk_count != min_walk_count and count_limit_flag):
                choice_flag = random.choices([0, 1], weights=[0.1, (max_walk_count - walk_count) / (max_walk_count - min_walk_count) + 0.1])[0]
                # 当 walk_count * 2 == max_walk_count 时 为平衡，走不走该点的概率一致
                # 游走次数小于该平衡点的节点以较大的概率游走
                # 游走次数大于该平衡点的节点以较小的概率游走

            if(choice_flag == 1):
                nxt_nodes.append(node)
                weights.append(freq / (math.sqrt(time_dis) + math.sqrt(walk_count)))
    except:
        nxt_nodes = []
    if len(nxt_nodes) == 0: 
        return sequence, time
    else:
        # 选择第一个大于 time 的时间
        nxt_node = random.choices(nxt_nodes, weights=weights)[0]
        timelist = timedict[(sequence[-1], nxt_node)]
        filter_timelist = timelist[bisect.bisect(timelist, time):]
        nxt_time = random.choices(filter_timelist, weights=[1 / (exe_time - time) for exe_time in filter_timelist])[0]

    graph[sequence[-1]][nxt_node]['walkcount'] += 1
    return tag_random_walk(graph, timedict, (*sequence, nxt_node), nxt_time, step = step-1)

def tpg_random_walk(tpg_extime_dict, tpg_node, time):

    nxt_tpg_nodes = []
    nxt_pid_weights = []
    for pid, time_sequence in tpg_extime_dict.items():
        
        if(pid == tpg_node): continue
        if(time_sequence[-1] <= time): continue

        filter_timelist = time_sequence[bisect.bisect(time_sequence, time):]
        time_dis = filter_timelist[0] - time

        nxt_tpg_nodes.append(pid)
        nxt_pid_weights.append(len(filter_timelist) / time_dis)

    if(len(nxt_tpg_nodes) == 0):
        return None
    else:
        return random.choices(nxt_tpg_nodes, weights=nxt_pid_weights)[0]

def random_walk_main_infos(inputs):
    config, sequences, tpg_extime_dict, tag_graphs, tag_edget_dict, tag_extime_dict = inputs
    paths = []
    
    k, step = config.rw_k, config.rw_step
    count_limit_flag = True if 'count' in config.params else False

    # Statistic
    s_tag_node = []
    s_tag_edge = []
    s_tag_edge_unique = []

    s_tag_walk_count = []

    s_tag_node_cover = []
    s_tag_edge_cover = []
    s_tag_path_len = []


    tpg_nodes = list(tpg_extime_dict.keys())
    for i in range(k): # 总重复次数, 满足概率覆盖
        for tpg_node in tpg_nodes: # 遍历每一个 TPG 节点
            tag_graph, tag_graph_timedict = tag_graphs[tpg_node], tag_edget_dict[tpg_node]
            tag_nodes = list(tag_graph.nodes)
            if(i == 0):
                s_tag_node.append(len(tag_nodes)) # Statistic
                s_tag_edge.append(sum([len(value) for edge, value in tag_graph_timedict.items()])) # Statistic
                s_tag_edge_unique.append(len(tag_graph_timedict)) # Statistic

            for tag_node in tag_nodes: # 遍历每一个 TAG 节点
                tag_graph, tag_graph_timedict = tag_graphs[tpg_node], tag_edget_dict[tpg_node]
                
                time = tag_extime_dict[tag_node] # 不同 TAG 节点对应的游走起始时间不同，初始化映射解决
                
                path_item = [] # 存储 TAG 上游走的序列
                start_tpg_node = tpg_node
                start_tag_node = tag_node

                s_tag_walk_count_item = 0 # statistic
                while True: # 一直游走直到时间结束
                    
                    path, nxt_time = tag_random_walk(tag_graph, tag_graph_timedict, (start_tag_node, ), time, step = step, count_limit_flag=count_limit_flag)

                    if path:
                        path_item.extend(path)
                        time = nxt_time
                        s_tag_walk_count_item += 1 # statistic
                        s_item_node = len(set(path))
                        s_item_nodes = len(tag_graph.nodes)
                        s_item_edge = len(set(zip(path, path[1:])))
                        s_item_edges = len(tag_graph_timedict)

                        s_tag_node_cover.append(s_item_node/s_item_nodes)
                        s_tag_edge_cover.append(s_item_edge/s_item_edges)
                        s_tag_path_len.append(len(path))
                    else:
                        raise ValueError("算法处理下 应当不存在空 PATH") # 算法处理下 应当不存在空 PATH
                    
                    next_tpg_node = tpg_random_walk(tpg_extime_dict, start_tpg_node, time)

                    if(next_tpg_node is None): # PID 无路可走 
                        break
                    
                    # PID 一旦可走就代表 TAG 中有路可走，因此 TAG 生成的 PATH 应该都是有效的
                    # 更新 TAG
                    tag_graph, tag_graph_timedict = tag_graphs[next_tpg_node], tag_edget_dict[next_tpg_node]

                    # 筛选 TAG 中起始节点
                    nxt_nodes = []
                    nxt_weights = []
                    for edge, time_sequence in tag_graph_timedict.items():
                        if(time_sequence[-1] <= time+1): continue # 过滤过时节点
                        filter_timelist = time_sequence[bisect.bisect(time_sequence, time+1):] # 获取可用时间序列 这里要筛选的是起始节点，所有用 time + 1 进行过滤 node_e 那么 node_s 就可达
                        node_s, node_e = edge
                        nxt_nodes.append(node_s)
                        nxt_weights.append(len(filter_timelist) / (filter_timelist[0] - time - 1)) # 频率 / 时间跨度
                    
                    if(len(nxt_nodes) == 0):
                        break
                    else:
                        start_tpg_node = next_tpg_node
                        start_tag_node = random.choices(nxt_nodes, weights=nxt_weights)[0]
                        nxt_time = inf
                        for edge, time_sequence in tag_graph_timedict.items():
                            if(time_sequence[-1] <= time+1): continue
                            node_s, node_e = edge
                            if(node_s != start_tag_node): continue
                            nxt_time = min(nxt_time, time_sequence[bisect.bisect(time_sequence, time+1)] - 1)
                        if(nxt_time != inf): time = nxt_time
                s_tag_walk_count.append(s_tag_walk_count_item)
                if(len(path_item) >= 5):
                    paths.append(path_item) # 一次完全的游走路径
    if(len(paths) == 0):
        paths.append(sequences)

    # return paths
    return (s_tag_node, s_tag_edge, s_tag_edge_unique, s_tag_walk_count, s_tag_node_cover, s_tag_edge_cover, s_tag_path_len)

def random_walk_main(inputs):
    config, sequences, tpg_extime_dict, tag_graphs, tag_edget_dict, tag_extime_dict = inputs
    paths = []
    
    k, step = config.rw_k, config.rw_step
    count_limit_flag = True if 'count' in config.params else False

    tpg_nodes = list(tpg_extime_dict.keys())
    for i in range(k): # 总重复次数, 满足概率覆盖
        for tpg_node in tpg_nodes: # 遍历每一个 TPG 节点
            tag_graph, tag_graph_timedict = tag_graphs[tpg_node], tag_edget_dict[tpg_node]
            tag_nodes = list(tag_graph.nodes)
            for tag_node in tag_nodes: # 遍历每一个 TAG 节点
                tag_graph, tag_graph_timedict = tag_graphs[tpg_node], tag_edget_dict[tpg_node]
                
                time = tag_extime_dict[tag_node] # 不同 TAG 节点对应的游走起始时间不同，初始化映射解决
                
                path_item = [] # 存储 TAG 上游走的序列
                start_tpg_node = tpg_node
                start_tag_node = tag_node

                while True: # 一直游走直到时间结束
                    
                    path, nxt_time = tag_random_walk(tag_graph, tag_graph_timedict, (start_tag_node, ), time, step = step, count_limit_flag=count_limit_flag)

                    if path:
                        path_item.extend(path)
                        time = nxt_time
                    else:
                        raise ValueError("算法处理下 应当不存在空 PATH") # 算法处理下 应当不存在空 PATH
                    
                    next_tpg_node = tpg_random_walk(tpg_extime_dict, start_tpg_node, time)

                    if(next_tpg_node is None): # PID 无路可走 
                        break
                    
                    # PID 一旦可走就代表 TAG 中有路可走，因此 TAG 生成的 PATH 应该都是有效的
                    # 更新 TAG
                    tag_graph, tag_graph_timedict = tag_graphs[next_tpg_node], tag_edget_dict[next_tpg_node]

                    # 筛选 TAG 中起始节点
                    nxt_nodes = []
                    nxt_weights = []
                    for edge, time_sequence in tag_graph_timedict.items():
                        if(time_sequence[-1] <= time+1): continue # 过滤过时节点
                        filter_timelist = time_sequence[bisect.bisect(time_sequence, time+1):] # 获取可用时间序列 这里要筛选的是起始节点，所有用 time + 1 进行过滤 node_e 那么 node_s 就可达
                        node_s, node_e = edge
                        nxt_nodes.append(node_s)
                        nxt_weights.append(len(filter_timelist) / (filter_timelist[0] - time - 1)) # 频率 / 时间跨度
                    
                    if(len(nxt_nodes) == 0):
                        break
                    else:
                        start_tpg_node = next_tpg_node
                        start_tag_node = random.choices(nxt_nodes, weights=nxt_weights)[0]
                        nxt_time = inf
                        for edge, time_sequence in tag_graph_timedict.items():
                            if(time_sequence[-1] <= time+1): continue
                            node_s, node_e = edge
                            if(node_s != start_tag_node): continue
                            nxt_time = min(nxt_time, time_sequence[bisect.bisect(time_sequence, time+1)] - 1)
                        if(nxt_time != inf): time = nxt_time
                if(len(path_item) >= 5):
                    paths.append(path_item) # 一次完全的游走路径

    if(len(paths) == 0):
        paths.append(sequences)

    return paths

def analysis_infos(infos, title="Global"):
    tag_infos = {
        'tag_node': [],
        'tag_edge': [],
        'tag_edge_unique': []
    }
    tag_walk_count_infos = {
        'tag_walk_count': []
    }
    tag_rw_infos = {
        'tag_node_cover': [],
        'tag_edge_cover': [],
        'tag_path_len': []
    }
    for info in infos:
        (s_tag_node, s_tag_edge, s_tag_edge_unique, s_tag_walk_count, s_tag_node_cover, s_tag_edge_cover, s_tag_path_len) = info
        tag_infos['tag_node'].extend(s_tag_node)
        tag_infos['tag_edge'].extend(s_tag_edge)
        tag_infos['tag_edge_unique'].extend(s_tag_edge_unique)

        tag_walk_count_infos['tag_walk_count'].extend(s_tag_walk_count)

        tag_rw_infos['tag_node_cover'].extend(s_tag_node_cover)
        tag_rw_infos['tag_edge_cover'].extend(s_tag_edge_cover)
        tag_rw_infos['tag_path_len'].extend(s_tag_path_len)

    pd_tag_infos = pd.DataFrame(tag_infos)
    pd_tag_walk_count_infos = pd.DataFrame(tag_walk_count_infos)
    pd_tag_rw_infos = pd.DataFrame(tag_rw_infos)

    pd_tag_infos.to_excel(f'./Analysis/RWinfos/{title}_tag_infos.xlsx', index=False)
    pd_tag_walk_count_infos.to_excel(f'./Analysis/RWinfos/{title}_tag_walk_count_infos.xlsx', index=False)
    pd_tag_rw_infos.to_excel(f'./Analysis/RWinfos/{title}_tag_rw_infos.xlsx', index=False)
    with open('./Analysis/RWinfos/result.txt', 'a', encoding='utf-8') as f:
        f.write(f'\n\n{title}\n\n')
        f.write(f'{pd_tag_infos.describe()}\n\n')
        f.write(f'{pd_tag_walk_count_infos.describe()}\n\n')
        f.write(f'{pd_tag_rw_infos.describe()}\n\n')
    

def load_corpus_pid(config, pid_api_seqs, tpg_datas, tag_datas, labels):

    corpus = []

    title = -1

    inputs = []
    for pid_api_seq, tpg_data, tag_data, label in tqdm(zip(pid_api_seqs, tpg_datas, tag_datas, labels), total=len(tpg_datas)):

        if(label != title and title != -1): continue
        
        tpg_extime_dict = tpg_data
        tag_edges, tag_edget_dict, tag_extime_dict = tag_data
        
        tag_graphs = { pid: conduct_graph(edges) for pid, edges in tag_edges.items()}

        sequences = []
        for item in pid_api_seq:
            sequences.extend(item)
        inputs.append((config, sequences, tpg_extime_dict, tag_graphs, tag_edget_dict, tag_extime_dict))

    with multiprocessing.Pool(processes=8) as p:
        corpus = list(tqdm(p.imap(random_walk_main, inputs), total=len(inputs), desc='游走进度'))

    # flag = 10 if title == -1 else 5
    # inputs = random.sample(inputs, int(len(inputs) / flag))
    # with multiprocessing.Pool(processes=12) as p:
    #     infos = list(tqdm(p.imap(random_walk_main_infos, inputs), total=len(inputs), desc='游走进度'))    
    # analysis_infos(infos, title=('Global' if title == -1 else ('Black' if title == 0 else 'White')))
    # exit(0)
    return corpus

def create_alias_table(normalized_prob):
    length = len(normalized_prob)
    accept, alias = [0] * length, [0] * length
    
    # small,big 存放比1小和比1大的索引
    small, big = [], []
    # 归一化转移概率 * 转移概率数
    transform_N = np.array(normalized_prob) * length
    # 根据概率放入small large
    for i, prob in enumerate(transform_N):
        if prob < 1.0:
            small.append(i)
        else:
            big.append(i)
 
    while small and big:
        small_idx, large_idx = small.pop(), big.pop()
        accept[small_idx] = transform_N[small_idx] # 接收这个点的概率
        alias[small_idx] = large_idx # 竞争的 large_idx
        transform_N[large_idx] = transform_N[large_idx] - (1 - transform_N[small_idx])
        if np.float32(transform_N[large_idx]) < 1.:
            small.append(large_idx)
        else:
            big.append(large_idx)
 
    while big:
        large_idx = big.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1
    return accept, alias

def get_alias_edge(G, p, q, t, v):

    unnormalized_probs = []    
    for x in G.neighbors(v):        
        weight = 1.0
        if x == t:# d_tx == 0            
            unnormalized_probs.append(weight/p)        
        elif G.has_edge(x, t):# d_tx == 1            
            unnormalized_probs.append(weight)        
        else:# d_tx == 2            
            unnormalized_probs.append(weight/q)    
    norm_const = sum(unnormalized_probs)    
    normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
    return create_alias_table(normalized_probs)

def preprocess_transition_probs(G, p, q):
    alias_nodes = {}    
    for node in G.nodes():        
        unnormalized_probs = [1.0 for nbr in G.neighbors(node)]        
        norm_const = sum(unnormalized_probs)        
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]                 
        alias_nodes[node] = create_alias_table(normalized_probs)
    alias_edges = {}
    for edge in G.edges():        
        alias_edges[edge] = get_alias_edge(G, p, q, edge[0], edge[1])
    
    return alias_nodes, alias_edges

def alias_sample(accept, alias):
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

def random_walk_node2vec_once(start_node, walk_length, G, p, q):
    alias_nodes, alias_edges = preprocess_transition_probs(G, p, q)
    walk = [start_node]
    while len(walk) < walk_length:        
        cur = walk[-1]
        cur_nbrs = list(G.neighbors(cur))     
        if len(cur_nbrs) > 0:
            if len(walk) == 1:                
                walk.append(cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])            
            else:                
                prev = walk[-2]                
                edge = (prev, cur)                
                next_node = cur_nbrs[alias_sample(alias_edges[edge][0],alias_edges[edge][1])]                
                walk.append(next_node)        
        else:            
            break
    return walk
def random_walk_node2vec(inputs):
    config, sequences, item_graph = inputs
    p = config.p
    q = config.q
    r = config.r
    l = config.l

    paths = []
    for _ in range(r):
        for start_node in item_graph.nodes():
            walk_length = l 
            path = random_walk_node2vec_once(start_node, walk_length, item_graph, p, q)
            if(len(path) <= 3): continue
            paths.append(path)
    if(len(paths) == 0):
        paths.append(sequences[:l])
    return paths
    

def load_corpus_node2vec(config, pid_api_seqs, graph_datas):
    corpus = []

    inputs = []
    for pid_api_seq, graph_data in tqdm(zip(pid_api_seqs, graph_datas), total=len(graph_datas)):
        
        _, edges = graph_data

        item_graph = conduct_graph(edges)
        
        sequences = []
        for item in pid_api_seq:
            sequences.extend(item)
        inputs.append((config, sequences, item_graph))
    with multiprocessing.Pool(processes=16) as p:
        corpus = list(tqdm(p.imap(random_walk_node2vec, inputs), total=len(inputs), desc='Node2Vec 游走进度'))
    
    return corpus
def load_corpus(config, pid_api_seqs, tpg_datas, tag_datas, labels):
    if(config.pid_flag == 'pid'):
        return load_corpus_pid(config, pid_api_seqs, tpg_datas, tag_datas, labels)
    elif(config.pid_flag == 'node2vec'):
        return load_corpus_node2vec(config, pid_api_seqs, tpg_datas)
    