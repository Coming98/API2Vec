import bisect
from cmath import inf
import math
import random
import networkx as nx

def tag_random_walk(graph, timedict, sequence, step=49, count_limit_flag=True):

    if(step == 0):
        return sequence
    walk_counts = [params['walkcount'] for edge, params in graph.edges.items()]
    max_walk_count = max(walk_counts)
    min_walk_count = min(walk_counts)

    try:
        neighbors = graph[sequence[-1]].items()
        nxt_nodes = [] # 下一个节点集合
        weights = [] # 目标节点的权重

        for node, params in neighbors:
            edge = (sequence[-1], node)
            walk_count = params['walkcount']
            time_sequence = timedict[edge]

            freq = len(time_sequence)
            if(max_walk_count != min_walk_count):
                choice_flag = random.choices([0, 1], weights=[0.1, (max_walk_count - walk_count) / (max_walk_count - min_walk_count) + 0.1])[0]
                # 当 walk_count * 2 == max_walk_count 时 为平衡，走不走该点的概率一致
                # 游走次数小于该平衡点的节点以较大的概率游走
                # 游走次数大于该平衡点的节点以较小的概率游走
            else:
                choice_flag = 1

            if(choice_flag == 1):
                nxt_nodes.append(node)
                weights.append(freq / (2 * math.sqrt(walk_count)))
    except:
        nxt_nodes = []
    if len(nxt_nodes) == 0: 
        return sequence
    else:
        # 选择第一个大于 time 的时间
        nxt_node = random.choices(nxt_nodes, weights=weights)[0]

    graph[sequence[-1]][nxt_node]['walkcount'] += 1
    return tag_random_walk(graph, timedict, (*sequence, nxt_node), step = step-1)

def tpg_random_walk(tpg_extime_dict, tpg_node):

    nxt_tpg_nodes = []
    nxt_pid_weights = []
    for pid, time_sequence in tpg_extime_dict.items():
        
        if(pid == tpg_node): continue
        if len(time_sequence):
            nxt_tpg_nodes.append(pid)
            nxt_pid_weights.append(len(time_sequence))

    if(len(nxt_tpg_nodes) == 0):
        return tpg_node
    else:
        return random.choices(nxt_tpg_nodes, weights=nxt_pid_weights)[0]

def roam_without_time(inputs):
    config, sequences, tpg_extime_dict, tag_graphs, tag_edget_dict, tag_extime_dict = inputs
    
    tpg_nodes = list(tpg_extime_dict.keys())
    k, step = 3, 49
    paths = []
    total_step = len(sequences) # 总长度限制
    count = 0
    while count < k:
        tpg_node = tpg_nodes[0]
        tag_graph, tag_graph_timedict = tag_graphs[tpg_node], tag_edget_dict[tpg_node]
        tag_nodes = list(tag_graph.nodes)
        if(len(tag_nodes) == 0): continue
        path_item = [] # 存储 TAG 上游走的序列
        tag_node = tag_nodes[0]
        
        start_tpg_node = tpg_node
        start_tag_node = tag_node

        while len(path_item) < total_step: # 一直游走直到时间结束
            
            path = tag_random_walk(tag_graph, tag_graph_timedict, (start_tag_node, ), step = step)

            if path:
                path_item.extend(path)
            else:
                raise ValueError("算法处理下 应当不存在空 PATH") # 算法处理下 应当不存在空 PATH
            
            next_tpg_node = tpg_random_walk(tpg_extime_dict, start_tpg_node)

            # PID 一旦可走就代表 TAG 中有路可走，因此 TAG 生成的 PATH 应该都是有效的
            # 更新 TAG

            if next_tpg_node and len(list(tag_graphs[next_tpg_node].nodes)):
                tag_graph, tag_graph_timedict = tag_graphs[next_tpg_node], tag_edget_dict[next_tpg_node]
                start_tpg_node = next_tpg_node
                start_tag_node = list(tag_graph.nodes)[0]
            else:
                break
        if len(path_item) > 5:
            paths.append(path_item)
        count += 1

    if(len(paths) == 0):
        paths.append(sequences)

    return paths

def roam_without_time_analysis(inputs):
    config, sequences, tpg_extime_dict, tag_graphs, tag_edget_dict, tag_extime_dict = inputs
    
    tpg_nodes = list(tpg_extime_dict.keys())
    k, step = 3, 49
    total_step = len(sequences) # 总长度限制
    count = 0

    s_tag_walk_count = []

    s_tag_node_cover = []
    s_tag_edge_cover = []
    s_tag_path_len = []

    g_edge_cover = []

    while count < k:
        cover_info = {} # 游走完成后整个图的覆盖情况

        tpg_node = tpg_nodes[0]
        tag_graph, tag_graph_timedict = tag_graphs[tpg_node], tag_edget_dict[tpg_node]
        tag_nodes = list(tag_graph.nodes)
        if(len(tag_nodes) == 0): continue
        path_item = [] # 存储 TAG 上游走的序列
        tag_node = tag_nodes[0]
        
        start_tpg_node = tpg_node
        start_tag_node = tag_node

        s_tag_walk_count_item = 0
        while len(path_item) < total_step: # 一直游走直到时间结束
            if start_tpg_node not in cover_info:
                cover_info[start_tpg_node] = {
                    'total_edge': len(tag_graph_timedict), 
                    'edge_cover': set(),
                }
            path = tag_random_walk(tag_graph, tag_graph_timedict, (start_tag_node, ), step = step)

            if path:
                path_item.extend(path)
                _ = [ cover_info[start_tpg_node]['edge_cover'].add(f'{u},{v}') for u, v in zip(path, path[1:]) ]
                s_item_node = len(set(path))
                s_item_nodes = len(tag_graph.nodes)
                s_item_edge = len(set(zip(path, path[1:])))
                s_item_edges = len(tag_graph_timedict)

                s_tag_node_cover.append(s_item_node/s_item_nodes)
                s_tag_edge_cover.append(s_item_edge/s_item_edges)
                s_tag_path_len.append(len(path))
            else:
                raise ValueError("算法处理下 应当不存在空 PATH") # 算法处理下 应当不存在空 PATH
            
            next_tpg_node = tpg_random_walk(tpg_extime_dict, start_tpg_node)

            # PID 一旦可走就代表 TAG 中有路可走，因此 TAG 生成的 PATH 应该都是有效的
            # 更新 TAG

            if next_tpg_node and len(list(tag_graphs[next_tpg_node].nodes)):
                tag_graph, tag_graph_timedict = tag_graphs[next_tpg_node], tag_edget_dict[next_tpg_node]
                if next_tpg_node != start_tpg_node:
                    s_tag_walk_count_item += 1 # statistic
                start_tpg_node = next_tpg_node
                start_tag_node = list(tag_graph.nodes)[0]
            else:
                break
        s_tag_walk_count.append(s_tag_walk_count_item)
        
        count += 1

        total, cover = 0, 0
        for tpg_node in cover_info:
            total += cover_info[tpg_node]['total_edge']
            cover += len(cover_info[tpg_node]['edge_cover'])
        g_edge_cover.append(cover / total)


    return (g_edge_cover, s_tag_walk_count, s_tag_node_cover, s_tag_edge_cover, s_tag_path_len)
    
