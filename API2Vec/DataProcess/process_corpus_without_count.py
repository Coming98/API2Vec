import bisect
from cmath import inf
import math
import random
import networkx as nx

def tag_random_walk(graph, timedict, sequence, time, step=49):

    if(step == 0):
        return sequence, time

    try:
        neighbors = graph[sequence[-1]].items()
        nxt_nodes = [] # 下一个节点集合
        weights = [] # 目标节点的权重

        for node, params in neighbors:
            edge = (sequence[-1], node)
            time_sequence = timedict[edge]

            # TIME Filter
            if(time_sequence[-1] <= time): continue
            
            filter_timelist = time_sequence[bisect.bisect(time_sequence, time):]
            freq = len(filter_timelist)
            time_dis = filter_timelist[0] - time

            # 根据时间跨度筛选节点，时间跨度过大的节点以较高的概率不游走
            choice_flag = random.choices([0, 1], weights=[time_dis / step, 2.])[0]

            if(choice_flag == 1):
                nxt_nodes.append(node)
                weights.append(freq / (math.sqrt(time_dis) * 2))
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

def roam_without_count(inputs):
    config, sequences, tpg_extime_dict, tag_graphs, tag_edget_dict, tag_extime_dict = inputs
    paths = []
    
    k, step = config.rw_k, config.rw_step

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
                    
                    path, nxt_time = tag_random_walk(tag_graph, tag_graph_timedict, (start_tag_node, ), time, step = step)

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

def roam_without_count_analysis(inputs):
    """
        节点的覆盖率不变
        重点查看
        1. 整体边的覆盖率
        2. TAG 边的长度
        3. TAG 游走一次的边的覆盖率
    """
    config, sequences, tpg_extime_dict, tag_graphs, tag_edget_dict, tag_extime_dict = inputs
    k, step = config.rw_k, config.rw_step

    s_tag_walk_count = []

    s_tag_node_cover = []
    s_tag_edge_cover = []
    s_tag_path_len = []

    g_edge_cover = []
    
    tpg_nodes = list(tpg_extime_dict.keys())
    for i in range(k): # 总重复次数, 满足概率覆盖
        cover_info = {} # 游走完成后整个图的覆盖情况
        for tpg_node in tpg_nodes: # 遍历每一个 TPG 节点
            tag_graph, tag_graph_timedict = tag_graphs[tpg_node], tag_edget_dict[tpg_node]
            tag_nodes = list(tag_graph.nodes)
            for tag_node in tag_nodes: # 遍历每一个 TAG 节点
                tag_graph, tag_graph_timedict = tag_graphs[tpg_node], tag_edget_dict[tpg_node]
                
                time = tag_extime_dict[tag_node] # 不同 TAG 节点对应的游走起始时间不同，初始化映射解决
                
                start_tpg_node = tpg_node
                start_tag_node = tag_node
                s_tag_walk_count_item = 0
                while True: # 一直游走直到时间结束
                    if start_tpg_node not in cover_info:
                        cover_info[start_tpg_node] = {
                            'total_edge': len(tag_graph_timedict), 
                            'edge_cover': set(),
                        }
                    # TAG 的一次游走
                    path, nxt_time = tag_random_walk(tag_graph, tag_graph_timedict, (start_tag_node, ), time, step = step)

                    if path:
                        # TPG + TAG 整体的边覆盖
                        _ = [ cover_info[start_tpg_node]['edge_cover'].add(f'{u},{v}') for u, v in zip(path, path[1:]) ]

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
        
        total, cover = 0, 0
        for tpg_node in cover_info:
            total += cover_info[tpg_node]['total_edge']
            cover += len(cover_info[tpg_node]['edge_cover'])
        g_edge_cover.append(cover / total)

    return (g_edge_cover, s_tag_walk_count, s_tag_node_cover, s_tag_edge_cover, s_tag_path_len)
