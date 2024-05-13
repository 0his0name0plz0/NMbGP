import os
import torch
import numpy as np
from train.Graph import Graph

from train.NodeCLDataset import NodeCLDataset2
def error(msg):
    print(f"\033[93m{msg}\033[0m")

def acc(pred,targets,mask):
    '''

    :param pred: shape (n,c)
    :param targets: shape(n)
    :return:
    '''
    pred_label = torch.argmax(pred,dim=1)
    correct_num = ((pred_label == targets)*mask).sum().item()
    total_num = mask.sum().item()
    return correct_num / total_num

def post_process(features,nodeNumList):
    '''
    (19580,32)的feature矩阵变成(600,126,32)的，这个126是结点数的最大值
    :param features: (19580,32)
    :param nodeNumList: [37, 12, ..., 48]   总和为19580，最大值为126
    :return:
    '''
    max_num = max(nodeNumList)
    features = list(torch.split(features, nodeNumList))
    for i, feature in enumerate(features):
        if feature.shape[0] != max_num:
            # (17,32) + (109,32) -> (126,32)
            feature = torch.cat((feature,torch.zeros((max_num-feature.shape[0],feature.shape[1])).to(feature.device)),dim=0)
            features[i] = feature   # (126,32)
    features = torch.stack(features)
    # print(features.shape)     # (600,126,32)
    return features

def construct_one_graph(dataset,graph_index):
    '''
    根据graph index来构建一整张大图，主要操作是改变edge index的值
    graph_index: (bs,) 表示参与构建大图的子图下标
    return：
    1、graph，构建出的一整张大图，包括.x和.edge_index两个属性
    2、nodeNumList：一个整数列表，记录了每个图里面有多少节点
    '''
    ret_node_features = []
    ret_edge_index = []
    nodeNumList = []
    cur = 0
    # print(f"construct graph : {graph_index}")
    for i in graph_index:
        num_nodes = dataset[i].x.shape[0]
        edge_idx = np.array(dataset[i].edge_index).T    # (28,2)
        edge_idx += cur

        ret_node_features.append(dataset[i].x)      # (32,3)
        ret_edge_index.append(edge_idx)             # (28,2)
        cur += num_nodes
        nodeNumList.append(num_nodes)
    ret_node_features = np.concatenate(ret_node_features,axis=0)    # (n,3)
    ret_edge_index = np.concatenate(ret_edge_index,axis=0).T        # (2,e)
    # print(ret_node_features.shape)
    # print(ret_edge_index.shape)
    return Graph(torch.from_numpy(ret_node_features), torch.from_numpy(ret_edge_index)),nodeNumList

def get_node_samples(nodeNumList,return_dataset=True):
    '''
    返回值格式：(bs,[graph_id,node_id])

    :param dataset:
    :return:
    '''
    # 构造 （图，节点） 对
    graph_node_pairs = []
    for graph_idx, num_nodes in enumerate(nodeNumList):
        graph_node_pairs.append([[graph_idx, node_idx] for node_idx in range(num_nodes)])
    graph_node_pairs = torch.from_numpy(np.concatenate(graph_node_pairs, axis=0))  # (19580,2)     [0,0] ~ [600,74]
    if return_dataset:
        return NodeCLDataset2(graph_node_pairs)
    else:
        return graph_node_pairs
