import torch
import torch.nn as nn
import numpy as np
from train.NodeCLDataset import NodeCLDataset
from torch.utils.data import DataLoader,random_split
from train.model import CalcModel,NodeCLLoss
from torch.optim import Adam
from train.Graph import Graph
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_node_samples(model, dataset, p1=1, p2=50):
    '''
    dataset.data:   Data(x=[19580, 3], edge_index=[2, 74564], y=[600])
    dataset.len():  600
    dataset[i]:     edge_index=[2, 168], x=[37, 3], y=[1]
    输出格式是：node_info（图的下标，图内节点下标），graph_info（图的下标），targets（正负样本0/1）
    第一个是用于定位一个节点，第二个是用于定位一个图
    '''
    ret_node_sampled = []
    ret_graph_sampled = []
    ret_target = []
    start = 0
    end = 0
    node_list = np.array(list(range(dataset.data.x.shape[0])))  # 0 ~  19580-1
    nodeNumList = []
    for i in range(dataset.len()):
        num_nodes = dataset[i].x.shape[0]
        nodeNumList.append(num_nodes)
    # 构造 （图，节点） 对
    node_graph_pairs = []
    for graph_idx,num_nodes in enumerate(nodeNumList):
        node_graph_pairs.append([[graph_idx, node_idx] for node_idx in range(num_nodes)])
    node_graph_pairs = torch.from_numpy(np.concatenate(node_graph_pairs,axis=0))          # (19580,2)     [0,0] ~ [600,74]
    # 接下来的采样，采样这个pairs就行了
    # pairs的下标与node_list的下标是一一对应的
    # 正样本：每个图采样k个节点与该图组成正样本，k的数量是全图节点数的一定比例
    for i, num_node in enumerate(nodeNumList):
        end += num_node
        k = int(p1 * num_node)
        # 挑选该图的节点，也就是start~end节点
        node_candidate = node_list[start:end]
        # 随机挑选k个
        node_sampled = np.random.choice(node_candidate, k, replace=len(node_candidate)<k)
        ret_node_sampled.append(node_graph_pairs[node_sampled])             # (k,2)
        ret_graph_sampled.append(torch.full(size=(k,),fill_value=i))    # 第i个图 (k)
        ret_target.append(torch.ones(size=(k,)))                       # (k)
        start = end
    start = 0
    end = 0
    # 负样本：对每个图采样k个其他随机图的节点
    for i, num_node in enumerate(nodeNumList):
        end += num_node
        k = int(p2 * num_node)
        # 去除该图的节点，也就是start~end节点
        node_candidate = np.concatenate([node_list[:start], node_list[end:]],axis=0)
        # 随机挑选k个
        node_sampled = np.random.choice(node_candidate, k, replace=len(node_candidate) < k)
        ret_node_sampled.append(node_graph_pairs[node_sampled])             # (k,2)
        ret_graph_sampled.append(torch.full(size=(k,), fill_value=i))   # 第i个图 (k)
        ret_target.append(torch.zeros(size=(k,)))                       # (k)
        start = end
    ret_node_sampled = torch.cat(ret_node_sampled, dim=0)  # (78320, 2)
    ret_graph_sampled = torch.cat(ret_graph_sampled,dim=0)  # (78320)
    ret_target = torch.cat(ret_target,dim=0)                # (78320)
    # print(ret_node_sampled.shape)
    # print(ret_graph_sampled.shape)
    # print(ret_target.shape)

    return NodeCLDataset(ret_node_sampled, ret_graph_sampled, ret_target)


    # not_split_embeddings, node_embeddings, nodeNumList = model(dataset)  # [600, 126, 96]
    # graph_embeddings = torch.sum(node_embeddings,dim=1) # [600,96]
    # ret_node_sampled = []
    # ret_graph_sampled = []
    # ret_target = []
    # # 正样本：每个图采样k个节点与该图组成正样本，k的数量是全图节点数的一定比例
    # for i,num_node in enumerate(nodeNumList):
    #     k = int(p1 * num_node)
    #     node_list = list(range(num_node))
    #     node_sampled = np.random.choice(node_list, k, replace=num_node < k)
    #     ret_node_sampled.append(node_embeddings[i,node_sampled])    # (k,96)
    #     ret_graph_sampled.append(graph_embeddings[i].unsqueeze(0).repeat(k,1))   # (k,96)
    #     ret_target.append(torch.ones(size=(k,)))    # (k)
    # # 负样本：对每个图采样k个其他随机图的节点
    # start = 0
    # end = 0
    # node_list = np.array(list(range(not_split_embeddings.shape[0])))  # 0 ~  19580-1
    # for i,num_node in enumerate(nodeNumList):
    #     end += num_node
    #     k = int(p2 * num_node)
    #     # 去除该图的节点，也就是start~end节点
    #     node_candidate = np.concatenate([node_list[:start], node_list[end:]],axis=0)
    #     # 随机挑选k个
    #     node_sampled = np.random.choice(node_candidate, k, replace=False)
    #     ret_node_sampled.append(not_split_embeddings[node_sampled])  # (k,96)
    #     ret_graph_sampled.append(graph_embeddings[i].unsqueeze(0).repeat(k, 1))  # (k,96)
    #     ret_target.append(torch.zeros(size=(k,)))  # (k)
    #
    #     start = end
    # ret_node_sampled = torch.cat(ret_node_sampled,dim=0)    # (2*k*600, 96)
    # ret_graph_sampled = torch.cat(ret_graph_sampled,dim=0)  # (2*k*600, 96)
    # ret_target = torch.cat(ret_target,dim=0)                # (2*k*600)
    # # 打乱
    # indices = torch.randperm(ret_node_sampled.shape[0])
    # ret_node_sampled = ret_node_sampled[indices]
    # ret_graph_sampled = ret_graph_sampled[indices]
    # ret_target = ret_target[indices]
    # print(ret_node_sampled.shape)     # [72171, 96]
    # print(ret_graph_sampled.shape)    # [72171, 96]
    # print(ret_target.shape)           # [72171]
    #
    # # return ret_node_sampled,ret_graph_sampled,ret_target
    # return NodeCLDataset(ret_node_sampled, ret_graph_sampled, ret_target)

def show_graph(graph,color='red',color2='green'):
    plt.clf()
    # pca = PCA(n_components=2)
    # points = pca.fit_transform(graph.x)
    points = graph.x[:,0:2]
    # print(points.shape)
    plt.scatter(x=points[:,0],y=points[:,1],color=color)
    for edge in graph.edge_index.T:
        line = points[edge]
        plt.plot(line[:, 0], line[:, 1],color=color2,linewidth=1)  # 'k-' 表示黑色的线
    # 设置横坐标范围
    plt.xlim(0, 35)

    # 设置纵坐标范围
    plt.ylim(0, 50)
    plt.show()

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

def train_node(model, dataset,batch_size=2048, epoch_num = 20):
    # out,out_post = model(dataset.data.x, dataset.data.edge_index, nodeNumList)
    # print(out.shape)      # [19580, 96]
    # print(out_post.shape) # [600, 126, 96]
    # 先考虑把节点直接跟图的embedding计算余弦相似度，看看效果，再考虑要不要加一个简单的线性预测层做特征变换

    # 先获取正样本和负样本，并且尽量保证每次epoch训练的节点都不一样，也就是每次epoch（或者间隔几个epoch之后要重新获取一次数据集）
    all_dataset = get_node_samples(model, dataset,p1=1,p2=1)
    train_size = int(0.8 * all_dataset.__len__())
    test_size = all_dataset.__len__() - train_size
    train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size])

    print(f"train data num:{train_dataset.__len__()}")
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=16,shuffle=False)
    # 同时更新两个模型
    calcModel = CalcModel(use_mlp=False,hidden_dim=model.num_layers * model.hidden_dim)
    opt1 = Adam(model.parameters(),lr=1e-3,weight_decay=5e-4)
    # opt2 = Adam(calcModel.parameters(),lr=1e-3,weight_decay=5e-4)
    lossModel = NodeCLLoss()
    # show_graph(dataset[0])
    for epoch in range(epoch_num):
        print(f"============epoch {epoch+1}=====================")
        for node_info,graph_info,targets in train_dataloader:
            '''
                node_info: (bs,2)
                graph_info: (bs,)
                targets: (bs,)
            '''
            # print(f"node_info:{node_info}")
            # print(f"graph_info:{graph_info}")
            # print(f"targets:{targets}")
            g,nodeNumList = construct_one_graph(dataset,torch.cat((node_info[:,0],graph_info),dim=0))
            # print(f"node_info[0][0]:{node_info[0][0]}")
            # print(f"graph_info[0]:{graph_info[0]}")
            # show_graph(dataset[node_info[0][0]],color='black',color2='gray')
            # show_graph(dataset[graph_info[0]],color='red',color2='green')
            # show_graph(g,color='purple',color2='pink')
            out,post_out = model(g,nodeNumList)
            # print(out.shape)          # [4982, 96]
            # print(post_out.shape)     # [128, 100, 96]
            post_out = post_out.reshape((-1,2,post_out.shape[-2],post_out.shape[-1]))   # (64, 2, 126, 96)
            row_indices = torch.arange(post_out.size(0))
            col_indices = node_info[:,1].type(torch.long)
            # tjl_indices = torch.tensor([0])
            node_emd = post_out[row_indices,1,col_indices]    # (bs,96)
            # print(f"node_emd.shape:{node_emd.shape}")
            graph_emd = torch.sum(post_out[:,0,:,:],dim=1)   # (bs,96)
            pred = calcModel(node_emd,graph_emd)                # (bs,)

            # print(pred.shape)
            loss = lossModel(pred,targets)
            print(f"loss:{loss.item()}")
            opt1.zero_grad()
            # opt2.zero_grad()
            loss.backward()
            opt1.step()
            # opt2.step()
        # with torch.no_grad():
        #     for node_info, graph_info, targets in test_dataloader:
        #         g, nodeNumList = construct_one_graph(dataset, torch.cat((node_info[:, 0], graph_info), dim=0))
        #         out, post_out = model(g, nodeNumList)
        #         post_out = post_out.reshape((-1, 2, post_out.shape[-2], post_out.shape[-1]))  # (64, 2, 126, 96)
        #         row_indices = torch.arange(post_out.size(0))
        #         col_indices = node_info[:, 1].type(torch.long)
        #         node_emd = post_out[row_indices, 0, col_indices]  # (bs,96)
        #         graph_emd = torch.sum(post_out[:, 1, :, :], dim=1)  # (bs,96)
        #         pred = calcModel(node_emd, graph_emd)  # (bs,)
        #         acc(pred,targets)
    torch.save(model.state_dict(),f'pretrained_models/epoch_{epoch_num}_NodeCLModel.pth')
    torch.save(calcModel.state_dict(),f'pretrained_models/epoch_{epoch_num}_CalcModel.pth')

def acc(pred,targets):
    print(pred,targets)
    acc_num = pred[targets == 1]
    print(acc_num)
    exit(1)