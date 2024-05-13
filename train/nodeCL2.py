import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,random_split
from train.model import CalcModel,NodeCLLoss
from torch.optim import Adam
from train.Graph import Graph
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.utils import construct_one_graph,get_node_samples

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


def train_node(model, dataset,args,batch_size=2048, epoch_num=200):
    # model.load_state_dict(torch.load('pretrained_models/epoch_20_NodeCLModel.pth', map_location='cpu'))
    # out,out_post = model(dataset.data.x, dataset.data.edge_index, nodeNumList)
    # print(out.shape)      # [19580, 96]
    # print(out_post.shape) # [600, 126, 96]
    # 先获取包含全部小图的整个大图
    all_g,all_node_list = construct_one_graph(dataset,list(range(len(dataset))))

    # 先获取正样本和负样本，并且尽量保证每次epoch训练的节点都不一样，也就是每次epoch（或者间隔几个epoch之后要重新获取一次数据集）
    all_dataset = get_node_samples(all_node_list)
    # train_size = int(0.8 * all_dataset.__len__())
    # test_size = all_dataset.__len__() - train_size
    # train_dataset, test_dataset = random_split(all_dataset, [train_size, test_size])

    print(f"train data num:{all_dataset.__len__()}")
    # 没必要分训练和测试，无监督训练嘛
    train_dataloader = DataLoader(all_dataset,batch_size=batch_size,shuffle=True)
    # test_dataloader = DataLoader(test_dataset,batch_size=test_dataset.__len__(),shuffle=False)
    # 同时更新两个模型
    calcModel = CalcModel(use_mlp=True,hidden_dim=model.num_layers * model.hidden_dim)
    opt1 = Adam(model.parameters(),lr=1e-3,weight_decay=5e-4)
    opt2 = Adam(calcModel.parameters(),lr=1e-3,weight_decay=5e-4)
    lossModel = NodeCLLoss()
    # show_graph(dataset[0])

    model = model.to(args['device'])
    calcModel = calcModel.to(args['device'])
    all_g.to(args['device'])
    # all_g = all_g.to(args['device'])
    # print(all_g.device)
    # print(all_g.x.device)

    # all_node_list.to(args['device'])
    for epoch in range(epoch_num):
        loss_record = []
        print(f"============epoch {epoch+1}=====================")
        for graph_node_batch in train_dataloader:
            '''
                graph_node_batch: (bs,2)
            '''
            # print(f"node_info:{node_info}")
            # print(f"graph_info:{graph_info}")
            # print(f"targets:{targets}")
            # 获取全图节点的emd：
            _, post_out = model(all_g, all_node_list)   # [600, 126, 96]

            node_emd = post_out[graph_node_batch[:,0].type(torch.long), graph_node_batch[:,1].type(torch.long),:]   # [bs, 96]
            node_emd = calcModel(node_emd)
            # graph_emd = torch.sum(post_out,dim=1) / torch.tensor(all_node_list).unsqueeze(-1).repeat(1,node_emd.shape[1])   # (600,96)
            graph_emd = torch.sum(post_out, dim=1)      # (600,96)
            positive_graph_emd = graph_emd[graph_node_batch[:,0].type(torch.long)]      # (bs,96)
            negative_graph_emd = graph_emd.unsqueeze(0).repeat(node_emd.shape[0],1,1)   # (bs,600,96)
            mask = torch.ones((negative_graph_emd.shape[0], negative_graph_emd.shape[1]))
            mask[torch.arange(mask.shape[0]), graph_node_batch[:,0].type(torch.long)] = 0   # (bs,600)
            loss = lossModel(node_emd,positive_graph_emd,negative_graph_emd,mask)
            # print(graph_emd.shape)
            print(f"loss:{loss}")
            opt1.zero_grad()
            opt2.zero_grad()
            loss.backward()
            opt1.step()
            opt2.step()
            loss_record.append(loss.item())
        print(f"avg loss:{sum(loss_record) / len(loss_record)}")
            # # g,nodeNumList = construct_one_graph(dataset,torch.cat((node_info[:,0],graph_info),dim=0))
            # # print(f"node_info[0][0]:{node_info[0][0]}")
            # # print(f"graph_info[0]:{graph_info[0]}")
            # # show_graph(dataset[node_info[0][0]],color='black',color2='gray')
            # # show_graph(dataset[graph_info[0]],color='red',color2='green')
            # # show_graph(g,color='purple',color2='pink')
            # out,post_out = model(g,nodeNumList)
            # # print(out.shape)          # [4982, 96]
            # # print(post_out.shape)     # [128, 100, 96]
            # post_out = post_out.reshape((-1,2,post_out.shape[-2],post_out.shape[-1]))   # (64, 2, 126, 96)
            # row_indices = torch.arange(post_out.size(0))
            # col_indices = node_info[:,1].type(torch.long)
            # # tjl_indices = torch.tensor([0])
            # node_emd = post_out[row_indices,0,col_indices]    # (bs,96)
            # graph_emd = torch.mean(post_out[:,1,:,:],dim=1)    # (bs,96)
            # pred = calcModel(node_emd,graph_emd)                # (bs,)
            #
            # # print(pred.shape)
            # loss = lossModel(pred,targets)
            # print(f"loss:{loss.item()}")
            # opt1.zero_grad()
            # # opt2.zero_grad()
            # loss.backward()
            # opt1.step()
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
    torch.save(model.state_dict(),f"pretrained_models/{args['dataset']}/epoch_{epoch_num}_NodeCLModel_withoutAGMTT_withMLP.pth")
    torch.save(calcModel.state_dict(),f"pretrained_models/{args['dataset']}/epoch_{epoch_num}_CalcModel.pth")

def acc(pred,targets):
    print(pred,targets)
    acc_num = pred[targets == 1]
    print(acc_num)
    exit(1)