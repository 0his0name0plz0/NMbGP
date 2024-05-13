import torch
# from torch_geometric.data import NeighborSampler
from torch_geometric.loader import NeighborSampler
import torch.nn as nn
from downstream.graph_classification.graph_classification import graph_classify,graph_classify2
from train.dataloader import *
from train.model import GraphSAGENet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils.utils import acc
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from train.model import TjlNet,CalcModel,BaseLine,SimplePredictor,SuperSimplePredictor
from train.nodeCL2 import train_node
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.utils import construct_one_graph
from train.eval_node import eval
from main import get_args,set_seed
from train.eval_pretrain import eval_pretrain

def get_dataset(dataset_name):
    print(f"====================>loading dataset: {dataset_name}")
    dataset = TUDataset(root=f'./data/{dataset_name}/', name=f'{dataset_name}', use_node_attr=True)
    print("done.")
    return dataset

def show_graph(dataset,graph_idx,train=False,model=None):
    colors = ['red','green','blue','black','yellow','gray','cyan','magenta','orange']
    color_cur = 0
    plt.clf()
    pca = PCA(n_components=2)
    x = []
    nodeNumList = []
    if train is False:
        for i in graph_idx:
            x.append(dataset[i].x)
            nodeNumList.append(dataset[i].x.shape[0])
    else:
        g,nodeNumList = construct_one_graph(dataset,graph_idx)
        with torch.no_grad():
            _, features = model(g, nodeNumList)        # (graph_num,126,96)
        for i,node_num in enumerate(nodeNumList):
            nodes_emd = features[i,:node_num,:]
            x.append(nodes_emd)
            # print(nodes_emd.shape)
    x = torch.cat(x,dim=0)
    # print(x.shape)

    x = pca.fit_transform(x)
    # points = pca.fit_transform(graph.x)
    p = 0
    for i,graph_id in enumerate(graph_idx):
        node_color = colors[color_cur]
        edge_color = colors[(color_cur+1) % len(colors)]

        graph = dataset[graph_id]
        # points = graph.x[:, 0:2]
        points = x[p:p+nodeNumList[i],:]
        p += nodeNumList[i]
        # print(points.shape)
        plt.scatter(x=points[:, 0], y=points[:, 1], color=node_color)
        for edge in graph.edge_index.T:
            line = points[edge]
            plt.plot(line[:, 0], line[:, 1], color=edge_color, linewidth=1, alpha=0.5)

        color_cur = (color_cur+1) % len(colors)

    # # 设置横坐标范围
    # plt.xlim(0, 35)
    #
    # # 设置纵坐标范围
    # plt.ylim(0, 50)
    plt.show()

def train(model,dataset):
    pass

def getModel(dataset,model_type):
    if model_type.lower() == 'gcn':
        return BaseLine(
            num_features=dataset.num_node_features,
            num_layers=3,
            gnn_type='GCN',
        )
    if model_type.lower() == 'gat':
        return BaseLine(
            num_features=dataset.num_node_features,
            num_layers=2,
            gnn_type='GAT',
            gat_heads=[4,1]
        )
    if model_type.lower() == 'gin':
        return BaseLine(
            num_features=dataset.num_node_features,
            num_layers=3,
            gnn_type='GIN',
        )
    if model_type.lower() == 'graphsage':
        return BaseLine(
            num_features=dataset.num_node_features,
            num_layers=3,
            gnn_type='graphSAGE',
        )

def main(args):
    k=args['k']
    dataset = get_dataset(args['dataset'])
    # GCN
    # model = BaseLine(
    #     num_features=dataset.num_node_features,
    #     num_layers=3,
    #     gnn_type='GCN',
    # )
    # GAT
    # model = BaseLine(
    #     num_features=dataset.num_node_features,
    #     num_layers=2,
    #     gnn_type='GAT',
    #     gat_heads=[4,1]
    # )
    model = getModel(dataset=dataset, model_type=args['baseline'])

    predictor = SuperSimplePredictor(
        input_dim=args['hidden_dim'],
        output_dim=dataset.num_classes,
    )
    # print(model)
    # all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))

    opt1 = Adam(model.parameters(),lr=1e-3,weight_decay=5e-4)
    opt2 = Adam(predictor.parameters(),lr=1e-3,weight_decay=5e-4)
    Loss = nn.CrossEntropyLoss()
    targets = [dataset[i].y for i in range(len(dataset))]
    targets = torch.cat(targets, dim=0)

    train_index=[]
    train_targets=[]
    test_index=[]
    test_targets=[]
    for label in torch.unique(targets):
        label_idx = torch.where(targets == label)[0]
        rand_idx = torch.randperm(len(label_idx))
        sampled_idx = label_idx[rand_idx[:k]]
        other_idx = label_idx[rand_idx[k:]]
        train_index.extend(sampled_idx)
        test_index.extend(other_idx)
        train_targets.extend(targets[sampled_idx])
        test_targets.extend(targets[other_idx])

    train_targets = torch.stack(train_targets, dim=0)
    test_targets = torch.stack(test_targets, dim=0)
    # print(train_index)
    # print(test_index)
    train_g,train_node_list = construct_one_graph(dataset,train_index)
    test_g,test_node_list = construct_one_graph(dataset,test_index)

        # train_features.append(post_out[sampled_idx])
        # train_clusters.append(cluster_ids[sampled_idx])
        # train_targets.append(targets[sampled_idx])
        # train_node_list.append(all_node_list[sampled_idx])
        #
        # test_features.append(post_out[other_idx])
        # test_clusters.append(cluster_ids[other_idx])
        # test_targets.append(targets[other_idx])
        # test_node_list.append(all_node_list[other_idx])

    # train
    # print(f"train num : {len(train_node_list)}")
    # print(f"test num : {len(test_node_list)}")
    for i in range(100):
        post_out = model(train_g, train_node_list)
        graph_emd = post_out.sum(1) # (120,32)
        out = predictor(graph_emd)  # (120,6)
        loss = Loss(out,train_targets)
        # print(loss)
        opt1.zero_grad()
        opt2.zero_grad()
        loss.backward()
        opt1.step()
        opt2.step()

    # test
    with torch.no_grad():
        post_out = model(test_g, test_node_list)
        graph_emd = post_out.sum(1)  # (120,32)
        out = predictor(graph_emd)  # (120,6)
        pred = torch.argmax(out,dim=-1)
        acc = pred == test_targets
        acc = acc.sum()
        print((acc/len(test_targets)).item())
    # print(dataset.data.x.shape)
    # print(dataset.data.edge_index.shape)
    # print(dataset[1].edge_index)
    # model = TjlNet(
    #     num_layers=3,
    #     GNN_model='gin',
    #     input_dim=dataset.num_node_features,
    #     hidden_dim=32,
    # )
    # print(model)
    # print(model)
    # print(dataset.num_classes)
    # print(dataset.num_node_features)
    # out,out_post = model(dataset)
    # print(out.shape)      # [19580, 96]
    # print(out_post.shape) # [600, 126, 96]

    # for i in range(600):
    #     print(f"{i}: {dataset[i].y}")
    # 接下来用自监督方法学习节点embedding
    # 可以考虑节点、边以及图的对比学习（如果时间充裕的话）
    # if args['need_pretrain']:
    #     train_node(model,args=args,dataset=dataset)
    # else:
    #     print(f"======================>loading model from: {args['check_point_model']}")
    #     model.load_state_dict(torch.load(f"pretrained_models/{args['dataset']}/{args['check_point_model']}"))
    #     print("done.")
    #
    # graph_classify(model,dataset,args)
    #
    #
    # # model.load_state_dict(torch.load('pretrained_models/epoch_20_NodeCLModel.pth', map_location='cpu'))
    # # show_graph(dataset,[2])
    # # show_graph(dataset,[2,101,201,301],train=True,model=model)
    # #
    # # eval(model,dataset)
    #
    # # model.load_state_dict(torch.load('pretrained_models/epoch_200_NodeCLModel_withoutAGMTT.pth', map_location='cpu'))       # 0.14
    # # eval_pretrain(model,dataset)
    #
    #
    # # model.load_state_dict(torch.load('pretrained_models/epoch_200_NodeCLModel_withoutAGMTT_withMLP.pth', map_location='cpu'))     # 0.17
    # # MLP = CalcModel(use_mlp=True,hidden_dim=model.num_layers * model.hidden_dim)
    # # MLP.load_state_dict(torch.load('pretrained_models/epoch_200_CalcModel.pth'))
    # # print(MLP)
    # # eval_pretrain(model,dataset,MLP=MLP)

    # eval(model, dataset)

if __name__ == '__main__':
    args = get_args()
    seeds = [0, 10, 100, 1000, 10000, 20000,34210,20051103, 1314520,20020707, 2000000,15461,35146,39655,33333]
    for seed in seeds:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>seed: {seed}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        set_seed(seed)  # 10 100 1000 10000 1000000 2000000
        main(args)