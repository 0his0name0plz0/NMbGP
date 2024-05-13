from sklearn.cluster import KMeans

from utils.utils import construct_one_graph,get_node_samples
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from train.model import NodeCLLossDownStream, GraphClassificationPromptModel, NodeCLLossDownStream2, \
    GraphClassificationPromptModel2, CalcModel, SuperSimplePredictor
from torch.optim import Adam

class GraphClusterDataset(Dataset):
    def __init__(self,graph_emd,graph_cluster,graph_label):
        super(GraphClusterDataset, self).__init__()
        self.graph_emd = graph_emd
        self.graph_cluster = graph_cluster
        self.graph_label = graph_label

    def __len__(self):
        return self.graph_emd.shape[0]

    def __getitem__(self, index):
        return self.graph_emd[index],self.graph_cluster[index],self.graph_label[index]


def get_graph_dataset(model,dataset,args):
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    with torch.no_grad():
        _, post_out = model(all_g, all_node_list)
    graph_emd = torch.sum(post_out, dim=1)
    # 聚类
    kmeans = KMeans(n_clusters=args['cluster_num'], random_state=0)
    kmeans.fit(graph_emd.numpy())
    # 中心点和标签，用中心点去初始化每个cluster里面的prompt
    cluster_centers = kmeans.cluster_centers_   # (3,96)
    cluster_ids = kmeans.labels_     # (600,)

    # return GraphClusterDataset(graph_emd=graph_emd,graph_cluster=cluster_ids),cluster_centers
    # print(graph_id_list)

def get_k_shot_dataset(model,dataset,args):
    '''

    :param model:
    :param dataset:
    :param args:
    :return: 返回值是：训练数据集、测试数据集、聚类中心点
    '''
    # seed = 2000000
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    with torch.no_grad():
        _, post_out = model(all_g, all_node_list)
    graph_emd = torch.sum(post_out, dim=1)
    # 聚类
    kmeans = KMeans(n_clusters=args['cluster_num'], random_state=0)
    kmeans.fit(graph_emd.numpy())
    # 中心点和标签，用中心点去初始化每个cluster里面的prompt
    cluster_centers = kmeans.cluster_centers_  # (3,96)
    cluster_ids = torch.tensor(kmeans.labels_)  # (600,)

    targets = [dataset[i].y for i in range(len(dataset))]
    targets = torch.cat(targets, dim=0)
    # print(targets.shape)
    k = args['k']
    train_features = []
    train_clusters = []
    train_targets = []
    test_features = []
    test_clusters = []
    test_targets = []
    for label in torch.unique(targets):
        label_idx = torch.where(targets == label)[0]
        rand_idx = torch.randperm(len(label_idx))
        sampled_idx = label_idx[rand_idx[:k]]
        other_idx = label_idx[rand_idx[k:]]
        train_features.append(graph_emd[sampled_idx])
        train_clusters.append(cluster_ids[sampled_idx])
        train_targets.append(targets[sampled_idx])
        test_features.append(graph_emd[other_idx])
        test_clusters.append(cluster_ids[other_idx])
        test_targets.append(targets[other_idx])
    train_features = torch.cat(train_features, dim=0)
    train_clusters = torch.cat(train_clusters, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_clusters = torch.cat(test_clusters, dim=0)
    test_targets = torch.cat(test_targets, dim=0)

    return GraphClusterDataset(graph_emd=train_features, graph_cluster=train_clusters, graph_label=train_targets),\
           GraphClusterDataset(graph_emd=test_features, graph_cluster=test_clusters, graph_label=test_targets),\
           cluster_centers


def graph_classify(pt_model,dataset,args):
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    train_dataset,test_dataset,cluster_centers = get_k_shot_dataset(pt_model,dataset,args)
    print(f"train num:{train_dataset.__len__()}, test num:{test_dataset.__len__()}")
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

    train_model = GraphClassificationPromptModel(
        targets_num=dataset.num_classes,
        prompt_dim=pt_model.hidden_dim * pt_model.num_layers,
        cluster_num=args['cluster_num'],
        prompt_num=args['prompt_num'],
        # cluster_centers=cluster_centers
    )
    print(train_model.prompts.shape)
    Loss = NodeCLLossDownStream()
    opt = Adam(train_model.parameters(),lr=args['lr_ds'],weight_decay=5e-3)
    best_acc=0
    for epoch in range(args['downstream_epoch']):
        for train_batch in train_dataloader:
            '''
                train_batch: (bs,96), (bs,), (bs,)
                第一个表示图的emd，第二个表示图的聚类id，第三个表示图的类别
            '''
            pred = train_model(train_batch[0],train_batch[1])
            # print(f"pred:{pred}")
            loss = Loss(pred,train_batch[2])
            # print(f"loss:{loss.item()}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(f"loss:{loss.item()}")
        # print(f"==================epoch {epoch+1}====================")
        acc_list = []
        for test_batch in test_dataloader:
            '''
                test_dataloader: (bs,96), (bs,), (bs,)
                第一个表示图的emd，第二个表示图的聚类id，第三个表示图的类别
            '''
            with torch.no_grad():
                pred = train_model(test_batch[0], test_batch[1])  # (bs,6,10)
            pred = pred.sum(-1)  # (bs,6)
            pred = torch.argmax(pred, dim=-1)
            acc = pred == test_batch[2]
            acc = acc.sum().item() / test_batch[0].shape[0]
            # print(f"acc : {acc}")
            acc_list.append(acc)
        acc_ = sum(acc_list) / len(acc_list)
        # print(f"test avg acc:{acc_}")
        best_acc = max(best_acc, acc_)

    print(f'best accuracy: {best_acc}')
        # print(f"test avg acc:{sum(acc_list) / len(acc_list)}")

    # acc_list = []
    # for test_batch in test_dataloader:
    #     '''
    #         test_dataloader: (bs,96), (bs,), (bs,)
    #         第一个表示图的emd，第二个表示图的聚类id，第三个表示图的类别
    #     '''
    #     with torch.no_grad():
    #         pred = train_model(test_batch[0], test_batch[1])    # (bs,6,10)
    #     pred = pred.sum(-1)     # (bs,6)
    #     pred = torch.argmax(pred,dim=-1)
    #     acc = pred == test_batch[2]
    #     acc = acc.sum().item() / test_batch[0].shape[0]
    #     # print(f"acc : {acc}")
    #     acc_list.append(acc)
    # print(f"avg acc:{sum(acc_list) / len(acc_list)}")


# 尝试使用节点嵌入而不是图嵌入
class GraphClusterDataset2(Dataset):
    def __init__(self,node_emd,graph_cluster,graph_label,node_num_list):
        super(GraphClusterDataset2, self).__init__()
        self.node_emd = node_emd
        self.graph_cluster = graph_cluster
        self.graph_label = graph_label
        self.node_num_list = node_num_list

    def __len__(self):
        return self.node_emd.shape[0]

    def __getitem__(self, index):
        return self.node_emd[index],self.graph_cluster[index],self.graph_label[index],self.node_num_list[index]

def get_k_shot_dataset2(model,calcModel,dataset,args):
    '''

    :param model:
    :param dataset:
    :param args:
    :return: 返回值是：训练数据集、测试数据集、聚类中心点
    '''
    # seed = 2000000
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    with torch.no_grad():
        _, post_out = model(all_g, all_node_list)   # (600,126,96)
    graph_emd = torch.sum(post_out, dim=1)
    # 聚类
    kmeans = KMeans(n_clusters=args['cluster_num'], random_state=0)
    kmeans.fit(graph_emd.numpy())
    # 中心点和标签，用中心点去初始化每个cluster里面的prompt
    cluster_centers = kmeans.cluster_centers_  # (3,96)
    cluster_ids = torch.tensor(kmeans.labels_)  # (600,)

    targets = [dataset[i].y for i in range(len(dataset))]
    targets = torch.cat(targets, dim=0)
    # print(targets.shape)
    k = args['k']
    train_features = []
    train_clusters = []
    train_targets = []
    train_node_list = []

    test_features = []
    test_clusters = []
    test_targets = []
    test_node_list = []
    all_node_list = torch.tensor(all_node_list)
    with torch.no_grad():
        post_out = calcModel(post_out)
    for label in torch.unique(targets):
        label_idx = torch.where(targets == label)[0]
        rand_idx = torch.randperm(len(label_idx))
        sampled_idx = label_idx[rand_idx[:k]]
        other_idx = label_idx[rand_idx[k:]]

        train_features.append(post_out[sampled_idx])
        train_clusters.append(cluster_ids[sampled_idx])
        train_targets.append(targets[sampled_idx])
        train_node_list.append(all_node_list[sampled_idx])

        test_features.append(post_out[other_idx])
        test_clusters.append(cluster_ids[other_idx])
        test_targets.append(targets[other_idx])
        test_node_list.append(all_node_list[other_idx])

    train_features = torch.cat(train_features, dim=0)
    train_clusters = torch.cat(train_clusters, dim=0)
    train_targets = torch.cat(train_targets, dim=0)
    train_node_list = torch.cat(train_node_list, dim=0)

    test_features = torch.cat(test_features, dim=0)
    test_clusters = torch.cat(test_clusters, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    test_node_list = torch.cat(test_node_list, dim=0)


    return GraphClusterDataset2(node_emd=train_features, graph_cluster=train_clusters, graph_label=train_targets, node_num_list=train_node_list),\
           GraphClusterDataset2(node_emd=test_features, graph_cluster=test_clusters, graph_label=test_targets, node_num_list=test_node_list),\
           cluster_centers

def graph_classify2(pt_model,dataset,args):
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    calcModel = CalcModel(use_mlp=True, hidden_dim=pt_model.num_layers * pt_model.hidden_dim)
    calcModel.load_state_dict(torch.load(f"pretrained_models/{args['dataset']}/{args['calc_model']}"))
    train_dataset,test_dataset,cluster_centers = get_k_shot_dataset2(pt_model,calcModel,dataset,args)
    # print(f"train num:{train_dataset.__len__()}, test num:{test_dataset.__len__()}")

    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

    train_model = GraphClassificationPromptModel2(
        targets_num=dataset.num_classes,
        prompt_dim=pt_model.hidden_dim * pt_model.num_layers,
        cluster_num=args['cluster_num'],
        prompt_num=args['prompt_num'],
        # cluster_centers=cluster_centers
    )
    print(train_model.prompts.shape)
    Loss = NodeCLLossDownStream2()
    opt = Adam(train_model.parameters(),lr=args['lr_ds'],weight_decay=5e-3)
    best_acc = 0
    for epoch in range(args['downstream_epoch']):
        for train_batch in train_dataloader:
            '''
                train_batch: (bs,126,96), (bs,), (bs,), (bs,)
                第一个表示图的节点emd，第二个表示图的聚类id，第三个表示图的类别，第四个表示图中节点数
            '''
            # print(train_batch[0].shape)
            # print(train_batch[1].shape)
            # print(train_batch[2].shape)
            # print(train_batch[3].shape)

            pred = train_model(train_batch[0],train_batch[1])
            # print(f"pred:{pred}")
            loss = Loss(pred,train_batch[2],node_list=train_batch[3])
            # print(f"loss:{loss.item()}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(f"loss:{loss.item()}")
        print(f"==================epoch {epoch+1}====================")
        acc_list = []
        for test_batch in test_dataloader:
            '''
                test_batch: (bs,126,96), (bs,), (bs,), (bs,)
                第一个表示图的节点emd，第二个表示图的聚类id，第三个表示图的类别，第四个表示图中节点数
            '''
            with torch.no_grad():
                pred = train_model(test_batch[0], test_batch[1])  # (bs,126,6,10)
            pred = pred.sum(-1)  # (bs,126,6)
            pred = pred.sum(-2)  # (bs,6)
            pred = torch.argmax(pred, dim=-1)   # (bs,)
            acc = pred == test_batch[2]
            acc = acc.sum().item() / test_batch[0].shape[0]
            # print(f"acc : {acc}")
            acc_list.append(acc)
        acc_ = sum(acc_list) / len(acc_list)
        print(f"test avg acc:{acc_}")
        best_acc = max(best_acc,acc_)

    print(f'best accuracy: {best_acc}')

def graph_classify_mlp(pt_model,dataset,args):
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    train_dataset,test_dataset,cluster_centers = get_k_shot_dataset(pt_model,dataset,args)
    print(f"train num:{train_dataset.__len__()}, test num:{test_dataset.__len__()}")
    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True)

    train_model = SuperSimplePredictor(
        input_dim=pt_model.hidden_dim * pt_model.num_layers,
        output_dim=dataset.num_classes,
    )
    # print(train_model.prompts.shape)
    Loss = nn.CrossEntropyLoss()
    opt = Adam(train_model.parameters(),lr=1e-3,weight_decay=5e-3)
    best_acc=0
    for epoch in range(100):
        loss_list = []
        for train_batch in train_dataloader:
            '''
                train_batch: (bs,96), (bs,), (bs,)
                第一个表示图的emd，第二个表示图的聚类id，第三个表示图的类别
            '''
            out = train_model(train_batch[0])
            loss = Loss(out, train_batch[2])
            loss_list.append(loss.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            # print(f"loss:{loss.item()}")
        loss_avg = sum(loss_list) / len(loss_list)
        # print(f"avg loss:{loss_avg}")
        # print(f"==================epoch {epoch+1}====================")
        acc_list = []
        for test_batch in test_dataloader:
            '''
                test_dataloader: (bs,96), (bs,), (bs,)
                第一个表示图的emd，第二个表示图的聚类id，第三个表示图的类别
            '''

            with torch.no_grad():
                out = train_model(test_batch[0])
            out = torch.argmax(out,dim=-1)
            acc = out == test_batch[2]
            acc = acc.sum().item() / test_batch[0].shape[0]
            # print(f"acc : {acc}")
            acc_list.append(acc)
        acc_ = sum(acc_list) / len(acc_list)
        # print(f"test avg acc:{acc_}")
        best_acc = max(best_acc, acc_)

    print(f'best accuracy: {best_acc}')
        # print(f"test avg acc:{sum(acc_list) / len(acc_list)}")