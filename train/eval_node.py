from train.NodeCLDataset import GraphDataset
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader,random_split
from train.model import SimplePredictor
from utils.utils import construct_one_graph
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from collections import defaultdict

def eval(model,dataset,k=5,batch_size=32):
    # 先获取包含全部小图的整个大图
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    with torch.no_grad():
        _,node_emd = model(all_g, all_node_list)      # (600,120,96)
    # features = torch.sum(node_emd, dim=1) / torch.tensor(all_node_list).unsqueeze(-1).repeat(1, node_emd.shape[-1])
    # features,_ = torch.max(node_emd, dim=1)
    # features = torch.mean(torch.topk(node_emd,k=10,dim=1)[0],dim=1)
    features = torch.sum(node_emd,dim=1)
    targets = [dataset[i].y for i in range(len(dataset))]
    targets = torch.cat(targets,dim=0)
    # 取每个类别里面k个样本做k-shot training
    train_features = []
    train_targets = []
    test_features = []
    test_targets = []
    for label in torch.unique(targets):
        label_idx = torch.where(targets == label)[0]
        rand_idx = torch.randperm(len(label_idx))
        sampled_idx = label_idx[rand_idx[:k]]
        other_idx = label_idx[rand_idx[k:]]
        train_features.append(features[sampled_idx])
        train_targets.append(targets[sampled_idx])
        test_features.append(features[other_idx])
        test_targets.append(targets[other_idx])
    train_features = torch.cat(train_features,dim=0)
    train_targets = torch.cat(train_targets,dim=0)
    test_features = torch.cat(test_features, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    print(f"train_features:{train_features.shape},train_targets:{train_targets.shape}")
    print(f"test_features:{test_features.shape},test_targets:{test_targets.shape}")

    # idx = NodeCLDataset2(torch.arange(dataset.len()))
    train_dataset = GraphDataset(train_features,train_targets)
    test_dataset = GraphDataset(test_features,test_targets)

    # train_size = 30
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(graph_dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictor = SimplePredictor(
        input_dim=features.shape[-1],
        hidden_dim=features.shape[-1],
        output_dim=dataset.num_classes
    )
    opt = Adam(predictor.parameters(),lr=1e-2,weight_decay=5e-4)
    Loss = CrossEntropyLoss()

    for epoch in range(100):
        print(f"===============epoch:{epoch}==================")
        for x,y in train_dataloader:
            pred = predictor(x)
            loss = Loss(pred,y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"loss:{loss}")

    with torch.no_grad():
        accumulate_acc = 0
        for x,y in test_dataloader:
            pred = predictor(x) # (bs,6)
            acc = torch.argmax(pred,dim=-1)
            acc = (acc == y).sum()
            accumulate_acc += acc.item()
        print(f"acc:{accumulate_acc / test_dataset.__len__()}")
