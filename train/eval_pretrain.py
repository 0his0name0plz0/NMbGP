
import torch

from train.model import NodeCLLoss
from utils.utils import construct_one_graph,get_node_samples
import numpy as np
from torch.nn.functional import cosine_similarity as cos
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def eval_pretrain(model,dataset,batch_size=2048,MLP=None):
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    all_dataset = get_node_samples(all_node_list)
    dataloader = DataLoader(all_dataset, batch_size=batch_size, shuffle=False)
    acc_list = []
    hit_num = torch.zeros((600,))
    Loss = NodeCLLoss()
    loss_list = []
    for pairs in dataloader:
        '''
            pairs: (bs,2)
        '''
        with torch.no_grad():
            _, post_out = model(all_g, all_node_list)  # [600, 126, 96]

        node_emd = post_out[pairs[:, 0].type(torch.long), pairs[:, 1].type(torch.long), :]  # (bs,96)
        if MLP is not None:
            node_emd = MLP(node_emd)
        graph_emd = torch.sum(post_out, dim=1)  # (600,96)
        graph_emd = graph_emd.unsqueeze(0).repeat(node_emd.shape[0], 1, 1)  # (bs,600,96)
        mask = torch.ones((graph_emd.shape[0], graph_emd.shape[1]))
        mask[torch.arange(mask.shape[0]), pairs[:, 0].type(torch.long)] = 0  # (bs,600)
        positive_graph_emd = torch.sum(post_out, dim=1)[pairs[:,0].type(torch.long)]
        with torch.no_grad():
            loss = Loss(node_emd,positive_graph_emd,graph_emd,mask)
            print(f"loss:{loss}")
            loss_list.append(loss.item())
        node_emd = node_emd.unsqueeze(1).repeat(1, graph_emd.shape[1], 1)
        sim = cos(node_emd, graph_emd, dim=-1)  # (bs,600)
        pred = torch.argmax(sim, dim=-1)  # (bs,)
        targets = pairs[:, 0]  # (bs,)
        # print(pred)
        # print(targets)
        # print("===============================================")
        acc = (pred == targets).sum()
        acc = acc / batch_size
        acc_list.append(acc.item())
        counts = torch.bincount(pred,minlength=post_out.shape[0])
        # print(f"counts:{counts}")
        hit_num += counts
    print(f"acc: {sum(acc_list) / len(acc_list)}")
    print(hit_num)
    print(hit_num.sum())
    print(f"loss avg:{sum(loss_list) / len(loss_list)}")
    for i in range(0, 600, 100):
        plt.clf()
        plt.bar(x=list(range(hit_num[i:i + 100].shape[0])), height=all_node_list[i:i+100])
        plt.savefig(f"{i}_target.png")
    for i in range(0,600,100):
        plt.clf()
        plt.bar(x=list(range(hit_num[i:i+100].shape[0])),height=list(hit_num[i:i+100].numpy()))
        plt.savefig(f"{i}_pred.png")

        # node_emd = post_out[graph_node_batch[:, 0].type(torch.long), graph_node_batch[:, 1].type(torch.long),:]  # [bs, 96]
        # # node_emd = calcModel(node_emd)
        # # graph_emd = torch.sum(post_out,dim=1) / torch.tensor(all_node_list).unsqueeze(-1).repeat(1,node_emd.shape[1])   # (600,96)
        # graph_emd = torch.sum(post_out, dim=1)  # (600,96)
        # positive_graph_emd = graph_emd[graph_node_batch[:, 0].type(torch.long)]  # (bs,96)
        # negative_graph_emd = graph_emd.unsqueeze(0).repeat(node_emd.shape[0], 1, 1)  # (bs,600,96)
        # mask = torch.ones((negative_graph_emd.shape[0], negative_graph_emd.shape[1]))
        # mask[torch.arange(mask.shape[0]), graph_node_batch[:, 0].type(torch.long)] = 0  # (bs,600)




    # node_emd = post_out[pairs[:,0].type(torch.long), pairs[:,1].type(torch.long), :]    # (19580,96)
    # graph_emd = torch.sum(post_out, dim=1)  # (600,96)
    # graph_emd = graph_emd.unsqueeze(0).repeat(node_emd.shape[0],1,1)    # (19580,600,96)
    # node_emd = node_emd.unsqueeze(1).repeat(1,graph_emd.shape[1],1)
    # sim = cos(node_emd,graph_emd,dim=-1)    # (19580,600)
    # pred = torch.argmax(sim,dim=-1)     # (19580,)
    # targets = pairs[:,0]    # (19580,)
    # acc = (pred == targets).sum()
    # print(acc)
def draw_graph_emd(model,dataset):
    '''
    展示图的emd，看看同一标签的图是否会在一起
    :param model:
    :param dataset:
    :return:
    '''
    all_g, all_node_list = construct_one_graph(dataset, list(range(len(dataset))))
    _, post_out = model(all_g, all_node_list)  # [600, 126, 96]
