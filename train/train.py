import torch
# from torch_geometric.data import NeighborSampler
from torch_geometric.loader import NeighborSampler
from train.dataloader import *
from train.model import GraphSAGENet,TjlNet
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from utils.utils import acc
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F


cfg = {
    'dataset': 'cora',
    'data_root': './dataset/cora/',
}
def masked_loss(out, label, mask):

    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss
def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc

class Predictor(torch.nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Predictor, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channel,in_channel),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channel,out_channel)
        )
    def forward(self,x):
        out = self.mlp(x)
        return out

def main():
    args = cfg
    # dataset = dataloader(args)
    # train_mask,test_mask = dataset.split_train_test(train_num=1000)
    dataset0 = Planetoid(root='data/Planetoid',name='cora',transform=NormalizeFeatures())
    dataset = dataset0[0]
    train_mask,test_mask = dataset.train_mask, dataset.test_mask
    print(f"train num:{train_mask.sum().item()}")
    print(f"test num:{test_mask.sum().item()}")

    # model = GraphSAGENet(in_channels=dataset.features.shape[1],hidden_channels=16,out_channels=dataset.num_targets)
    # model = GraphSAGENet(in_channels=dataset.x.shape[1], hidden_channels=16, out_channels=dataset0.num_classes)
    model = TjlNet(num_layers=3,GNN_model='gin',input_dim=dataset.x.shape[1],hidden_dim=32)
    predictor = Predictor(96,dataset0.num_classes)

    loss_model = CrossEntropyLoss()

    # sampler = NeighborSampler(dataset.edge_index, sizes=[10, 5], batch_size=3, shuffle=True)
    # print(dataset.features.shape)
    # print(dataset.targets.shape)
    opt = Adam(model.parameters(),lr=1e-2,weight_decay=5e-4)
    opt2 = Adam(predictor.parameters(),lr=1e-2,weight_decay=5e-4)
    for epoch in range(20):
        # pred = model(dataset.features,dataset.edge_index)
        # pred = pred*(train_mask.unsqueeze(1))
        # loss = loss_model(pred,dataset.targets * train_mask)
        model.train()
        # pred = model(dataset.x, dataset.edge_index)
        emd = model(dataset)    # (2708,96)
        # print(emd.shape)
        pred = predictor(emd)
        loss = loss_model(pred[train_mask],dataset.y[train_mask])
        opt.zero_grad()
        opt2.zero_grad()
        loss.backward()
        opt.step()
        opt2.step()
        print(loss.item())

    with torch.no_grad():
        emd = model(dataset)
        pred = predictor(emd).argmax(dim=1)
        print(f"accuracy:{(pred[test_mask] == dataset.y[test_mask]).sum().item() / test_mask.sum().item()}")
    # for batch_size, n_id, adjs in sampler:
    #     print(f"batch_size:{batch_size}\n n_id:{n_id.shape}\n adjs:{adjs}\n")
    #     pred = model(dataset.features,adjs)