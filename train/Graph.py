import torch
class Graph(torch.nn.Module):
    def __init__(self,x,edge_index):
        super(Graph, self).__init__()
        self.x = x.clone().detach()
        self.edge_index = edge_index.clone().detach()
    def to(self,device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
