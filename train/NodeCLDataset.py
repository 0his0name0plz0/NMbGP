from torch.utils.data import Dataset

class NodeCLDataset(Dataset):
    def __init__(self,node_sampled,graph_sampled,target):
        super(NodeCLDataset, self).__init__()
        self.node_sampled = node_sampled
        self.graph_sampled = graph_sampled
        self.target = target

    def __len__(self):
        return self.node_sampled.shape[0]

    def __getitem__(self, index):
        return self.node_sampled[index],self.graph_sampled[index],self.target[index]

class NodeCLDataset2(Dataset):
    def __init__(self,graph_node_pairs):
        super(NodeCLDataset2, self).__init__()
        self.graph_node_pairs = graph_node_pairs

    def __len__(self):
        return self.graph_node_pairs.shape[0]

    def __getitem__(self, index):
        return self.graph_node_pairs[index]

class GraphDataset(Dataset):
    def __init__(self,graph_emd,targets):
        super(GraphDataset, self).__init__()
        self.graph_emd = graph_emd
        self.targets = targets

    def __len__(self):
        return self.graph_emd.shape[0]

    def __getitem__(self, index):
        return self.graph_emd[index], self.targets[index]
