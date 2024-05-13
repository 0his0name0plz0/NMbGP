import os
from utils.utils import error
import numpy as np
import torch
class Dataset():
    def __init__(self,features,targets,node_dict,target_dict,edge_index):
        '''
            :param
                features: torch.tensor: [n,1433]
                targets: torch.tensor: [n]
                node_dict: dict
                target_dict: dict
                edge_index: torch.tensor: [2,edge_num]
        '''
        super(Dataset,self).__init__()
        self.num_nodes = features.shape[0]
        self.num_feature = features.shape[1]
        self.num_targets = len(target_dict.keys())
        self.features = features
        self.targets = targets
        self.node_dict = node_dict
        self.target_dict = target_dict
        self.edge_index = edge_index

    def split_train_test(self,train_num=120,shuffle=True):
        idx = range(0, self.num_nodes)
        if shuffle is True:
            idx_train = np.random.choice(idx,train_num,replace=False)
        else:
            idx_train = np.array(list(range(0, train_num)))
        idx_test = list(set(idx) - set(idx_train))

        self.train_mask = torch.full((self.num_nodes,),False)
        self.test_mask = torch.full((self.num_nodes,), False)
        self.train_mask[idx_train] = True
        self.test_mask[idx_test] = True
        return self.train_mask,self.test_mask

        #
        # self.train_x, self.train_y = self.features[idx_train], self.targets[idx_train]
        # self.test_x, self.test_y = self.features[idx_test], self.targets[idx_test]
        #
        # train_dict = {}
        # test_dict = {}
        # cur = 0
        # for i in idx_train:
        #     if not i in train_dict.keys():
        #         train_dict[i] = cur
        #         cur += 1
        # cur = 0
        # for i in idx_test:
        #     if not i in test_dict.keys():
        #         test_dict[i] = cur
        #         cur += 1
        #
        # edges = self.edge_index.T
        # mask_train = torch.tensor([((s.item() in idx_train) and (t.item() in idx_train)) for (s,t) in edges[:]])
        # mask_test = ~mask_train
        # # self.train_edge = edges[(edges[:,0] in idx_train) & (edges[:,1] in idx_train),:].T
        # # self.test_edge = edges[(edges[:,0] in idx_test) & (edges[:,1] in idx_test),:].T
        # self.train_edge = edges[mask_train].T
        # self.test_edge = edges[mask_test].T
        # return self.train_x, self.train_edge, self.train_y, self.test_x, self.test_edge, self.test_y

def dataloader(args):
    if args['dataset'].lower() == 'cora':
        return cora_dataset(args)

def cora_dataset(args):
    data_root = args['data_root']
    '''
        下面的文件应该是./cora.cites 和 ./cora.content
    '''
    cite_path = os.path.join(data_root,'cora.cites')
    content_path = os.path.join(data_root,'cora.content')
    # assert os.path.exists(cite_path) and os.path.exists(content_path)
    if not os.path.exists(cite_path):
        error(f"{cite_path} not exits")
        return None
    if not os.path.exists(content_path):
        error(f"{content_path} not exits")
        return None
    edge_index = []
    node_dict = {}
    cur = 0
    with open(cite_path,'r') as reader:
        line = reader.readline()
        while not len(line) == 0:
            s,t = list(map(np.longlong, line.split()))
            if s not in node_dict.keys():
                node_dict[s] = cur
                cur = cur + 1
            if t not in node_dict.keys():
                node_dict[t] = cur
                cur = cur + 1
            edge_index.append([node_dict[s],node_dict[t]])
            line = reader.readline()
    edge_index = np.array(edge_index,dtype=np.longlong).T
    features = torch.zeros((len(node_dict.keys()),1433))
    targets = []
    target_dict = {}
    cur = 0
    with open(content_path,'r') as reader:
        line = reader.readline()
        while not len(line) == 0:
            line_split = line.split()
            node_idx = node_dict[np.longlong(line_split[0])]

            target = line_split[-1]
            if target not in target_dict.keys():
                target_dict[target] = cur
                cur += 1
            target = target_dict[target]
            targets.append(target)

            features[node_idx] = torch.tensor(list(map(np.longlong,line_split[1:-1])))

            line = reader.readline()
    return Dataset(features,torch.tensor(targets),node_dict,target_dict,torch.from_numpy(edge_index))
