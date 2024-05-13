import torch
from torch.nn import Dropout
from torch_geometric.nn import SAGEConv,GCNConv,GINConv,GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
import torch.nn as nn
from utils.utils import post_process

class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGENet, self).__init__()
        self.dpt1 = Dropout(p=0.5)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.dpt2 = Dropout(p=0.5)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, features, edge_index):
        # x = self.dpt1(features)
        x = self.conv1(features, edge_index)
        x = F.relu(x)
        x = self.dpt2(x)
        x = self.conv2(x, edge_index)
        return x

    def l2_loss(self):
        loss = None
        for p in self.conv1.parameters():
            if loss is None:
                loss = p.pow(2).sum()
            else:
                loss += p.pow(2).sum()
        # for p in self.conv2.parameters():
        #     if loss is None:
        #         loss = p.pow(2).sum()
        #     else:
        #         loss += p.pow(2).sum()
        return loss


class BaseLine(torch.nn.Module):
    def __init__(self, num_features, hidden_channels=32, num_layers=2, dropout_rate=0.5, gnn_type='GCN',gat_heads=None):
        super(BaseLine, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        if gnn_type == 'GCN':
            self.convs.append(GCNConv(num_features, hidden_channels))
        elif gnn_type == 'GIN':
            self.convs.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(num_features, hidden_channels, heads=gat_heads[0], concat=True))
        elif gnn_type.lower() == 'graphsage':
            self.convs.append(SAGEConv(num_features, hidden_channels))


        if gnn_type == 'GAT':
            self.bns.append(nn.BatchNorm1d(hidden_channels * gat_heads[0]))
        else:
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        for i in range(1, num_layers):
            if gnn_type == 'GCN':
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            elif gnn_type == 'GIN':
                self.convs.append(GINConv(torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels)
                )))
            elif gnn_type == 'GAT':
                self.convs.append(GATConv(hidden_channels * gat_heads[i-1], hidden_channels,heads=gat_heads[i], concat=True))
            elif gnn_type.lower() == 'graphsage':
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))

            self.bns.append(nn.BatchNorm1d(hidden_channels))

    def forward(self, data, nodeNumList):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if i < self.num_layers - 1:
                x = self.bns[i](x)
        out_post = post_process(x, nodeNumList)
        return out_post


class TjlNet(torch.nn.Module):
    def __init__(self,num_layers,GNN_model,input_dim,hidden_dim,activation='relu'):
        '''

        :param num_layers: GNN的层数
               GNN_model: GNN的类型，可以是GCN、GIN、GraphSAGE
               input_dim,hidden_dim: 输入、隐藏的维度
               activation: 激活函数类型
        '''
        super(TjlNet, self).__init__()
        self.num_layers = num_layers
        self.GNN_model = GNN_model
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = torch.nn.Dropout(p=0.5)
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        else:
            raise NotImplementedError
        self.convList, self.bns = self.init_model()

    def init_model(self):
        convList = nn.ModuleList()
        bns = nn.ModuleList()
        if self.GNN_model.lower() == 'gin':
            # 如果有多层，则input_dim -> hidden_dim -> output_dim
            for i in range(self.num_layers):
                if i == 0:
                    layer = nn.Sequential(
                        nn.Linear(self.input_dim,self.hidden_dim),
                        self.activation,
                        nn.Linear(self.hidden_dim,self.hidden_dim)
                    )
                else:
                    layer = nn.Sequential(
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        self.activation,
                        nn.Linear(self.hidden_dim, self.hidden_dim)
                    )
                convLayer = GINConv(nn=layer,aggr='sum')
                bn = nn.BatchNorm1d(self.hidden_dim)
                convList.append(convLayer)
                bns.append(bn)
        elif self.GNN_model.lower() == 'gcn':
            raise NotImplementedError
        else:
            raise NotImplementedError

        return convList, bns

    def forward(self, graphs, nodeNumList=None):
        # print(f"batch graph num:{len(nodeNumList)}")
        # # 输入一个数据集（可能有多个图），先获取每个图的节点数nodeNumList
        # nodeNumList = []
        # for i in range(graphs.len()):
        #     num_nodes = graphs[i].x.shape[0]
        #     nodeNumList.append(num_nodes)
        # print(max(nodeNumList))
        # 获取features、edge_index，这些应该是整个图的合集
        features, edge_index = graphs.x, graphs.edge_index
        x_record = []
        x_post_record = []
        out = features
        for i in range(self.num_layers):

            out = self.convList[i](out,edge_index)
            out = F.relu(out)
            out = self.bns[i](out)
            out = self.dropout(out)
            # 放入record
            x_record.append(out)
            if nodeNumList is not None:
                # 接下来要把 (19580,32)的feature矩阵变成(600,126,32)的，这个126是结点数的最大值
                out_post = post_process(out,nodeNumList)    # (600,126,32)
                x_post_record.append(out_post)
        if nodeNumList is not None:
            return torch.cat(x_record, -1), torch.cat(x_post_record, -1)
        else:
            return torch.cat(x_record, -1)

# 这个模型接受输入(node,graph)，预测他俩之间的互信息
# 互信息的计算可以用node的embedding过两层MLP，再与graph的embedding计算相似度
# 目前就简单地用node的embedding直接与graph计算相似度
class CalcModel(torch.nn.Module):
    def __init__(self,use_mlp=False,hidden_dim=0):
        super(CalcModel, self).__init__()
        self.MLP = None
        if use_mlp:
            self.MLP = nn.Sequential(
                nn.Linear(hidden_dim,hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,hidden_dim),
            )

    def forward(self,node):
        # if self.MLP is not None:
        #     node = self.MLP(node)
        # out = torch.sigmoid(F.cosine_similarity(node,graph,dim=-1))
        out = self.MLP(node)
        return out

# 计算节点级别对比学习的损失，计算公式是：
# -1/n * E(log(pred * mask_true) + log((1 - pred * mask_false)))
class NodeCLLoss(torch.nn.Module):
    def __init__(self,t=0.5):
        super(NodeCLLoss, self).__init__()
        self.t = t
    '''
        TODO：应该把全部的负样本（某个节点和其他全部的图的相似度）加起来，然后再用 exp(positive) / E(exp(negative))，
        来最大化更新网络的参数
    '''
    # def forward(self,pred,targets,delta=2):
    #     mask_true = targets.clone().type(torch.bool)
    #     mask_false = (1-targets).clone().type(torch.bool)
    #     log_positive = torch.zeros_like(pred)
    #     log_negative = torch.zeros_like(pred)
    #     log_positive[mask_true] = torch.log(pred[mask_true])
    #     log_negative[mask_false] = torch.log(1-pred[mask_false])
    #     log_all = log_positive + log_negative
    #     return -torch.mean(log_all)

    def forward(self,node_emd,positive_graph_emd,negative_graph_emd,mask):
        '''

        :param node_emd:    (bs,96)
        :param positive_graph_emd:  (bs,96)
        :param negative_graph_emd:  (bs,600,96)
        :param mask:    (bs,600)
        :return:
        '''
        positive_score = torch.exp(F.cosine_similarity(node_emd,positive_graph_emd,dim=-1) / self.t)   # (bs,)
        negative_score = torch.exp(F.cosine_similarity(node_emd.unsqueeze(1).repeat(1,negative_graph_emd.shape[1],1), negative_graph_emd,dim=-1) / self.t) # (bs,600)
        negative_score = torch.sum(negative_score * mask,dim=-1)    # (bs)
        # print(positive_score,negative_score)
        score = -torch.log(positive_score / negative_score)
        return score.mean()

class SimplePredictor(torch.nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(SimplePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,output_dim)
        )

    def forward(self,x):
        out = self.mlp(x)
        return out

class SuperSimplePredictor(torch.nn.Module):
    def __init__(self,input_dim,output_dim):
        super(SuperSimplePredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim,output_dim)
        )

    def forward(self,x):
        out = self.mlp(x)
        return out
class GraphClassificationPromptModel(torch.nn.Module):
    def __init__(self,targets_num,prompt_dim,cluster_num,prompt_num,cluster_centers=None):
        '''

        :param targets_num:     目标标签数（每个数据集的值不一样，但是固定的）
        :param prompt_dim:      每个prompt的维度(应该与node和graph的embedding维度一样)

        下面是超参数
        :param cluster_num:     把图聚类的聚类数
        :param prompt_num:      每一个类别里面的prompt数
        cluster_centers:        表示每个聚类的中心点，维度（cluster_num,prompt_dim）
        这个模块里面定义cluster_num个prompts池，每个池子里面有targets_num行，每一行代表对一个target的prompt预测，然后每一行有prompt_num个tokens，每一个token都有prompt_dim维的值
        也就是说prompts的形状是 (cluster_num, targets_num, prompt_num, prompt_dim)
        '''
        super(GraphClassificationPromptModel, self).__init__()
        if cluster_centers is None:
            self.prompts = nn.Parameter(torch.randn(cluster_num, targets_num, prompt_num, prompt_dim))
        else:
            self.prompts = nn.Parameter(torch.tensor(cluster_centers)[:,None,None,:].repeat((1,targets_num,prompt_num,1)))
            # print(self.prompts.shape)

    def forward(self,graph_emd,cluster_id):
        '''

        :param graph_emd:   (bs,96)
               cluster_id:  (bs,)，是这bs个图对应的聚类号
               由于选取的是某个聚类中的prompt，所以这个prompt的shape应该是(targets_num, prompt_num, prompt_dim),
               去预测这targets_num*prompt_num个节点所对应的图
        :return:
        返回的应该是，对于每个bs，预测出一个（targets_num, prompt_num）的矩阵，每个元素是相似度
        所以返回的格式应该是（bs, targets_num, prompt_num）
        '''
        a = self.prompts[cluster_id.type(torch.long)]   # (bs,6,10,96)
        b = graph_emd                                   # (bs,96)
        b = b[:,None,None,:].repeat((1,a.shape[1],a.shape[2],1))    # (bs,6,10,96)
        # print(a.shape)
        # print(b.shape)
        ret = F.cosine_similarity(a,b,dim=-1)
        return ret

class NodeCLLossDownStream(torch.nn.Module):
    def __init__(self,t=0.5):
        super(NodeCLLossDownStream, self).__init__()
        self.t = t

    def forward(self,pred,targets):
        '''
        :param pred: (bs,6,10) 表示预测的相似度
        :param targets: (bs,)  表示图的类别，对应pred的第2维
        :return:
        '''
        pred = torch.exp(pred / self.t)
        total_sum = pred.sum(-1).sum(-1)        # (bs,)
        row = torch.arange(0,pred.shape[0])     # 0 ~ bs-1
        col = targets
        # print(pred.sum(-1))
        positive_score = pred[row,col].sum(-1)  # (bs,)
        negative_score = total_sum - positive_score
        # print(f"positive:{positive_score}")
        # print(f"negative:{negative_score}")
        score = -torch.log(positive_score / negative_score)
        return score.mean()


# 使用节点嵌入
class GraphClassificationPromptModel2(torch.nn.Module):
    def __init__(self,targets_num,prompt_dim,cluster_num,prompt_num,cluster_centers=None):
        '''

        :param targets_num:     目标标签数（每个数据集的值不一样，但是固定的）
        :param prompt_dim:      每个prompt的维度(应该与node和graph的embedding维度一样)

        下面是超参数
        :param cluster_num:     把图聚类的聚类数
        :param prompt_num:      每一个类别里面的prompt数
        cluster_centers:        表示每个聚类的中心点，维度（cluster_num,prompt_dim）
        这个模块里面定义cluster_num个prompts池，每个池子里面有targets_num行，每一行代表对一个target的prompt预测，然后每一行有prompt_num个tokens，每一个token都有prompt_dim维的值
        也就是说prompts的形状是 (cluster_num, targets_num, prompt_num, prompt_dim)
        '''
        super(GraphClassificationPromptModel2, self).__init__()
        if cluster_centers is None:
            self.prompts = nn.Parameter(torch.randn(cluster_num, targets_num, prompt_num, prompt_dim))
        else:
            self.prompts = nn.Parameter(torch.tensor(cluster_centers)[:,None,None,:].repeat((1,targets_num,prompt_num,1)))
            # print(self.prompts.shape)

    def forward(self,node_emd,cluster_id):
        '''

        :param node_emd:   (bs,96)
               cluster_id:  (bs,)，是这bs个图对应的聚类号
               由于选取的是某个聚类中的prompt，所以这个prompt的shape应该是(targets_num, prompt_num, prompt_dim),
               去预测这targets_num*prompt_num个节点所对应的图
        :return:
        返回的应该是，对于每个bs，预测出一个（targets_num, prompt_num）的矩阵，每个元素是相似度
        所以返回的格式应该是（bs, targets_num, prompt_num）
        '''
        a = self.prompts[cluster_id.type(torch.long)]   # (bs,6,10,96)
        a = a[:,None,:,:,:].repeat((1,node_emd.shape[1],1,1,1)) # (bs,126,6,10,96)
        b = node_emd                                   # (bs,126,96)
        b = b[:,:,None,None,:].repeat((1,1,a.shape[2],a.shape[3],1))    # (bs,126,6,10,96)
        # print(a.shape)
        # print(b.shape)
        ret = F.cosine_similarity(a,b,dim=-1)   # (bs,126,6,10)
        return ret

class NodeCLLossDownStream2(torch.nn.Module):
    def __init__(self,t=0.5):
        super(NodeCLLossDownStream2, self).__init__()
        self.t = t

    def forward(self,pred,targets,node_list):
        '''
        :param pred: (bs,126,6,20) 表示预测的相似度
        :param targets: (bs,)  表示图的类别，对应pred的第3维
        node_list: (bs,) 每一个图的节点数，用于生成mask:(bs,126)
        :return:
        '''
        pred = torch.exp(pred / self.t)

        mask = torch.zeros(pred.shape[:2])  # (bs,126)
        for i in range(mask.shape[0]):
            mask[i,:node_list[i]] = 1

        pred = pred*mask[:,:,None,None].repeat(1,1,pred.shape[2],pred.shape[3])

        total_sum = pred.sum(-1).sum(-1).sum(-1)        # (bs,)
        row = torch.arange(0,pred.shape[0])     # 0 ~ bs-1
        node_ = torch.arange(0,pred.shape[1])    # 0~126-1
        col = targets
        # print(pred.sum(-1))
        positive_score = pred[row,:,col].sum(-1).sum(-1)  # (bs,)
        # print(positive_score.shape)
        negative_score = total_sum - positive_score
        # print(f"positive:{positive_score}")
        # print(f"negative:{negative_score}")
        score = -torch.log(positive_score / negative_score)
        return score.mean()
