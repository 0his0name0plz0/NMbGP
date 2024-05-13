import sys
import torch
import numpy as np
import random
sys.path.append('./')

from train.train_2 import main

def get_args():
    args={
        'dataset':'PROTEINS',   #   PROTEINS,ENZYMES,PROTEINS_full,Synthie
        'batch_size':2048,
        'pretrain_epoch':200,
        'need_pretrain':False,
        'check_point_model':'epoch_200_NodeCLModel_withoutAGMTT_withMLP.pth',
        'calc_model':'epoch_200_CalcModel.pth',
        'k':20,
        'prompt_num':50,
        'downstream_epoch':40,
        # 聚类参数
        'cluster_num':8,

        'lr_ds':5e-2,
        'hidden_dim':32,

        'downstream_method':'graph',     # node,graph,mlp
        'device':'cpu',
        'baseline':'gin'    # gcn,graphsage,gat,gin

    }
    return args

def set_seed(seed):
    print(f"seed:{seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    args = get_args()
    seeds = [0,10,100,1000,10000,20000,1314520,2000000]
    # seeds = [0]
    for seed in seeds:
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>seed: {seed}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        set_seed(seed)         # 10 100 1000 10000 1000000 2000000
        main(args)
