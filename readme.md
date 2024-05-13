要运行本框架（包括预训练和下游任务过程，只需运行main.py）：

```shell
python main.py
```

相关参数在main.py中更改，以下是部分参数的解释：

```
need_pretrain: 是否需要进行预训练，如果已经有预训练好的模型则置为False
check_point_model：预训练模型的名称
k：k-shot的参数k
prompt_num：下游任务中每个图类别设置的提示向量数量
cluster_num：下游任务的聚类数
downstream_method：可选项为：'node','graph','mlp'，其中node代表进行NMbGP-N，graph代表NMbGP-G，mlp代表NMbGP-MLP方法
```

baseline方法对应的文件是base.py：

```shell
python base.py
```

通过修改main.py中参数`baseline`来修改baseline的类型，可选项有：`gcn,graphsage,gat,gin`.