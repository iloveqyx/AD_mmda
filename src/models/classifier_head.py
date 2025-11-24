#这个模块是通过src/models/fusion_gated_attn.py得到的zf进行一次分类，后续再根据相似度来确定是否要打上伪标签。
#目前分类头比较简单，就是一个线性层，但是由于最后要对所有样本进行对比学习，所以这里的分类头后续还会因为对比学习而再训练。
import torch
import torch.nn as nn
import torch.nn.functional as F

#简单的MLP分类头
class ClassifierHead(nn.Module):
    def __init__(self,
                 input_dim:int,
                 num_classes:int,
                 hidden_dim:int=256,
                 pdrop:float=0.1):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),# 输入维度到隐藏维度
                nn.ReLU(inplace=True),# 激活函数
                nn.Dropout(pdrop),# 随机置零,在门控那里也有，一样的
                nn.Linear(hidden_dim, num_classes)# 隐藏维度到类别数
            )
        else:
            self.net = nn.Linear(input_dim, num_classes)


    def forward(self, z:torch.Tensor):
        if z.dim() == 3:
#            B,L,D = z.shape
#            z = z.view(B*L, D)  # 展平为 [B*L, D]
#===不知道哪个效果会更好，mean之后再分类，还是直接展平分类===AI说池化效果会更好一些
            z = z.mean(dim=1)  # 对序列维度取平均，变为 [B, D]
        logits = self.net(z)  # net
        probs = F.softmax(logits, dim=-1)  # 预测类别概率
        return logits, probs