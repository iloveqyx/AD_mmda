#通过 src/prototypes/cpmpute_prototype.py 得到了每个类别的原型后，接下来计算相似度，但是这里还没有考虑伪标签的问题
import torch
import torch.nn.functional as F
from typing import Tuple

#==== 计算余弦相似度 ====
def cosine_similarity_prototype(
        z: torch.Tensor,  # [B, L, D] or [B, D]
        prototype: torch.Tensor,  # [num_classes, D]
        tau: float = 20.0, # 温度参数,数值越大越尖锐
) -> Tuple[torch.Tensor, torch.Tensor]:
    #这里的Tuple是返回两个值，第一个是相似度，第二个是预测的类别，Tuple的作用类似于一个容器，可以包含多个不同类型的值
    
    #不确定这里还需不需要进行归一化处理,因为在compute_prototype中已经归一化过了，如果出错改这里
    z_norm = F.normalize(z, p=2, dim=-1)
    prototype_norm = F.normalize(prototype, p=2, dim=-1)

    #cosine 相似度
    logits = tau*(z_norm @ prototype_norm.t())  #.t  转置,@ 表示矩阵乘法,结果形状 [B, L, num_classes] or [B, num_classes]
    sim_prototype = F.softmax(logits, dim=-1)  # 预测类别,-1表示最后一个维度

    return sim_prototype, logits