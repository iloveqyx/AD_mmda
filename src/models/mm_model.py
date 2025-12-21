#我现在有了特征融合代码src/models/fusion_gated_attn.py
#也写了分类头代码src/models/classifier_head.py
#这个代码是为了实现多模态模型的整体结构，把融合和分类头结合起来
import torch
from .fusion_gated_attn import GatedAttnFusionNet
from .classifier_head import ClassifierHead

class MMModel(torch.nn.Module):
    def __init__(self,
                 da:int,
                 dt:int,
                 num_classes:int,
                 dproj:int=256,
                 nhead:int=4,
                 gate_mode:str="channel",
                 cls_hidden_dim:int=256,
                 pdrop:float=0.1):
        super().__init__()
        #融合网络
        self.fusion_net = GatedAttnFusionNet(da, dt, dproj, nhead, pdrop, gate_mode)
        #分类
        self.classifier_head = ClassifierHead(input_dim=dproj, num_classes=num_classes,
                                     hidden_dim=cls_hidden_dim, pdrop=pdrop)
    def forward(self, Xa:torch.Tensor, Xt:torch.Tensor):
        za,zt,zf,g = self.fusion_net(Xa, Xt)  #融合特征
        logits , probs = self.classifier_head(zf)  #分类
        return logits, probs, (za, zt, zf, g)
    
    #到这里，打标签工作只剩和相似度结合然后决定伪标签了。