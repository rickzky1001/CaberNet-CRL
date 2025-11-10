import torch
import torch.nn as nn
from typing import List

class cond_nn(nn.Module):
    def __init__(self, layer_dims: List[int]):
        super(cond_nn, self).__init__()
        
        layers = []
        # 遍历维度列表，创建线性层和激活函数
        for i in range(len(layer_dims) - 1):
            in_features = layer_dims[i]
            out_features = layer_dims[i+1]
            
            # 添加线性层
            layers.append(nn.Linear(in_features, out_features))
            
            # 在除最后一层外的所有层之后添加ReLU激活函数
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
                
        # 使用 Sequential 容器组合所有层
        self.mlp = nn.Sequential(*layers)
        self.relu=nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.mlp(x))
