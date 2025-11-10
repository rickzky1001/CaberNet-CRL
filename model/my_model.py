import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.seed import set_seed
from model.cond_nn import cond_nn
    
class f_selector_gate(nn.Module):
    def __init__(self, num_features: int,activation):
        super().__init__()
        # self.gates = nn.Parameter(torch.full((num_features,), -1, dtype=torch.float))
        self.gates = nn.Parameter(torch.empty(num_features).normal_(mean=0.01, std=0.0))
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = lambda x: x
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=0)
        elif activation == 'relu':
            self.activation = nn.ReLU()
        print(f'using {self.activation} activation')
    def forward(self, x: torch.Tensor):
        sigmoid_activated = self.activation(self.gates)
        weights = self.activation(self.gates)
        gated_x = x * weights[None, None, :]  # 广播到 (B, T, D)
        return gated_x, sigmoid_activated
    

class my_model(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, topk_h=0.8, topk_f=0.8, tau=1.0,use_f_selector_gumbel=False,use_f_selector_gate=False,use_indy_loss=False):
        super().__init__()
        if use_f_selector_gate:
            self.f_selector = f_selector_gate(num_features=input_size,activation='softmax')
        else:
            self.f_selector = None
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.cond_nn = cond_nn([4,int(hidden_size/2),int(hidden_size),int(hidden_size*2)])
        
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        self.use_f_selector_gate = use_f_selector_gate  
        self.use_indy_loss = use_indy_loss
        self.hidden_size = hidden_size
    def forward(self, x):
        f=x[:,:, :-4]
        l= x[:, 0, -4:] 
        if self.use_f_selector_gate:
            f, f_sigmoid_activation = self.f_selector(f)
        h, _ = self.lstm(f)  # 忽略最后两维（量纲特征）
        h= h[:, -1, :]  # 取最后一个时间步的输出
        factor = self.cond_nn(l)  # 使用最后两维作为条件输入
        out = h * factor[:,:self.hidden_size] + factor[:,self.hidden_size:]  # 量纲特征的仿射变换
        h_selected = out
        out = self.relu(self.fc(h_selected))
        return out, h if self.use_indy_loss else None,  f_sigmoid_activation if self.use_f_selector_gate else None

def domain_wise_test(model, test_sets, model_name,device):
    from utils.loss import nmse_loss
    model.eval()
    for i, ((city, L), data) in enumerate(test_sets.items()):
        X_test = torch.tensor(data['X'], dtype=torch.float32).to(device)
        y_test = data['y']
        if model_name=='my_model':
            y_pred = model(X_test)[0].cpu().detach().numpy()
        elif model_name=='lstm':
            y_pred = model(X_test).cpu().detach().numpy()
        nmse = nmse_loss(y_pred, y_test)
        print(f"Test NMSE for {city} L{L}: {nmse.item()}")

def print_gate_params(model,f_columns):
    if hasattr(model, 'use_f_selector_gate') and model.use_f_selector_gate:
        # 获取最终的门控权重
        final_gates = model.f_selector.gates.detach().cpu()
        weights = torch.sigmoid(final_gates)  # 归一化为概率分布
        sorted_weights, sorted_indices = torch.sort(weights, descending=True)
        print("\n--- Sorted Feature Importance Weights ---")
        for i in range(len(sorted_weights)):
            idx = sorted_indices[i].item()
            print(f"{f_columns[idx]}: sigmoid Weight = {sorted_weights[i]:.4f}")
