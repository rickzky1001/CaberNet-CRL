from utils.feature_engineering import load_processed_ds
from utils.dataset_split import train_val_test_split,train_val_k_test_split,train_val_test_split_by_domain,train_val_k_test_split_by_domain
import torch
from utils.seed import set_seed
from utils.train import domain_wise_training
from model.my_model import my_model
import numpy as np
import torch

def compute_representation_jacobian(model, X, device="cuda"):
    """
    计算每个 raw feature 对 representation h 的贡献
    Args:
        model: my_model 实例
        X: 输入张量, shape [B, T, d]
    Returns:
        contrib_matrix: [H, d]，表示每个 latent dim 对每个 raw feature 的敏感度
    """
    X = X.to(device)
    X.requires_grad_(True)

    # forward 到 h
    f = X[:, :, :-4]   # 去掉最后4个量纲特征
    f_gated, _ = model.f_selector(f)

    h, _ = model.lstm(f_gated)
    h = h[:, -1, :]    # [B, H]


    H = h.shape[1]
    d = f_gated.shape[2]     # raw features 数量

    contrib_matrix = torch.zeros(H, d, device=device)

    for j in range(H):
        grad = torch.autograd.grad(
            outputs=h[:, j].sum(),
            inputs=f,
            create_graph=False,
            retain_graph=True
        )[0]  # [B, T, d]

        # 对 batch 和时间步平均
        grad_matrix = grad.abs().mean(dim=(0, 1))  # [d]
        contrib_matrix[j] = grad_matrix

    return contrib_matrix.detach().cpu()

def jacobian_debias(J, gate, eps=1e-8):
    # 转 tensor
    if isinstance(J, np.ndarray):
        J = torch.tensor(J, dtype=torch.float32)
    if isinstance(gate, np.ndarray):
        gate = torch.tensor(gate, dtype=torch.float32)
    # softmax 归一化得到 g
    g = torch.softmax(gate, dim=0) if gate.dim() == 1 else gate
    # 逐列消除 gate (等价于右乘 diag(1/g))
    J_no_gate = J / (g.unsqueeze(0) + eps)
    return J_no_gate

if __name__=='__main__':
    seed = 43
    set_seed(seed)

    epochs=6
    lr=3e-3
    input_size=8 # (不包括量纲特征)
    hidden_size=5


    num_layer=1
    batch_size = 510
    shuffle=False #对llo无效
    time_window_step = 12
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset=load_processed_ds(folder='result/processed_ds')

    f_columns = dataset['Beijing'][22][0].columns.drop('total_energy')
    train_dic={'Beijing': [23], 'Hefei': [9,10], 'Huizhou': [15,16],'Jiling':[]}
    val_dic={'Beijing': [], 'Hefei': [], 'Huizhou': [],'Jiling':[]}
    test_dic={'Beijing': [22], 'Hefei': [], 'Huizhou': [],'Jiling':[]}

    combined_train_dic = {city: train_dic.get(city, []) + val_dic.get(city, []) for city in set(train_dic) | set(val_dic)}
    train_sets, val_sets, test_sets = train_val_k_test_split_by_domain(dataset, train_dic=combined_train_dic, test_dic=test_dic, standardization='zscore', exclude_columns=[],val_ratio=0.2, duration=12, shuffle=shuffle)

    lambda_var=2e-2
    lambda_sparse=5e-4
    lambda_indy=1e-1

    model = my_model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer,use_h_selector=False,use_f_selector_gumbel=False,use_f_selector_gate=True,use_indy_loss=True).to(device)
    with torch.no_grad():
        model.f_selector.gates.copy_(torch.tensor([-0.05107395, -0.06949169,  0.11833728, -0.04039218, -0.11290828, -0.07613983, 0.09646182,  0.08395815]))
    model.f_selector.gates.requires_grad_(False)
    train_losses,test_losses=domain_wise_training(model=model,use_my_model=True,lambda_var=lambda_var,lambda_sparse=lambda_sparse,lambda_indy=lambda_indy,device=device,train_sets=train_sets,val_sets=val_sets,test_sets=test_sets,epoch=epochs,lr=lr,batch_size=batch_size,warmup_epochs=1000)
    np.save('result/SCM/gates.npy',model.f_selector.gates.detach().cpu().numpy())
    X_train, _, _, _,X_test, _ = train_val_k_test_split(dataset, train_dic=combined_train_dic, test_dic=test_dic, exclude_columns=[],standardization='zscore', val_ratio=0.2, duration=12, shuffle=shuffle)
    X=X_train
    X = torch.tensor(X, dtype=torch.float32).to(device)
    model.train()
    gate = np.load("result/SCM/gates.npy")
    J = compute_representation_jacobian(model, X,device=device)
    J_debiased=jacobian_debias(J,gate)
    np.save("result/SCM/J_matrix.npy", J.detach().cpu().numpy())
    np.save("result/SCM/J_debiased_matrix.npy", J_debiased.detach().cpu().numpy())

    y_pred, z,_,_= model(X)

    import lingam
    z=z.detach().cpu().numpy()
    y_pred=y_pred.detach().cpu().numpy()
    variable = np.concatenate((z, y_pred.reshape(-1, 1)), axis=1)
    variable= np.asarray(variable)
    ling = lingam.DirectLiNGAM(random_state=seed)
    ling.fit(variable)
    order = ling.causal_order_            # 因果顺序
    B = ling.adjacency_matrix_            # 邻接矩阵(系数)
    np.save("result/SCM/B_matrix.npy", B)