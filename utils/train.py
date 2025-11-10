import torch
from utils.dataset_split import train_val_test_split_by_domain,MixedDomainDataset,collate_fn,load_processed_ds
from torch.utils.data import DataLoader
from model.my_model import my_model
from utils.loss import nmse_loss
import sys
# domain_wise_training：
# 1.以每floor为domain, 计算每个domain的nmse, 然后进行算术平均做为loss
# 2.确保了每个batch为混合的domain数据
def domain_wise_training(model,use_my_model,device,train_sets,val_sets,test_sets,lambda_var,lambda_sparse,lambda_indy,epoch,lr,batch_size):
    sparse_loss_method = 'l1BER'
    print(f"--- Using {sparse_loss_method} for sparse loss. ---")
    train_domain2id = {dk: i for i, dk in enumerate(train_sets.keys())}
    val_domain2id = {dk: i for i, dk in enumerate(val_sets.keys())}
    test_domain2id = {dk: i for i, dk in enumerate(test_sets.keys())}
    train_dataset = MixedDomainDataset(train_sets,train_domain2id)
    train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
    )
    difficulty_weights_train = calculate_cv_difficulty_weights(train_sets, train_domain2id).to(device)
    difficulty_weights_val = calculate_cv_difficulty_weights(val_sets, val_domain2id).to(device)
    # if warmup_epochs > 0:
    #     # 确保 f_selector 存在
    #     if model.use_f_selector_gate:
    #         for param in model.f_selector.parameters():
    #             param.requires_grad = False
    #         print(f"--- Freezing feature selector parameters. ---")

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = {dk: [] for dk in train_domain2id.values()}
    val_losses = {dk: [] for dk in val_domain2id.values()}
    test_losses = {dk: [] for dk in test_domain2id.values()}
    for epoch in range(epoch):

        model.train()
        domain_losses={dk: [] for dk in train_domain2id.values()}
        var_losses = []
        sparse_losses = []
        indy_losses = []
        for X, y, dom in train_loader:
            X, y = X.to(device), y.to(device)
            if use_my_model:
                y_pred, h,  f_sigmoid_activation= model(X)
            else:
                y_pred = model(X)
            domain_nmses = [] 
            for domain_id in train_domain2id.values():
                idx = (dom==domain_id)
                p, t = y_pred[idx], y[idx]
                nmse= nmse_loss(p, t)
                diff_weight = 1.0 / difficulty_weights_train[domain_id]
                domain_losses[int(domain_id)].append(nmse.item())
                domain_nmses.append(diff_weight*nmse)
            stacked = torch.stack(domain_nmses)
            reconstruction_loss = torch.nanmean(stacked)
            var = torch.var(stacked, unbiased=True)
            var_losses.append(var.item())  
            loss= reconstruction_loss + lambda_var * var
            # indy
            if h is not None:
                indy_loss = calculate_indy_loss(h)
                loss += lambda_indy * indy_loss
                indy_losses.append(indy_loss.item())
            # sparse
            if f_sigmoid_activation is not None: #gate
                sparse_loss = calculate_f_sigmoid_activation_loss(f_sigmoid_activation, method=sparse_loss_method,epoch=epoch)
                loss += lambda_sparse * sparse_loss
                sparse_losses.append(sparse_loss.item())
            opt.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #梯度裁剪
            opt.step()
        for domain_id in train_domain2id.values():
            loss_list = domain_losses[domain_id]
            train_losses[int(domain_id)].append(torch.nanmean(torch.tensor(loss_list)).item()) #nmse
        avg_rec_loss = torch.nanmean(torch.tensor([domain_losses[i] for i in train_domain2id.values()])).item()
        avg_var_loss = torch.mean(torch.tensor(var_losses)).item()
        print_str = f'epoch{epoch+1} train: avg_rec_loss: {avg_rec_loss:.4f}, var_loss: {lambda_var * avg_var_loss:.4f}'
        if indy_losses:
            avg_indy_loss = torch.mean(torch.tensor(indy_losses)).item()
            print_str += f', indy_loss: {lambda_indy * avg_indy_loss:.4f}'
        if sparse_losses:
            avg_sparse_loss = torch.mean(torch.tensor(sparse_losses)).item()
            print_str += f', sparse_loss: {lambda_sparse * avg_sparse_loss:.4f}'
        print_str += f', gates: {model.f_selector.gates.detach().cpu().numpy()}'
        if epoch+1 >0:
            print(print_str,end='-----')
        
        else:
            print(print_str)
            continue
        model.eval()
        with torch.no_grad():
            for (city, L), data in val_sets.items():
                domain_id = val_domain2id[(city, L)]
                X_val = torch.tensor(data['X'], dtype=torch.float32).to(device)
                y_val = torch.tensor(data['y'], dtype=torch.float32).to(device).unsqueeze(1)
                if use_my_model:
                    y_pred,_,_ = model(X_val)
                else:
                    y_pred = model(X_val)
                # 计算 NMSE
                nmse= nmse_loss(y_pred, y_val)
                diff_weight = 1.0 / difficulty_weights_val[domain_id]
                val_losses[domain_id].append(diff_weight*nmse.item())
            print(f"val: {torch.mean(torch.tensor([val_losses[i][-1] for i in val_domain2id.values()])).item()}", end='-----')
            for (city, L), data in test_sets.items():
                domain_id = test_domain2id[(city, L)]
                X_test = torch.tensor(data['X'], dtype=torch.float32).to(device)
                y_test = torch.tensor(data['y'], dtype=torch.float32).to(device).unsqueeze(1)
                if use_my_model:
                    y_pred,_,_ = model(X_test)
                else:
                    y_pred = model(X_test)
                # 计算 NMSE
                nmse= nmse_loss(y_pred, y_test)
                test_losses[domain_id].append(nmse.item())
            print(f"test: {torch.mean(torch.tensor([test_losses[i][-1] for i in test_domain2id.values()])).item()}")
        
    return train_losses,val_losses

def calculate_indy_loss(h):
    """
    Normalized Independence Loss:
    Average absolute correlation between different features
    """
    if h is None or h.shape[0] < 2:
        return torch.tensor(0.0, device=h.device)

    corr_matrix = torch.corrcoef(h.T)  # shape: [d, d]
    identity_matrix = torch.eye(corr_matrix.shape[0], device=h.device)
    
    # mask to ignore diagonal
    off_diag_mask = ~torch.eye(corr_matrix.shape[0], dtype=bool, device=h.device)
    off_diag_corr = torch.abs(corr_matrix - identity_matrix)[off_diag_mask]

    # normalize by number of off-diagonal elements
    loss = off_diag_corr.mean()
    return loss
def calculate_f_sigmoid_activation_loss(f_sigmoid_activation,method,epoch):
    if method == 'l1': #仅适用于l1
        sparse_loss = torch.norm(f_sigmoid_activation, p=1)
    elif method == 'BER': #仅适用于sigmoid
        sparse_loss = -torch.sum(
        f_sigmoid_activation * torch.log(f_sigmoid_activation) + 
        (1 - f_sigmoid_activation) * torch.log(1 - f_sigmoid_activation))
    elif method == 'l1BER': #适用于sigmoid
        if epoch < 1e+3: 
            sum_loss = torch.norm(f_sigmoid_activation,p=1)
        else:
            if epoch == 1e+3:
                print(f"--- Excluding l1 loss for feature selection. ---")
            sum_loss = torch.tensor(0.0, device=f_sigmoid_activation.device)
        BER_loss = -torch.sum(
            f_sigmoid_activation * torch.log(f_sigmoid_activation) +
            (1 - f_sigmoid_activation) * torch.log(1 - f_sigmoid_activation))
        sparse_loss = 10*sum_loss + BER_loss
    return sparse_loss / len(f_sigmoid_activation)  # 除以特征数做归一化
import numpy as np
def calculate_cv_difficulty_weights(train_sets: dict, train_domain2id: dict) -> torch.Tensor:
    """
    使用X和y的变异系数(CV)的均值来计算每个domain的难度系数。
    """
    num_domains = len(train_domain2id)
    difficulties = torch.ones(num_domains)

    print("Calculating CV-based difficulty weights for each domain...")
    for domain_key, data in train_sets.items():
        domain_id = train_domain2id[domain_key]
        # 我们只取一个时间步的数据来计算，因为特征在时间步上是重复的
        X = data['X'][:, 0, :-4] 
        y = data['y']

        # --- 1. 计算 y 的变异系数 ---
        y_mean_abs = np.mean(np.abs(y))
        y_std = np.std(y)
        cv_y = y_std / (y_mean_abs)
        # --- 2. 计算 X 的平均变异系数 ---
        # 逐列计算CV
        X_mean_abs = np.mean(np.abs(X), axis=0)
        X_std = np.std(X, axis=0)
        cv_X_cols = X_std / (X_mean_abs)
        # 取所有特征CV的平均值
        cv_X = np.nanmean(cv_X_cols) # 使用nanmean以防某些列的CV是NaN

        # --- 3. 综合难度 ---
        # 将两者的CV相加作为综合难度。+1作为平滑项。
        combined_difficulty = np.mean([cv_y, cv_X])
        difficulties[domain_id] = combined_difficulty
        print(f"  Domain {domain_key}: CV_y={cv_y:.4f}, CV_X={cv_X:.4f}, Combined={combined_difficulty:.4f}")

    return difficulties