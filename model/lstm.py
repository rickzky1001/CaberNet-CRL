import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model.cond_nn import cond_nn
import numpy as np
import matplotlib.pyplot as plt
from utils.train import MixedDomainDataset, collate_fn,calculate_cv_difficulty_weights
import random
from utils.loss import nmse_loss
class lstm(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.cond_nn = cond_nn([4,int(hidden_size/2),int(hidden_size),int(hidden_size*2)])

        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
    def forward(self, x):
        out, _ = self.lstm(x[:, :, :-4])  # 忽略最后（量纲特征）
        factor = self.cond_nn(x[:, -1, -4:])  # 使用最后作为条件输入
        out = out[:, -1, :]*factor[:,:self.hidden_size] + factor[:,self.hidden_size:]  # 量纲特征的仿射变换
        out = self.relu(self.fc(out))
        return out
class lstm_baseline(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, output_size=1):
        super(lstm_baseline, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
    def forward(self, x):
        out, _ = self.lstm(x)  # 忽略最后（量纲特征）
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.relu(self.fc(out))
        return out
def lstm_train(model,X_train, X_val, y_train, y_val, device,batch_size,epoch=50, lr=0.01):

    # 转换为张量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    # 创建 Dataset 和 DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 打乱

    criterion = nmse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 存储每个 epoch 的 loss
    train_losses = []
    val_losses = []

    # 开始训练
    for e in range(epoch):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        total_loss = total_loss / len(train_loader)
        train_losses.append(total_loss)

        # 验证
        model.eval()
        with torch.no_grad():
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val.view(-1, 1)).item()
            val_losses.append(val_loss)

        print(f"Epoch {e+1}, Train Loss: {total_loss:.8f}, Val Loss: {val_loss:.8f}")
    # 画训练 loss 曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Val Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('result/graphs/lstm_training_curve.png')
    plt.close()
    return model
def lstm_domain_wise_training(model,device,train_sets,val_sets, test_sets,epoch,lr,batch_size):
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
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = {dk: [] for dk in train_domain2id.values()}
    val_losses = {dk: [] for dk in val_domain2id.values()}
    test_losses = {dk: [] for dk in test_domain2id.values()}
    for epoch in range(epoch):

        model.train()
        domain_losses={dk: [] for dk in train_domain2id.values()}
        for X, y, dom in train_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            domain_nmses = []
            for domain_id in train_domain2id.values():
                idx = (dom==domain_id)
                p, t = y_pred[idx], y[idx]
                nmse= nmse_loss(p, t)
                diff_weight = 1.0 / difficulty_weights_train[domain_id]
                domain_losses[int(domain_id)].append(nmse.item())
                domain_nmses.append(diff_weight*nmse)
            loss = torch.nanmean(torch.stack(domain_nmses))
            opt.zero_grad()
            loss.backward()
            opt.step()
        for domain_id in train_domain2id.values():
            loss_list = domain_losses[domain_id]
            train_losses[int(domain_id)].append(torch.nanmean(torch.tensor(loss_list)).item()) #nmse
        avg_rec_loss = torch.nanmean(torch.tensor([domain_losses[i] for i in train_domain2id.values()])).item()
        print_str = f'epoch{epoch+1} train: avg_rec_loss: {avg_rec_loss:.4f}'
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
                y_pred = model(X_test)
                # 计算 NMSE
                nmse= nmse_loss(y_pred, y_test)
                test_losses[domain_id].append(nmse.item())
            print(f"test: {torch.mean(torch.tensor([test_losses[i][-1] for i in test_domain2id.values()])).item()}")
    return train_losses,val_losses
def lstm_test(model,X_test,y_test,device):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        pred = model(X_test.to(device)).cpu().numpy()
        true = y_test
    return pred,true
