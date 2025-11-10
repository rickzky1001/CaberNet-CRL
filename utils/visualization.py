import matplotlib.pyplot as plt
import torch
import os
import shutil
import sys
#可视化结果窗口化
def time_windowization(pred,true,time_window_step=12):
    final=pred.shape[0]-(pred.shape[0]%time_window_step)
    new_pred=pred[:final].reshape(-1,time_window_step).sum(axis=1)
    new_true=true[:final].reshape(-1,time_window_step).sum(axis=1)
    return new_pred,new_true

def pred_plot(pred,true,time_window_step,graph_title):
    new_pred,new_true=time_windowization(pred,true,time_window_step=time_window_step)
    plt.figure(figsize=(12, 6))
    plt.plot(new_true, label='True Value', alpha=0.7)
    plt.plot(new_pred, label='Predicted Value', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('True vs Predicted')
    plt.legend()
    plt.grid(True)
    if 'lstm' in graph_title:
        model = 'lstm'
    elif 'my_model' in graph_title:
        model = 'my_model'
    elif 'xgboost' in graph_title:
        model = 'xgboost'
    else:
        sys.exit(0)
        print('无法保存')
    save_dir = f'result/graphs/{model}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{graph_title}.png')
    plt.close()
def domain_pred_plot(model, train_sets,val_sets,test_sets, time_window_step,model_name,device):
# 绘制所有预测图
    for i, ((city, L), data) in enumerate(train_sets.items()):
        X_train = torch.tensor(data['X'], dtype=torch.float32).to(device)
        y_train = data['y']
        if model_name=='my_model':
            y_train_pred = model(X_train)[0].cpu().detach().numpy()
        elif model_name=='lstm':
            y_train_pred = model(X_train).cpu().detach().numpy()
        pred_plot(y_train_pred, y_train, time_window_step=time_window_step, graph_title=f'{model_name}_train_{city}_L{L}')
    for i, ((city, L), data) in enumerate(val_sets.items()):
        X_val = torch.tensor(data['X'], dtype=torch.float32).to(device)
        y_val = data['y']
        if model_name=='my_model':
            y_pred = model(X_val)[0].cpu().detach().numpy()
        elif model_name=='lstm':
            y_pred = model(X_val).cpu().detach().numpy()
        pred_plot(y_pred, y_val, time_window_step=time_window_step, graph_title=f'{model_name}_val_{city}_L{L}')
    for i, ((city, L), data) in enumerate(test_sets.items()):
        X_test = torch.tensor(data['X'], dtype=torch.float32).to(device)
        y_test = data['y']
        if model_name=='my_model':
            y_pred = model(X_test)[0].cpu().detach().numpy()
        elif model_name=='lstm':
            y_pred = model(X_test).cpu().detach().numpy()
        pred_plot(y_pred, y_test, time_window_step=time_window_step, graph_title=f'{model_name}_test_{city}_L{L}')