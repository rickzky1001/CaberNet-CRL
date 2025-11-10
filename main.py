from utils.feature_engineering import load_processed_ds
from utils.dataset_split import train_val_test_split,train_val_k_test_split,train_val_test_split_by_domain,train_val_k_test_split_by_domain
from model.lstm import lstm,lstm_test,lstm_train,lstm_domain_wise_training
import torch
from utils.seed import set_seed
from utils.visualization import pred_plot, domain_pred_plot
from model.my_model import my_model,domain_wise_test,print_gate_params
from utils.train import domain_wise_training
import argparse
from utils.train import nmse_loss
from utils.loss import nmse_loss
import sys
import os
import shutil
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['lstm', 'my_model'], help='Model type to use')
parser.add_argument('--train', type=str, default='normal', choices=['normal', 'domain_wise'])
parser.add_argument('--LLO', type=int, default=1, choices=[1,0], help='Whether to use leave-one-out validation')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=3e-6)
args = parser.parse_args()
if __name__=='__main__':
    seed = 41
    set_seed(seed)

    input_size=8 # (不包括量纲特征)
    hidden_size=64
    num_layer=1
    batch_size = 510
    shuffle=False #对llo无效
    time_window_step = 12
    lingam_threshold= 0

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset=load_processed_ds(folder='result/processed_ds')

    f_columns = dataset['Beijing'][22][0].columns.drop('total_energy')
    train_dic={'Beijing': [22], 'Hefei': [9,10], 'Huizhou': [15,16],'Jiling':[]}
    val_dic={'Beijing': [], 'Hefei': [], 'Huizhou': [],'Jiling':[]}
    test_dic={'Beijing': [23], 'Hefei': [], 'Huizhou': [],'Jiling':[]}

    exclude_columns = []  # 排除的特征

    if args.model == 'my_model':
        use_my_model=True
        if args.LLO == 1:
            train_sets, val_sets, test_sets = train_val_test_split_by_domain(dataset,train_dic=train_dic, val_dic=val_dic, test_dic=test_dic,standardization='zscore',duration=12,exclude_columns=exclude_columns)
        elif args.LLO == 0:
            combined_train_dic = {city: train_dic.get(city, []) + val_dic.get(city, []) for city in set(train_dic) | set(val_dic)}
            train_sets, val_sets, test_sets = train_val_k_test_split_by_domain(dataset, train_dic=combined_train_dic, test_dic=test_dic, standardization='zscore', exclude_columns=exclude_columns,val_ratio=0.2, duration=12, shuffle=shuffle)
        
        # 训练模型
        lambda_var=2e-2
        lambda_sparse=5e-4
        # lambda_indy=5e-4
        lambda_indy=0

        model = my_model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer,use_f_selector_gumbel=False,use_f_selector_gate=True,use_indy_loss=True).to(device)
        
        train_losses,test_losses=domain_wise_training(model=model,use_my_model=use_my_model,lambda_var=lambda_var,lambda_sparse=lambda_sparse,lambda_indy=lambda_indy,device=device,train_sets=train_sets,val_sets=val_sets,test_sets=test_sets,epoch=args.epochs,lr=args.lr,batch_size=batch_size)
        save_dir = f'result/graphs/{args.model}'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        domain_pred_plot(model=model,train_sets=train_sets, val_sets=val_sets,test_sets=test_sets, time_window_step=time_window_step,model_name='my_model',device=device)
        domain_wise_test(model=model, test_sets=test_sets, model_name='my_model',device=device)
        print_gate_params(model,f_columns)
    elif args.model == 'lstm':
        
        if args.LLO == 1:
            if args.train == 'domain_wise':
                train_sets, val_sets, test_sets = train_val_test_split_by_domain(dataset, train_dic=train_dic,val_dic=val_dic, test_dic=test_dic, standardization='zscore', exclude_columns=exclude_columns, duration=12)
            elif args.train == 'normal':
                X_train, y_train, X_val, y_val,X_test, y_test = train_val_test_split(dataset, train_dic=train_dic, val_dic=val_dic,test_dic=test_dic,standardization='zscore', duration=12)
        elif args.LLO == 0:
            combined_train_dic = {city: train_dic.get(city, []) + val_dic.get(city, []) for city in set(train_dic) | set(val_dic)}
            if args.train == 'domain_wise':
                train_sets, val_sets, test_sets = train_val_k_test_split_by_domain(dataset, train_dic=combined_train_dic, test_dic=test_dic, standardization='zscore', exclude_columns=exclude_columns,val_ratio=0.2, duration=12, shuffle=shuffle)
            elif args.train == 'normal':
                X_train, y_train, X_val, y_val,X_test, y_test = train_val_k_test_split(dataset, train_dic=combined_train_dic, test_dic=test_dic, exclude_columns=exclude_columns,standardization='zscore', val_ratio=0.2, duration=12, shuffle=shuffle)
        model = lstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer).to(device)
        # 训练模型
        if args.train == 'domain_wise':
            train_losses,test_losses= lstm_domain_wise_training(model=model,train_sets=train_sets, val_sets=val_sets, test_sets=test_sets, device=device, batch_size=batch_size, epoch=args.epochs, lr=args.lr)
        elif args.train == 'normal':
            model = lstm_train(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, device=device, batch_size=batch_size, epoch=args.epochs, lr=args.lr)
        # 绘制预测图
        save_dir = f'result/graphs/{args.model}'
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if args.train == 'domain_wise':
            domain_pred_plot(model=model,train_sets=train_sets, val_sets=val_sets,test_sets=test_sets, time_window_step=time_window_step,model_name='lstm',device=device)
            domain_wise_test(model=model, test_sets=test_sets, model_name='lstm', device=device)
        elif args.train == 'normal':
            y_train_pred=model(torch.tensor(X_train, dtype=torch.float32).to(device)).cpu().detach().numpy()
            pred_plot(y_train_pred,y_train,time_window_step=12,graph_title='lstm_train')
            y_val_pred=model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().detach().numpy()
            pred_plot(y_val_pred,y_val,time_window_step=12,graph_title='lstm_val')
            pred,true=lstm_test(model,X_test,y_test,device=device)
            pred_plot(pred,true,time_window_step=12,graph_title='lstm_test')
            print('NMSE on test set:', nmse_loss(pred, true).item())