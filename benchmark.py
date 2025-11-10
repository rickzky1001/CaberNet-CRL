from utils.data_read import data_read
from utils.seed import set_seed
from model.lstm import lstm_baseline,lstm_train
import torch
if __name__=='__main__':
    seed=43
    set_seed(seed)
    device='cuda:1' if torch.cuda.is_available() else 'cpu'
    hidden_size=64
    batch_size=510

   #选取训练测试集合，5个用于训练，1个用于测试 
   #所有集合：'Beijing': [22,23], 'Hefei': [9,10], 'Huizhou': [15,16],'Jiling':[] #Jiling不参与所有任务
    train_dic={'Beijing': [23], 'Hefei': [9,10], 'Huizhou': [15,16],'Jiling':[]}
    test_dic={'Beijing': [22], 'Hefei': [], 'Huizhou': [],'Jiling':[]}
    # val是train的20%
    X_train, y_train, X_val, y_val,X_test, y_test = data_read(train_dic,test_dic)
    model = lstm_baseline(input_size=X_train.shape[1], hidden_size=64, num_layers=1).to(device)
    # 训练模型
    epoch=20 
    lr= 5e-4
    model = lstm_train(model=model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, device=device, batch_size=batch_size, epoch=epoch, lr=lr)