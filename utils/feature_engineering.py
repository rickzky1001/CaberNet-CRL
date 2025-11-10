import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import os
import pandas as pd
from collections import defaultdict

def load_processed_ds(folder='result/processed_ds'):
    """
    从指定文件夹读取所有 city/L/idx.csv 文件，还原为原始的 processed_ds 结构。

    返回:
        processed_ds: dict[city][L] = list of DataFrames
    """
    processed_ds = defaultdict(lambda: defaultdict(list))

    for city in os.listdir(folder):
        city_path = os.path.join(folder, city)
        if not os.path.isdir(city_path):
            continue

        for L in os.listdir(city_path):
            L_path = os.path.join(city_path, L)
            if not os.path.isdir(L_path):
                continue

            df_list = []
            for filename in sorted(os.listdir(L_path), key=lambda x: int(os.path.splitext(x)[0])):
                if filename.endswith('.csv'):
                    file_path = os.path.join(L_path, filename)
                    df = pd.read_csv(file_path)
                    df_list.append(df)

            processed_ds[city][int(L)] = df_list

    return dict(processed_ds)

def feature_extraction(df,exclude_columns):
    #exclude_columns是个列表
    X_df = df.copy().drop(columns=exclude_columns)
    X_df['y']=df['total_energy'] - df['total_energy'].shift(1)  # 用差分预测增量
    X_df=X_df.drop(columns=['total_energy'])
    X_df.dropna(inplace=True)
    y=X_df['y']
    
    # 计算均值、标准差和四分位数
    energy_mean = np.mean(y)
    energy_std = np.std(y)
    energy_q1 = np.percentile(y, 25)  # 下四分位数 (Q1)
    energy_q3 = np.percentile(y, 75)  # 上四分位数 (Q3)

    # 将统计量作为新特征添加到 X_df
    X_df['energy_mean'] = energy_mean
    X_df['energy_std'] = energy_std
    X_df['energy_q1'] = energy_q1
    X_df['energy_q3'] = energy_q3
    return X_df.drop(columns=['y']), X_df['y']

def Xy_time_window_process(X,y,duration=12):
    X_list=[]
    y_list=[]
    X_values = X.values
    y_values = y.values
    for t in range(duration,len(X_values)-duration):
        y_list.append(y_values[t])
        X_list.append(X_values[t-duration:t])
    return np.array(X_list),np.array(y_list)

