import numpy as np
from utils.feature_engineering import Xy_time_window_process,load_processed_ds,feature_extraction
from torch.utils.data import DataLoader
import pandas as pd

#这个是用于标准化的
def compute_global_stats(processed_ds, exception=[]):
    """
    计算全局统计量（均值和标准差或最大值和最小值）
    """
    all_data = []
    for city, floor_dict in processed_ds.items():
        for L, df_list in floor_dict.items():
            for df in df_list:
                if df is None or df.empty:
                    continue
                X_df, _ = feature_extraction(df,exclude_columns=exception)
                # 删掉异常的区域
                if (X_df[['energy_mean', 'energy_std', 'energy_q1', 'energy_q3']].drop_duplicates() <= 0.0001).values.any():
                    continue
                else:
                    all_data.append(X_df)
    global_data = pd.concat(all_data, axis=0)
    mean = global_data.mean()
    std = global_data.std()
    min_val = global_data.min()
    max_val = global_data.max()
    return mean, std, min_val, max_val

def X_standardization(X_df, mean=None, std=None, min_val=None, max_val=None, exception=[], standardization='zscore'):
    """
    对数据进行标准化，支持 zscore 和 minmax 两种方式
    """
    X_scaled = X_df.copy()
    for col in X_scaled.columns:
        if col not in exception:
            if standardization == 'zscore':
                X_scaled[col] = (X_scaled[col] - mean[col]) / std[col]
            elif standardization == 'minmax':
                X_scaled[col] = (X_scaled[col] - min_val[col]) / (max_val[col] - min_val[col])
    return X_scaled

def train_val_test_split(processed_ds, train_dic: dict, val_dic: dict,test_dic: dict, standardization, exclude_columns,duration=12):
    """
    将数据划分为训练集和测试集，并进行标准化和时间窗口处理
    """
    X_train_all, y_train_all = [], []
    X_val_all, y_val_all = [], []
    X_test_all, y_test_all = [], []
    
    # 计算全局统计量
    mean, std, min_val, max_val = compute_global_stats(processed_ds,exception=exclude_columns)
    for city, floor_dict in processed_ds.items():
        for L, df_list in floor_dict.items():
            if not df_list:
                continue
            for df in df_list:
                if df is None or df.empty:
                    continue

                # 是否属于 train/test
                is_train = city in train_dic and L in train_dic[city]
                is_val = city in val_dic and L in val_dic[city]
                is_test = city in test_dic and L in test_dic[city]

                if not is_train and not is_test and not is_val:
                    continue  # 忽略未指定的部分

                # 1. 提取特征和标签
                X_df, y_raw = feature_extraction(df,exclude_columns=exclude_columns)

                # 2. 标准化
                X_scaled = X_standardization(
                    X_df, mean=mean, std=std, min_val=min_val, max_val=max_val, standardization=standardization
                )

                # 3. 时间窗口处理
                X_windowed, y_windowed = Xy_time_window_process(X_scaled, y_raw, duration)

                if is_train:
                    X_train_all.append(X_windowed)
                    y_train_all.append(y_windowed)
                elif is_val:
                    X_val_all.append(X_windowed)
                    y_val_all.append(y_windowed)
                elif is_test:
                    X_test_all.append(X_windowed)
                    y_test_all.append(y_windowed)

    X_train = np.concatenate(X_train_all, axis=0) if X_train_all else np.array([])
    y_train = np.concatenate(y_train_all, axis=0) if y_train_all else np.array([])
    X_val = np.concatenate(X_val_all, axis=0) if X_val_all else np.array([])
    y_val = np.concatenate(y_val_all, axis=0) if y_val_all else np.array([])
    X_test = np.concatenate(X_test_all, axis=0) if X_test_all else np.array([])
    y_test = np.concatenate(y_test_all, axis=0) if y_test_all else np.array([])

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_val_k_test_split(processed_ds, train_dic: dict, test_dic: dict, standardization, exclude_columns,shuffle,val_ratio= 0.2,duration=12):
    """
    将数据划分为训练集、验证集和测试集。
    验证集是从训练集中按比例(val_ratio)随机抽取的。
    """
    from sklearn.model_selection import train_test_split

    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    
    # 1. 计算全局统计量
    mean, std, min_val, max_val = compute_global_stats(processed_ds,exception=exclude_columns)

    # 2. 遍历数据，分离出训练集和测试集
    for city, floor_dict in processed_ds.items():
        for L, df_list in floor_dict.items():
            if not df_list:
                continue
            for df in df_list:
                if df is None or df.empty:
                    continue

                is_train = city in train_dic and L in train_dic[city]
                is_test = city in test_dic and L in test_dic[city]

                if not is_train and not is_test:
                    continue

                # 特征提取
                X_df, y_raw = feature_extraction(df,exclude_columns=exclude_columns)

                # 标准化
                X_scaled = X_standardization(
                    X_df, mean=mean, std=std, min_val=min_val, max_val=max_val, standardization=standardization
                )

                # 时间窗口处理
                X_windowed, y_windowed = Xy_time_window_process(X_scaled, y_raw, duration)

                if is_train:
                    X_train_all.append(X_windowed)
                    y_train_all.append(y_windowed)
                elif is_test:
                    X_test_all.append(X_windowed)
                    y_test_all.append(y_windowed)

    # 3. 拼接所有训练和测试数据
    X_train_full = np.concatenate(X_train_all, axis=0) if X_train_all else np.array([])
    y_train_full = np.concatenate(y_train_all, axis=0) if y_train_all else np.array([])
    X_test = np.concatenate(X_test_all, axis=0) if X_test_all else np.array([])
    y_test = np.concatenate(y_test_all, axis=0) if y_test_all else np.array([])

    # 4. 从完整训练集中按比例划分出验证集
    if X_train_full.size > 0 and val_ratio > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_ratio,shuffle=shuffle)
    else:
        X_train, y_train = X_train_full, y_train_full
        X_val, y_val = np.array([]), np.array([])

    return X_train, y_train, X_val, y_val, X_test, y_test

def train_val_test_split_by_domain(processed_ds, train_dic: dict, val_dic: dict, test_dic: dict, standardization, exclude_columns,duration=12):
    """
    将每个 (city, L) 作为一个独立的数据集，分别放入训练/测试集合
    返回：
        train_sets: dict[(city, L)] = {'X': ..., 'y': ...}
        test_sets: dict[(city, L)] = {'X': ..., 'y': ...}
    """
    train_sets = {}
    val_sets = {}
    test_sets = {}

    # 计算全局统计量
    mean, std, min_val, max_val = compute_global_stats(processed_ds,exception=exclude_columns)

    for city, floor_dict in processed_ds.items():
        for L, df_list in floor_dict.items():
            if not df_list:
                continue

            # 是否属于 train/test
            is_train = city in train_dic and L in train_dic[city]
            is_val = city in val_dic and L in val_dic[city]
            is_test = city in test_dic and L in test_dic[city]

            if not is_train and not is_test and not is_val:
                continue  # 忽略未指定的部分

            X_list, y_list = [], []

            for df in df_list:
                if df is None or df.empty:
                    continue

                # 1. 特征提取
                X_df, y_raw = feature_extraction(df,exclude_columns=exclude_columns)

                # 2. 标准化
                X_scaled = X_standardization(
                    X_df, mean=mean, std=std, min_val=min_val, max_val=max_val,standardization=standardization
                )

                # 3. 窗口处理
                X_windowed, y_windowed = Xy_time_window_process(X_scaled, y_raw, duration)

                X_list.append(X_windowed)
                y_list.append(y_windowed)

            # 拼接该楼所有数据
            if X_list and y_list:
                X_all = np.concatenate(X_list, axis=0)
                y_all = np.concatenate(y_list, axis=0)

                if is_train:
                    train_sets[(city, L)] = {'X': X_all, 'y': y_all}
                elif is_val:
                    val_sets[(city, L)] = {'X': X_all, 'y': y_all}
                elif is_test:
                    test_sets[(city, L)] = {'X': X_all, 'y': y_all}

    return train_sets, val_sets, test_sets
def train_val_k_test_split_by_domain(processed_ds, train_dic: dict, test_dic: dict, val_ratio: float, standardization, exclude_columns, shuffle,duration=12):
    """
    按域划分数据集，并从每个训练域中按比例抽取验证集。
    返回：
        train_sets: dict[(city, L)] = {'X': ..., 'y': ...}
        val_sets: dict[(city, L)] = {'X': ..., 'y': ...}
        test_sets: dict[(city, L)] = {'X': ..., 'y': ...}
    """
    from sklearn.model_selection import train_test_split
    train_sets_full = {}
    test_sets = {}

    # 1. 计算全局统计量
    mean, std, min_val, max_val = compute_global_stats(processed_ds,exception=exclude_columns)

    # 2. 遍历数据，分离出完整的训练域和测试域
    for city, floor_dict in processed_ds.items():
        for L, df_list in floor_dict.items():
            if not df_list:
                continue

            is_train = city in train_dic and L in train_dic[city]
            is_test = city in test_dic and L in test_dic[city]

            if not is_train and not is_test:
                continue

            X_list, y_list = [], []
            for df in df_list:
                if df is None or df.empty:
                    continue

                X_df, y_raw = feature_extraction(df,exclude_columns=exclude_columns)
                X_scaled = X_standardization(X_df, mean=mean, std=std, min_val=min_val, max_val=max_val,standardization=standardization)
                X_windowed, y_windowed = Xy_time_window_process(X_scaled, y_raw, duration)
                X_list.append(X_windowed)
                y_list.append(y_windowed)

            if X_list and y_list:
                X_all = np.concatenate(X_list, axis=0)
                y_all = np.concatenate(y_list, axis=0)
                domain_key = (city, L)
                if is_train:
                    train_sets_full[domain_key] = {'X': X_all, 'y': y_all}
                elif is_test:
                    test_sets[domain_key] = {'X': X_all, 'y': y_all}

    # 3. 从每个训练域中划分出验证集
    train_sets = {}
    val_sets = {}
    if val_ratio > 0:
        for domain_key, data in train_sets_full.items():
            X_train_full, y_train_full = data['X'], data['y']
            if len(X_train_full) > 1:
                X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_ratio,shuffle=shuffle)
                train_sets[domain_key] = {'X': X_train, 'y': y_train}
                val_sets[domain_key] = {'X': X_val, 'y': y_val}
            else: # 如果样本太少，则不划分验证集
                train_sets[domain_key] = {'X': X_train_full, 'y': y_train_full}
    else:
        train_sets = train_sets_full

    return train_sets, val_sets, test_sets

from torch.utils.data import Dataset
import torch
class MixedDomainDataset(Dataset):
    def __init__(self, domain_dict, domain2id):
        self.samples = []
        for dk, data in domain_dict.items():
            xid = domain2id[dk]          # <<< 新增
            X, y = data['X'], data['y']
            for i in range(len(X)):
                self.samples.append((X[i], y[i], xid)) 
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y, domain_key = self.samples[idx]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), domain_key
def collate_fn(batch):
    Xs, ys, domain_ids = zip(*batch)
    ys = [y if y.ndim > 0 else y.unsqueeze(0) for y in ys]
    return (
        torch.stack(Xs),
        torch.stack(ys),
        torch.tensor(domain_ids)
    )


if __name__ == '__main__':
    from utils.seed import set_seed
    set_seed(42)
    dataset=load_processed_ds(folder='result/processed_ds')
    # X_train, y_train, X_val, y_val, X_test, y_test = train_test_split(dataset,train_dic={'Beijing': [], 'Hefei': [9,10],'Jiling':[]},val_dic={'Beijing': [22], 'Hefei': [],'Jiling':[]}, test_dic={'Beijing': [23], 'Hefei': [],'Jiling':[]},standardization='minmax',duration=12)
    # print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    train_sets, val_sets, test_sets = train_val_test_split_by_domain(dataset,train_dic={'Beijing': [], 'Hefei': [9],'Jiling':[]},val_dic={'Beijing': [22], 'Hefei': [],'Jiling':[]}, test_dic={'Beijing': [22], 'Hefei': [],'Jiling':[]},standardization='zscore',duration=12)
    train_domain2id = {dk: i for i, dk in enumerate(train_sets.keys())}
    test_domain2id = {dk: i for i, dk in enumerate(test_sets.keys())}
    train_dataset = MixedDomainDataset(train_sets,train_domain2id)