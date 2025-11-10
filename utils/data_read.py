from utils.dataset_split import train_val_k_test_split
from utils.feature_engineering import load_processed_ds
def data_read(train_dic,test_dic):
    val_dic={'Beijing': [], 'Hefei': [], 'Huizhou': [],'Jiling':[]}
    combined_train_dic = {city: train_dic.get(city, []) + val_dic.get(city, []) for city in set(train_dic) | set(val_dic)}
    dataset=load_processed_ds(folder='result/processed_ds')
    X_train, y_train, X_val, y_val,X_test, y_test = train_val_k_test_split(dataset, train_dic=combined_train_dic, test_dic=test_dic, exclude_columns=[],standardization='zscore', val_ratio=0.2, duration=12, shuffle=False)
    return X_train, y_train, X_val, y_val,X_test, y_test