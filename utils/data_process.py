import numpy as np
import pandas as pd
import sys
#Process 1:构建dataset字典
# 1.读数据函数
def data_read(city,floor,raw_data):
    suffix = '_raw_data.csv' if raw_data else '_20241101_20250707.csv'
    if city == 'Huizhou':
        # 惠州数据的文件名格式不同
        raw_data=pd.read_csv(f'Data_export/{city}/Sensor/{city}_{floor}L/{city}_{floor}L{suffix}')
        if 'index' in raw_data.columns: 
            raw_data.rename(columns={'index':'time'},inplace=True)
        description=pd.read_csv(rf'Data_export/{city}/Sensor/{city}_{floor}L/Description.csv')
    else:
        raw_data=pd.read_csv(f'Data_export/{city}/{city}_{floor}L/{city}_{floor}L{suffix}')
        if 'index' in raw_data.columns: 
            raw_data.rename(columns={'index':'time'},inplace=True)
        description=pd.read_csv(rf'Data_export/{city}/{city}_{floor}L/Description.csv')
    return raw_data,description
# 2.增加星期和工作状态
def add_weekday(raw_data, holiday=None, workday=None):
    data = raw_data.copy()
    data['time'] = pd.to_datetime(data['time'])
    # weekday: 0=Monday, ..., 6=Sunday
    data['weekday'] = data['time'].dt.dayofweek

    # 基础工作时间条件：9:00 <= hour < 18:00
    hour = data['time'].dt.hour
    is_work_hour = (hour >= 9) & (hour < 18)

    # 默认工作条件：周一到周五 且在工作时间
    is_weekday = data['weekday'] < 5
    is_work = is_weekday & is_work_hour

    # 处理 holiday（强制这天 9:00–18:00 为非工作时间）
    if holiday is not None:
        holiday = pd.to_datetime(holiday)
        is_holiday_time = data['time'].dt.normalize().isin(holiday) & is_work_hour
        is_work[is_holiday_time] = False

    # 处理 workday（强制这天 9:00–18:00 为工作时间）
    if workday is not None:
        workday = pd.to_datetime(workday)
        is_force_work_time = data['time'].dt.normalize().isin(workday) & is_work_hour
        is_work[is_force_work_time] = True

    data['is_work'] = is_work.astype(int)
    return data
# 3.计算总能耗
def add_total_energy(df,description,energy_type):
    data=df.copy()
    if energy_type=='ac':
        y_feature_names=description[(description['feature_name'].str.contains('total_active_power_energy|meter_total_active_energy'))&(description['device_name'].str.contains('空调'))]['feature_name'].values
    elif energy_type=='total':
        y_feature_names=description[(description['feature_name'].str.contains('total_active_power_energy|meter_total_active_energy'))]['feature_name'].values
    data['total_energy']=data[y_feature_names].sum(axis=1,skipna=False)
    return data
# 4.整合构建dataset字典
def dataset(energy_type,use_raw_data):
    holiday=['2025-01-01','2025-01-28','2025-01-29','2025-01-30','2025-01-31','2025-02-01','2025-02-02','2025-02-03','2025-02-04','2025-04-04','2025-04-05','2025-04-06','2025-05-01','2025-05-02','2025-05-03','2025-05-04','2025-05-05','2025-05-31','2025-06-01','2025-06-02']
    workday=['2025-01-26','2025-01-08','2025-04-27']
    # b1,b1_d=data_read('Beijing',1,raw_data=use_raw_data)
    b22,b22_d=data_read('Beijing',22,raw_data=use_raw_data)
    b23,b23_d=data_read('Beijing',23,raw_data=use_raw_data)

    # hf1,hf1_d=data_read('Hefei',1,raw_data=use_raw_data)
    hf9,hf9_d=data_read('Hefei',9,raw_data=use_raw_data)
    hf10,hf10_d=data_read('Hefei',10,raw_data=use_raw_data)

    # hz1,hfz_d=data_read('Huizhou',1,raw_data=use_raw_data)
    hz15,hz15_d=data_read('Huizhou',15,raw_data=use_raw_data)
    hz16,hz16_d=data_read('Huizhou',16,raw_data=use_raw_data)

    # j1,j1_d=data_read('Jiling',1,raw_data=use_raw_data)
    j25,j25_d=data_read('Jiling',25,raw_data=use_raw_data)
    j26,j26_d=data_read('Jiling',26,raw_data=use_raw_data)
    ds = {
        'Beijing': {
            # 1: {'data': add_total_energy(add_weekday(b1,holiday,workday),b1_d,energy_type=energy_type), 'd': b1_d},
            22: {'data': add_total_energy(add_weekday(b22,holiday,workday),b22_d,energy_type=energy_type), 'd': b22_d},
            23: {'data': add_total_energy(add_weekday(b23,holiday,workday),b23_d,energy_type=energy_type), 'd': b23_d}
        },
        'Hefei': {
            # 1: {'data': add_total_energy(add_weekday(hf1,holiday,workday),hf1_d,energy_type=energy_type), 'd': hf1_d},
            9: {'data': add_total_energy(add_weekday(hf9,holiday,workday),hf9_d,energy_type=energy_type), 'd': hf9_d},
            10: {'data': add_total_energy(add_weekday(hf10,holiday,workday),hf10_d,energy_type=energy_type), 'd': hf10_d}
        },
        'Huizhou': {
            # 1: {'data': add_total_energy(add_weekday(hf1,holiday,workday),hf1_d,energy_type=energy_type), 'd': hf1_d},
            15: {'data': add_total_energy(add_weekday(hz15,holiday,workday),hz15_d,energy_type=energy_type), 'd': hz15_d},
            16: {'data': add_total_energy(add_weekday(hz16,holiday,workday),hz16_d,energy_type=energy_type), 'd': hz16_d}
        },
        'Jiling': {
            # 1: {'data': add_total_energy(add_weekday(j1,holiday,workday),j1_d,energy_type=energy_type), 'd': j1_d},
            25: {'data': add_total_energy(add_weekday(j25,holiday,workday),j25_d,energy_type=energy_type), 'd': j25_d},
            26: {'data': add_total_energy(add_weekday(j26,holiday,workday),j26_d,energy_type=energy_type), 'd': j26_d}
        }
    }
    return ds

#Process 2:数据处理
# 1.选择列
def column_selection(data,column_list):
    import re
    columns=list(data.columns[data.columns.str.contains('|'.join(map(re.escape, column_list)))])
    columns.insert(0,'time')
    return data[columns]
def light_lighting_convert(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    # 找出所有包含 light_intensity 的列
    target_cols = [col for col in df_copy.columns if 'light_lighting' in col]
    # 定义转换函数
    def lux_to_level(x):
        if pd.isna(x):
            return np.nan
        if x <= 5:
            return 0
        elif x <= 50:
            return 1
        elif x <= 100:
            return 2
        elif x <= 500:
            return 3
        elif x <= 2000:
            return 4
        else:
            return 5
    # 应用于每一列
    for col in target_cols:
        df_copy[col] = df_copy[col].apply(lux_to_level)

    return df_copy
# 2.小部分缺失填充
def linear_interpolation(
    df: pd.DataFrame,
    max_gap: int = 12,
    method: str = "linear",
    limit_direction: str = "both",
    **interp_kwargs,
) -> pd.DataFrame:
    out = df.copy()
    for col in out:
        s = out[col]
        if col=='time':
            continue
        elif not s.isna().any():
            continue 
        na = s.isna()
        grp_id = na.ne(na.shift()).cumsum()
        gap_size = na.groupby(grp_id).transform("sum")
        small_gap = na & (gap_size <= max_gap)
        interp = s.interpolate(
            method=method, limit_direction=limit_direction, **interp_kwargs
        )
        out[col] = s.where(~small_gap, interp)
    return out

# 3. 行填补
def row_interpolation(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    res = df.copy()
    for token in feature_list:
        cols = [c for c in res.columns if token in c]
        if not cols:
            continue  # nothing matches this token
        row_avg = res[cols].mean(axis=1, skipna=True)
        for c in cols:
            res[c] = res[c].fillna(row_avg)
    return res
# 4. 切分有效样本
#目的：摘取valid行超过max_valid_length行的区域
# 1). index1是断续的(断是因为中间的行删掉了)，index2是连续的,找到index1[i+1]-index1[i]>1的所有i(breakpoint_list)
# 2). 则0,breakpoint_list[0]间连续，breakpoint_list[0]+1,breakpoint_list[1]间连续...breakpoint_list[-1]+1,-1间连续
# 3). 要求连续行大于max_valid_length，则对每个区间需分别满足index2[breakpoint_list[0]]-0>max_valid_length和index2[breakpoint_list[2]]-index2[breakpoint_list[1]]>max_valid_length...和index2[-1]-index2[breakpoint_list[-1]]>max_valid_length
# 4). 共len(breakpoint_list+1)个区间
def valid_sample_split(df_filled):
    max_valid_length=96
    result=[]
    temp=df_filled.reset_index(names='index1').dropna().reset_index(drop=True).reset_index(names='index2')
    next_index=temp['index1'].shift(-1)
    current_index=temp['index1']
    is_breakpoint=(next_index-current_index)>1
    breakpoint_list=np.where(is_breakpoint)[0]
    if len(breakpoint_list)>0:
        point_list=np.insert(np.insert(breakpoint_list,0,0),len(breakpoint_list)+1,temp['index2'].values[-1])
        for i in range(len(point_list)-1):
            if point_list[i+1]-point_list[i]>max_valid_length:
                result.append(temp.drop(columns=['index2','index1']).iloc[point_list[i]+1:point_list[i+1]+1].reset_index(drop=True))
        return result
    else:
        return [temp.drop(columns=['index2','index1'])]
    
# 5.处理后的数据集构建
def process_all(ds, column_list, feature_list):
    processed_ds = {}
    for city, buildings in ds.items():
        processed_ds[city] = {}
        for L, content in buildings.items():
            data = content['data']
            df = column_selection(data, column_list)
            df= light_lighting_convert(df)
            df_filled = row_interpolation(
                linear_interpolation(df),
                feature_list=feature_list
            )
            result = valid_sample_split(df_filled)

            processed_ds[city][L] = result
    return processed_ds

# 6.统一维度
def aggregate_sensor_features(df: pd.DataFrame,aggregate_features_list: list, keep_list=['is_work','total_energy']) -> pd.DataFrame:
    agg_features = {}

    for keyword in aggregate_features_list:
        if type(keyword) == str:
            cols = [col for col in df.columns if keyword in col]
            name=keyword

        elif type(keyword) == list:
            cols = [col for col in df.columns if ((keyword[0] in col) or (keyword[1] in col))]
            if 'temperature' in keyword[0]:
                name='temperature'
            elif 'light' in keyword[0]:
                name='light'
            else:
                print('never seen this')
                sys.exit(0)
        agg_features[f'{name}_mean'] = df[cols].mean(axis=1)
        # agg_features[f'{name}_std'] = df[cols].std(axis=1)
        # agg_features[f'{name}_max'] = df[cols].max(axis=1)
        # agg_features[f'{name}}_min'] = df[cols].min(axis=1)

    agg_df = pd.DataFrame(agg_features)
    keep_df = df[keep_list].copy() if keep_list else pd.DataFrame()

    result_df = pd.concat([agg_df, keep_df], axis=1)
    return result_df
def aggregate_processed_dict(processed, aggregate_features_list):
    new_processed = {}

    for city, floor_dict in processed.items():
        new_processed[city] = {}
        for L, df_list in floor_dict.items():
            new_df_list = []
            for df in df_list:
                if df is not None and not df.empty:
                    agg_df = aggregate_sensor_features(df, aggregate_features_list)
                    new_df_list.append(agg_df)
                else:
                    new_df_list.append(df)  # 保留原始 None 或空 df
            new_processed[city][L] = new_df_list

    return new_processed
import os
import shutil
def save_processed_ds(processed_ds, folder='result/processed_ds'):
    """
    将 processed_ds 保存为多个 CSV 文件，结构为 folder/city/L/idx.csv。
    在保存之前清空目标文件夹中的所有文件。
    """
    # 清空目标文件夹
    if os.path.exists(folder):
        shutil.rmtree(folder)
    
    # 保存数据集
    for city, city_dict in processed_ds.items():
        for L, df_list in city_dict.items():
            subfolder = os.path.join(folder, str(city), str(L))
            os.makedirs(subfolder, exist_ok=True)

            for idx, df in enumerate(df_list):
                if df is not None and not df.empty:
                    filename = f"{idx}.csv"
                    filepath = os.path.join(subfolder, filename)
                    df.to_csv(filepath, index=False)

if __name__ == '__main__':
    #合肥1L的环境数据只有一个device
    #吉林1L的环境数据只有一个device
    #吉林26L的环境数据只有2个device

    column_list=set(['total_energy','weekday','is_work'])|set(['light_lighting','temp_temperature','environment_temperature','environment_humidity','light_intensity_level','environment_co2','environment_pressure','environment_tvoc','air_temperature'])

    ds = dataset(energy_type='ac',use_raw_data=True)
    
    feature_list=['temperature', 'humidity', 'light_intensity', 'lighting', 'co2', 'tvoc', 'pressure']
    processed = process_all(ds, column_list,feature_list)
    aggregate_feature_list=[['temp_temperature','evironment_temperature'], 'humidity', ['light_lighting','light_intensity_level'], 'co2', 'tvoc', 'pressure','air_temperature'] #最后一个为天气特征
    processed=aggregate_processed_dict(processed, aggregate_feature_list)

    save_processed_ds(processed)