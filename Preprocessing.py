import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Literal
import torch
from torch.utils.data import DataLoader, TensorDataset


# Constants for preprocessing
FILL_STRATEGIES = ('mean', 'median', 'most_frequent', 'constant')
SCALE_METHODS = ('minmax', 'standard')



def clean_data(
        df:pd.DataFrame, # 原始資料
        datetime_col:str # 時間欄位名稱
    ) -> pd.DataFrame:
    '''
    清洗資料，將時間欄位轉換為 datetime 格式並排序。
    傳回:
        - 清洗後的 DataFrame
    '''
    df[datetime_col] = pd.to_datetime(df[datetime_col], format='mixed')
    df = df.sort_values(by=datetime_col)
    df = df.reset_index(drop=True)
    return df


def remove_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    利用IQR方法替換DataFrame中數值欄位的離群值。
    離群值定義:
        小於 Q1 - factor * IQR 或大於 Q3 + factor * IQR
    用欄位中位數替代離群值。
    
    參數:
        df: 輸入的DataFrame
        factor: 控制離群值範圍的因子，預設1.5
    
    回傳:
        替換離群值後的DataFrame
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        median = df_clean[col].median()

        # 找出離群值的位置
        outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        # 用中位數替換離群值
        df_clean.loc[outliers, col] = median

    return df_clean


def fill_missing(
        df: pd.DataFrame,  # 原始資料
        strategy: Literal['mean', 'median', 'most_frequent', 'constant']='mean'  # 缺失值填補策略
    ) -> pd.DataFrame:
    '''
    填補缺失值，僅對數值型欄位進行填補。
    傳回:
        - 填補後的 DataFrame
    '''
    # 找出數值型欄位
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # 將 inf 和 -inf 替換為 NaN，避免 imputer 出錯
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 只對數值型欄位做補值
    imputer = SimpleImputer(strategy=strategy)
    df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
    
    # 其他欄位（如時間）直接保留
    df_others = df.drop(columns=numeric_cols)
    
    # 合併
    df_imputed = pd.concat([df_others, df_numeric], axis=1)
    
    # 保持原本欄位順序
    df_imputed = df_imputed[df.columns]
    
    return df_imputed


def scale_features(
        df:pd.DataFrame, # 原始資料
        method:Literal['minmax', 'standard']='minmax', # 特徵正規化方式，可選 'minmax', 'standard'
        scaler:object=None # 已存在的標準化器物件，若有則使用
        )-> tuple:
    '''
    對數值型欄位進行特徵縮放。
    傳回:
        - 縮放後的 DataFrame
        - 標準化器物件
    '''
    # 只選擇數值型欄位
    numeric_cols = df.select_dtypes(include=['number']).columns
    if scaler is None:
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError("method must be 'minmax' or 'standard'")
        
        scaled_numeric = scaler.fit_transform(df[numeric_cols])
    else:
        scaled_numeric = scaler.transform(df[numeric_cols])

    
    df_scaled_numeric = pd.DataFrame(scaled_numeric, columns=numeric_cols, index=df.index)
    # 其他欄位（如時間）保留
    df_others = df.drop(columns=numeric_cols)
    # 合併，並保持原欄位順序
    df_scaled = pd.concat([df_others, df_scaled_numeric], axis=1)
    df_scaled = df_scaled[df.columns]
    return df_scaled, scaler


def preprocess_for_lstm(
        df:pd.DataFrame, # 原始資料
        datetime_col:str, # 時間欄位名稱
        feature_cols:list, # 特徵(輸入)欄位清單
        target_cols:list, # 標籤(預測目標)欄位清單
        fill_strategy:Literal['mean', 'median', 'most_frequent', 'constant']='mean', # 缺失值填補策略
        scale_method:Literal['minmax', 'standard']='minmax', # 特徵正規化方式
        sequence_length:int=24, # LSTM 序列長度
        apply_scaler:dict={}
    )->tuple:
    '''
    對資料進行預處理，生成 LSTM 所需的特徵和標籤。
    傳回:
        - X: 特徵數組
        - y: 標籤數組
        - scaler: 標準化器物件
    '''
    # 若 df 不是 DataFrame，嘗試轉換
    if not isinstance(df, pd.DataFrame):
        try:
            # 若是 Gradio File 物件
            df = pd.read_csv(df.name)
        except AttributeError:
            raise ValueError("Input must be a pandas DataFrame or a file-like object.")

    df = clean_data(df, datetime_col)
    df = remove_outliers_iqr(df, factor=1.5)
    df = fill_missing(df, strategy=fill_strategy)

    feature_data = df[feature_cols]
    target_data = df[target_cols]

    feature_data, feature_scaler = scale_features(feature_data, method=scale_method, scaler=apply_scaler.get('feature', None))
    target_data, target_scaler = scale_features(target_data, method=scale_method, scaler=apply_scaler.get('target', None))
        
    feature_data = feature_data.values
    target_data = target_data.values

    X, y = [], []
    for i in range(len(feature_data) - sequence_length):
        X_seq = np.array(feature_data[i:i+sequence_length], dtype=np.float32)
        y_seq = np.array(target_data[i+sequence_length], dtype=np.float32)
        X.append(X_seq)
        y.append(y_seq)

    feature = np.array(X, dtype=np.float32)
    target = np.array(y, dtype=np.float32)

    assert not np.any(np.isnan(feature)), "Feature  contains NaN!"
    assert not np.any(np.isinf(feature)), "Feature  contains Inf!"
    assert not np.any(np.isnan(target)), "Target  contains NaN!"
    assert not np.any(np.isinf(target)), "Target  contains Inf!"

    return feature, target, feature_scaler, target_scaler


def process_to_dataloader(
        X:np.ndarray, # 特徵數據
        y:np.ndarray,
        batch_size:int=32,
        shuffle:bool=True
    )->DataLoader:
    """
    將特徵和標籤轉換為 PyTorch DataLoader 格式。
    傳回:
        - DataLoader 物件
    """

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



if __name__ == "__main__":
    # Example usage
    filepath = r'/mnt/c/UEO_AI/data/20240801.csv'
    feature_cols = ['Chiller_1_VLN_R', 'Chiller_1_VLN_S']
    target_cols = ['Chiller_1_VLN_avg', 'Chiller_1_I_S']
    df = pd.read_csv(filepath)
    X, y, *scalers = preprocess_for_lstm(df,
                                        datetime_col='DateTime', 
                                       feature_cols=feature_cols,
                                       target_cols=target_cols,
                                       fill_strategy='mean', 
                                       scale_method='minmax', 
                                       sequence_length=24)
    print("X shape:", X.shape, "X sample:", X[0])
    print("y shape:", y.shape, "y sample:", y[0])
    print("Scaler:", scalers)  # To save or use later for inverse transformation