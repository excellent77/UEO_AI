import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Literal

import torch
from torch.utils.data import DataLoader, TensorDataset


FILL_STRATEGIES = ('mean', 'median', 'most_frequent', 'constant')
SCALE_METHODS = ('minmax', 'standard')


def clean_data(
        df: pd.DataFrame,
        datetime_col: str
) -> pd.DataFrame:
    """
    清洗資料，將指定的時間欄位轉換為 datetime 格式，並依此欄位對 DataFrame 進行排序。

    Args:
        df (pd.DataFrame): 原始 DataFrame。
        datetime_col (str): 時間戳記欄位的名稱。

    Returns:
        pd.DataFrame: 經過清洗和排序後的 DataFrame。
    """
    if datetime_col is None:
        return df

    df[datetime_col] = pd.to_datetime(df[datetime_col], format='mixed')
    df = df.sort_values(by=datetime_col)
    df = df.reset_index(drop=True)
    return df


def remove_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """
    使用四分位距 (IQR) 方法來識別並替換 DataFrame 中數值欄位的離群值。
    離群值將被該欄位的中位數所取代。

    Args:
        df (pd.DataFrame): 輸入的 DataFrame。
        factor (float): 用於定義離群值邊界的因子，預設為 1.5。

    Returns:
        pd.DataFrame: 已替換離群值的 DataFrame。
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

        outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        df_clean.loc[outliers, col] = median

    return df_clean


def fill_missing(
        df: pd.DataFrame,
        strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'mean',
        *args,
        **kwargs
) -> pd.DataFrame:
    """
    對 DataFrame 中的數值型欄位進行缺失值填補。
    此函式會先將無窮大值替換為 NaN，然後使用指定策略進行填補。

    Args:
        df (pd.DataFrame): 原始 DataFrame。
        strategy (Literal['mean', 'median', 'most_frequent', 'constant']): 填補策略。
        *args, **kwargs: 傳遞給 `SimpleImputer` 的其他參數。

    Returns:
        pd.DataFrame: 已填補缺失值的 DataFrame。
    """
    numeric_cols = df.select_dtypes(include=['number']).columns

    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    imputer = SimpleImputer(strategy=strategy, *args, **kwargs)
    df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)

    df_others = df.drop(columns=numeric_cols)

    df_imputed = pd.concat([df_others, df_numeric], axis=1)

    df_imputed = df_imputed[df.columns]

    return df_imputed


def scale_features(
        df: pd.DataFrame,
        method: Literal['minmax', 'standard'] = 'minmax',
        scaler: object = None
) -> tuple:
    """
    對 DataFrame 中的數值型欄位進行特徵縮放。

    Args:
        df (pd.DataFrame): 原始 DataFrame。
        method (Literal['minmax', 'standard']): 縮放方法，可選 'minmax' 或 'standard'。
        scaler (object, optional): 若提供，則使用此已存在的縮放器進行轉換；否則，將重新擬合一個新的縮放器。

    Returns:
        tuple: 一個包含以下兩個元素的元組：
            - pd.DataFrame: 經過縮放處理的 DataFrame。
            - object: 用於縮放的縮放器物件。
    """
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
    df_others = df.drop(columns=numeric_cols)
    df_scaled = pd.concat([df_others, df_scaled_numeric], axis=1)
    df_scaled = df_scaled[df.columns]
    return df_scaled, scaler


def preprocess_for_lstm(
        df: pd.DataFrame,
        datetime_col: str,
        feature_cols: list,
        target_cols: list,
        fill_strategy: Literal['mean', 'median', 'most_frequent', 'constant'] = 'mean',
        scale_method: Literal['minmax', 'standard'] = 'minmax',
        sequence_length: int = 24,
        apply_scaler: dict = {},
        **kwargs
) -> tuple:
    """
    執行完整的資料預處理流程，將原始 DataFrame 轉換為適用於時序模型（如 LSTM）的序列資料。

    Args:
        df (pd.DataFrame): 原始 DataFrame 或 Gradio 的 File 物件。
        datetime_col (str): 時間戳記欄位的名稱。
        feature_cols (list): 作為模型輸入特徵的欄位列表。
        target_cols (list): 作為模型預測目標的欄位列表。
        fill_strategy (Literal): 缺失值填補策略。
        scale_method (Literal): 特徵縮放方法。
        sequence_length (int): 每個輸入序列的時間步長。
        apply_scaler (dict, optional): 若提供，則使用已存在的縮放器（例如 {'feature': scaler_x, 'target': scaler_y}）。
        **kwargs: 傳遞給 `fill_missing` 的其他參數。

    Returns:
        tuple: 一個包含以下四個元素的元組：
            - np.ndarray: 處理後的特徵序列 (X)。
            - np.ndarray: 處理後的目標序列 (y)。
            - object: 用於特徵的縮放器。
            - object: 用於目標的縮放器。
    """
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.read_csv(df.name)
        except AttributeError:
            raise ValueError("Input must be a pandas DataFrame or a file-like object.")

    df = clean_data(df, datetime_col)
    df = remove_outliers_iqr(df, factor=150)
    df = fill_missing(df, strategy=fill_strategy, **kwargs)

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
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True
) -> DataLoader:
    """
    將 NumPy 格式的特徵和標籤陣列轉換為 PyTorch 的 DataLoader。

    Args:
        X (np.ndarray): 特徵數據。
        y (np.ndarray): 標籤數據。
        batch_size (int): 每個批次的樣本數。
        shuffle (bool): 是否在每個 epoch 開始時打亂數據。

    Returns:
        DataLoader: PyTorch 的數據加載器物件。
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
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
    print("Scaler:", scalers)