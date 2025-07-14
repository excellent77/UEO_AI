import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer



FILL_STRATEGIES = ('mean', 'median', 'most_frequent', 'constant')
SCALE_METHODS = ('minmax', 'standard')



def clean_data(df:pd.DataFrame, datetime_col:str) -> pd.DataFrame:
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=datetime_col)
    df = df.reset_index(drop=True)
    return df


def fill_missing(df:pd.DataFrame , strategy:str='mean'): #strategy 可選 'mean', 'median', 'most_frequent', 'constant'
    # 找出數值型欄位
    numeric_cols = df.select_dtypes(include=['number']).columns
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


def scale_features(df, method='minmax'):
    # 只選擇數值型欄位
    numeric_cols = df.select_dtypes(include=['number']).columns
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("method must be 'minmax' or 'standard'")
    # 只對數值欄位做縮放
    scaled_numeric = scaler.fit_transform(df[numeric_cols])
    df_scaled_numeric = pd.DataFrame(scaled_numeric, columns=numeric_cols, index=df.index)
    # 其他欄位（如時間）保留
    df_others = df.drop(columns=numeric_cols)
    # 合併，並保持原欄位順序
    df_scaled = pd.concat([df_others, df_scaled_numeric], axis=1)
    df_scaled = df_scaled[df.columns]
    return df_scaled, scaler


def preprocess_for_lstm(df:pd.DataFrame, 
                        datetime_col:str, 
                        feature_cols:list,
                        target_cols:list,
                        fill_strategy:str='mean', 
                        scale_method:str='minmax', 
                        sequence_length:int=24)->tuple:
    
    # 若 df 不是 DataFrame，嘗試轉換
    if not isinstance(df, pd.DataFrame):
        try:
            # 若是 Gradio File 物件
            df = pd.read_csv(df.name)
        except AttributeError:
            raise ValueError("Input must be a pandas DataFrame or a file-like object.")

    df = clean_data(df, datetime_col)
    df = fill_missing(df, strategy=fill_strategy)
    df, scaler = scale_features(df, method=scale_method)

    X, y = [], []
    data = df.values
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length, [df.columns.get_loc(i) for i in feature_cols]])
        y.append(data[i+sequence_length, [df.columns.get_loc(i) for i in target_cols]])
    X = np.array(X)
    y = np.array(y)

    return X, y, scaler



if __name__ == "__main__":
    # Example usage
    filepath = r'/mnt/c/UEO_AI/data/20240801.csv'
    feature_cols = ['Chiller_1_VLN_R', 'Chiller_1_VLN_S']
    target_cols = ['Chiller_1_VLN_avg', 'Chiller_1_I_S']
    df = pd.read_csv(filepath)
    X, y, scaler = preprocess_for_lstm(df,
                                        datetime_col='DateTime', 
                                       feature_cols=feature_cols,
                                       target_cols=target_cols,
                                       fill_strategy='mean', 
                                       scale_method='minmax', 
                                       sequence_length=24)
    print("X shape:", X.shape, "X sample:", X[0])
    print("y shape:", y.shape, "y sample:", y[0])
    print("Scaler:", scaler)  # To save or use later for inverse transformation