import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class Golden_Sample:
    """
    一個參考歷史數據的專家系統 (專家樣本)。
    它會根據當前的環境狀態，從歷史數據庫中尋找最相似的狀態，
    並回傳當時所採取的動作。這可以作為一個與 RL Agent比較的基準。
    """
    def __init__(self, df: pd.DataFrame, state_cols: list, action_cols: list):
        """
        初始化 Golden_Sample。

        Args:
            df (pd.DataFrame): 包含歷史數據的 DataFrame (例如 train.csv 的內容)。
            state_cols (list): DataFrame 中代表「狀態(State)」的欄位名稱列表。
            action_cols (list): DataFrame 中代表「動作(Action)」的欄位名稱列表。
        """
        # 驗證傳入的欄位是否存在於 DataFrame 中
        missing_state_cols = [col for col in state_cols if col not in df.columns]
        if missing_state_cols:
            raise ValueError(f"State columns not found in DataFrame: {missing_state_cols}")

        missing_action_cols = [col for col in action_cols if col not in df.columns]
        if missing_action_cols:
            raise ValueError(f"Action columns not found in DataFrame: {missing_action_cols}")

        # 我們只儲存需要的欄位，並轉換為 NumPy array 以提高搜尋效能
        self.historical_states = df[state_cols].values
        self.historical_actions = df[action_cols].values
        
        print("Golden_Sample initialized.")
        print(f"  - Historical states shape: {self.historical_states.shape}")
        print(f"  - Historical actions shape: {self.historical_actions.shape}")

    def find_best_action(self, current_state: np.ndarray) -> np.ndarray:
        """
        根據當前狀態尋找歷史數據中最佳的動作。

        Args:
            current_state (np.ndarray): 當前的環境狀態，應為 1D array，其維度需與 state_cols 數量相同。

        Returns:
            np.ndarray: 根據最相似歷史狀態找到的對應動作，為 1D array。
        """
        if current_state.ndim != 1:
            raise ValueError(f"current_state must be a 1D numpy array, but got shape {current_state.shape}")
        if current_state.shape[0] != self.historical_states.shape[1]:
             raise ValueError(f"Dimension mismatch: current_state has {current_state.shape[0]} features, but historical states have {self.historical_states.shape[1]} features.")

        # 為了計算距離，需要將 current_state 轉換為 2D array (1, n_features)
        current_state_reshaped = current_state.reshape(1, -1)

        # 計算當前狀態與所有歷史狀態之間的歐幾里得距離
        # euclidean_distances 會回傳一個 (1, n_historical_states) 的距離矩陣
        distances = euclidean_distances(current_state_reshaped, self.historical_states)
        
        # 找到距離最小的那個歷史狀態的索引
        most_similar_idx = np.argmin(distances)
        
        # 回傳該索引對應的歷史動作
        best_action = self.historical_actions[most_similar_idx]
        
        return best_action