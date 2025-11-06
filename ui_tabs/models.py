import os
import time
import numpy as np
import pandas as pd
from typing import Literal
from collections import deque
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

MODEL_LIST = ("LSTM_Model", "GRU_Model", "Transformer_Model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64, dense_units=64, num_layers=2, dropout=0.2):
        """
        一個基於長短期記憶網路 (LSTM) 的時序預測模型。

        Args:
            input_size (int): 輸入特徵的維度。
            output_size (int): 輸出預測的維度。
            hidden_size (int): LSTM 隱藏層的維度。
            dense_units (int): 全連接層的單元數。
            num_layers (int): LSTM 的層數。
            dropout (float): Dropout 的比例。
        """
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dense1 = nn.Linear(hidden_size, dense_units)
        self.bn1 = nn.BatchNorm1d(dense_units)

        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dense_units, output_size)

    def forward(self, x):
        """
        定義模型的前向傳播邏輯。

        Args:
            x (torch.Tensor): 輸入的時序數據張量。

        Returns:
            torch.Tensor: 模型的輸出張量。
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        out = self.dense1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.drop(out)

        out = self.output_layer(out)

        return out


class GRU_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64, dense_units=64, num_layers=2, dropout=0.2):
        """
        一個基於門控循環單元 (GRU) 的時序預測模型。

        Args:
            input_size (int): 輸入特徵的維度。
            output_size (int): 輸出預測的維度。
            hidden_size (int): GRU 隱藏層的維度。
            dense_units (int): 全連接層的單元數。
            num_layers (int): GRU 的層數。
            dropout (float): Dropout 的比例。
        """
        super(GRU_Model, self).__init__()
        self.lstm = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dense1 = nn.Linear(hidden_size, dense_units)
        self.bn1 = nn.BatchNorm1d(dense_units)

        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dense_units, output_size)

    def forward(self, x):
        """
        定義模型的前向傳播邏輯。

        Args:
            x (torch.Tensor): 輸入的時序數據張量。

        Returns:
            torch.Tensor: 模型的輸出張量。
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]

        out = self.dense1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.drop(out)

        out = self.output_layer(out)

        return out
    

class Transformer_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2, nhead=4, dense_units=64, dropout=0.1):
        """
        一個基於 Transformer Encoder 的時序預測模型。

        Args:
            input_size (int): 輸入特徵的維度。
            output_size (int): 輸出預測的維度。
            hidden_size (int): Transformer 模型的內部維度 (d_model)。
            num_layers (int): Transformer Encoder 的層數。
            nhead (int): 多頭注意力機制的頭數。
            dense_units (int): 輸出前全連接層的單元數。
            dropout (float): Dropout 的比例。
        """
        super(Transformer_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.input_fc = nn.Linear(input_size, hidden_size)
        
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=dropout,
            batch_first=True # 要配合下面轉置
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, output_size)
        )

    def forward(self, x):
        """
        定義模型的前向傳播邏輯。

        Args:
            x (torch.Tensor): 輸入的時序數據張量。

        Returns:
            torch.Tensor: 模型的輸出張量。
        """
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        
        out = self.transformer_encoder(x)
        
        out = out[:, -1, :]
        out = self.fc_out(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        實現經典的 sine-cosine 位置編碼。
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        為輸入張量增加位置編碼資訊。

        Args:
            x (torch.Tensor): 輸入張量。

        Returns:
            torch.Tensor: 增加了位置編碼的張量。
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    
def build_model(
        model_name: Literal["LSTM_Model", "GRU_Model", "Transformer_Model"],
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        *args,
        **kwargs
) -> torch.nn.Module:
    """
    根據指定的模型名稱和參數，建立並返回一個模型實例。

    Args:
        model_name (str): 要建立的模型名稱 ("LSTM_Model", "GRU_Model", "Transformer_Model")。
        input_size (int), hidden_size (int), num_layers (int), output_size (int): 模型的結構參數。
        *args, **kwargs: 傳遞給模型建構函式的其他參數。

    Returns:
        torch.nn.Module: 初始化的 PyTorch 模型實例。
    """
    if model_name == 'LSTM_Model':
        return LSTM_Model(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
    elif model_name == 'GRU_Model':
        return GRU_Model(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
    elif model_name == 'Transformer_Model':
        return Transformer_Model(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers
            *args, **kwargs
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def evaluate_model(model, data_loader, criterion):
    """
    在給定的資料集上評估模型的平均損失。

    Args:
        model (nn.Module): 要評估的 PyTorch 模型。
        data_loader (DataLoader): 包含評估數據的 DataLoader。
        criterion: 用於計算損失的損失函數。

    Returns:
        float: 在整個資料集上的平均損失值。
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate_r2(model, data_loader, scaler):
    """
    在給定的資料集上評估模型的 R-squared (R2) 分數。

    Args:
        model (nn.Module): 要評估的 PyTorch 模型。
        data_loader (DataLoader): 包含評估數據的 DataLoader。
        scaler: 用於將預測值和真實值反標準化的縮放器。

    Returns:
        float: 計算出的 R2 分數。
    """
    predictions, real_values = predict(model, data_loader, scaler)
    return metrics.r2_score(real_values, predictions)


class Golden_Sample:
    def __init__(self, df: pd.DataFrame, state_cols: list, action_cols: list):
        """
        一個基於歷史數據的專家系統，採用類似 k-NN 的思想。
        它會根據當前的環境狀態，從歷史數據庫中尋找最相似的狀態，並返回當時所採取的動作。

        Args:
            df (pd.DataFrame): 包含歷史狀態和動作的 DataFrame。
            state_cols (list): DataFrame 中代表「狀態(State)」的欄位名稱列表。
            action_cols (list): DataFrame 中代表「動作(Action)」的欄位名稱列表。
        """
        missing_state_cols = [col for col in state_cols if col not in df.columns]
        if missing_state_cols:
            raise ValueError(f"State columns not found in DataFrame: {missing_state_cols}")
        missing_action_cols = [col for col in action_cols if col not in df.columns]
        if missing_action_cols:
            raise ValueError(f"Action columns not found in DataFrame: {missing_action_cols}")

        self.historical_states = df[state_cols].values
        self.historical_actions = df[action_cols].values

        print("Golden_Sample initialized.")
        print(f"  - Historical states shape: {self.historical_states.shape}")
        print(f"  - Historical actions shape: {self.historical_actions.shape}")

    def find_best_action(self, current_state: np.ndarray) -> np.ndarray:
        """
        根據當前狀態，在歷史數據中尋找最相似的狀態並返回其對應的動作。

        Args:
            current_state (np.ndarray): 當前的環境狀態，應為 1D 陣列。

        Returns:
            np.ndarray: 歷史數據中與當前狀態最相似的狀態所對應的動作。
        """
        if current_state.ndim != 1:
            raise ValueError(f"current_state must be a 1D numpy array, but got shape {current_state.shape}")
        if current_state.shape[0] != self.historical_states.shape[1]:
             raise ValueError(f"Dimension mismatch: current_state has {current_state.shape[0]} features, but historical states have {self.historical_states.shape[1]} features.")

        current_state_reshaped = current_state.reshape(1, -1)
        distances = metrics.pairwise.euclidean_distances(current_state_reshaped, self.historical_states)
        most_similar_idx = np.argmin(distances)
        return self.historical_actions[most_similar_idx]


class ReplayBuffer:
    def __init__(self, capacity=1000, sequence_length=60):
        """
        一個用於強化學習的簡單回放緩衝區，支持序列採樣。

        Args:
            capacity (int): 緩衝區的最大容量。
            sequence_length (int): 用於採樣的序列長度。
        """
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def add(self, state, action, reward, next_state, done):
        """
        向緩衝區中添加一個經驗元組 (s, a, r, s', d)。
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        從緩衝區中隨機採樣一個批次的序列數據。

        Args:
            batch_size (int): 要採樣的序列數量。

        Returns:
            tuple or None: 包含狀態、動作、獎勵等批次數據的元組，如果緩衝區大小不足則返回 None。
        """
        if len(self.buffer) < self.sequence_length + 1:
            return None

        indices = np.random.choice(len(self.buffer) - self.sequence_length, batch_size, replace=False)
        batch = []
        for idx in indices:
            trajectory = list(self.buffer)[idx:idx + self.sequence_length]
            states, actions, rewards, next_states, dones = zip(*trajectory)
            batch.append((
                np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)
            ))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.FloatTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).unsqueeze(-1).to(DEVICE)
        )

class LSTMCritic(nn.Module):
    def __init__(self, state_dim, action_dim, dense_units=32, dropout=0.2):
        """
        SAC 中的 Critic 網路，使用全連接層來評估 Q 值。
        """
        super().__init__()
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1)
        )

    def forward(self, state, action):
        """
        計算給定狀態和動作的 Q 值。

        Args:
            state (torch.Tensor): 狀態張量。
            action (torch.Tensor): 動作張量。

        Returns:
            torch.Tensor: 預測的 Q 值。
        """
        return self.q_network(torch.cat([state, action], dim=-1))

class LSTMActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=4):
        """
        SAC 中的 Actor 網路，使用 LSTM 來處理序列狀態並生成動作。
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state_emb, sequence=True):
        """
        根據狀態嵌入生成動作。

        Args:
            state_emb (torch.Tensor): 經過編碼器處理的狀態嵌入。
            sequence (bool): 如果為 True，返回整個序列的動作；否則只返回最後一個時間步的動作。

        Returns:
            torch.Tensor: 生成的動作張量。
        """
        out, _ = self.lstm(state_emb)
        if not sequence:
            out = out[:, -1, :]
        action = torch.tanh(self.fc(out))
        return action

class SoftActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, encode_dim=64):
        """
        Soft Actor-Critic (SAC) 演算法的整合模型，包含 Actor、兩個 Critic 和一個狀態編碼器。
        """
        super().__init__()
        self.mu_layer = nn.Linear(encode_dim, action_dim)
        self.log_sigma_layer = nn.Linear(encode_dim, action_dim)

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encode_dim)
        )
        self.actor = LSTMActor(encode_dim, action_dim)
        self.critic_1 = LSTMCritic(state_dim, action_dim)
        self.critic_2 = LSTMCritic(state_dim, action_dim)

    def log_prob(self, state, action):
        """
        計算在給定狀態下，採取某個動作的對數機率。

        Args:
            state (torch.Tensor): 狀態張量。
            action (torch.Tensor): 動作張量。

        Returns:
            torch.Tensor: 對應動作的對數機率。
        """
        encoded = self.encoder(state)
        mu = self.mu_layer(encoded)
        log_sigma = self.log_sigma_layer(encoded).clamp(-20, 2)
        sigma = log_sigma.exp()
        dist = D.Normal(mu, sigma)
        return dist.log_prob(action).sum(-1, keepdim=True)

    def forward(self, state):
        """
        根據狀態生成確定性動作（主要用於推論）。

        Args:
            state (torch.Tensor): 狀態張量。

        Returns:
            torch.Tensor: 生成的動作。
        """
        encoded = self.encoder(state)
        actions = self.actor(encoded)
        return actions
    
    def evaluate_action(self, state, action):
        """
        由兩個 Critic 網路評估狀態-動作對的 Q 值。

        Args:
            state (torch.Tensor): 狀態序列張量。
            action (torch.Tensor): 動作序列張量。

        Returns:
            tuple: 兩個 Critic 網路各自預測的 Q 值。
        """
        last_state = state[:, -1, :]
        last_action = action[:, -1, :]
        q1 = self.critic_1(last_state, last_action)
        q2 = self.critic_2(last_state, last_action)
        return q1, q2

def compute_reward(row_array, reward_keys, w_pf=1.0, w_p=1.0, w_kw_ratio=1.0, epsilon=1e-6, **kwargs):
    """
    根據環境模型預測的多個獎勵相關特徵，加權計算出一個純量的獎勵值。

    Args:
        row_array (np.ndarray or pd.Series): 包含獎勵相關特徵的一行數據。
        reward_keys (list): 獎勵特徵的名稱列表。
        w_pf (float), w_p (float), w_kw_ratio (float): 各獎勵項的權重。
        epsilon (float): 用於避免除以零的小常數。

    Returns:
        float: 計算出的純量獎勵值。
    """
    if isinstance(row_array, pd.Series):
        row_dict = row_array.to_dict()
    else:
        row_dict = dict(zip(reward_keys, row_array))

    reward = 0.0
    pf = row_dict.get(next((k for k in reward_keys if 'PF_avg' in k), "PF_avg"), 0)
    kw = row_dict.get(next((k for k in reward_keys if 'KW_tot' in k), "KW_tot"), 0)
    kvar = row_dict.get(next((k for k in reward_keys if 'Kvar_tot' in k), "Kvar_tot"), 0)
    power_sum = kw + kvar + epsilon
    reward += w_pf * pf
    reward -= w_p * power_sum
    reward += w_kw_ratio * (kw / power_sum)
    return reward

def _train_sac_step(sac, optimizers, schedulers, environment_model, loss_fn, reward_keys, batch, gamma, alpha, is_warmup=False, **reward_kwargs):
    """
    執行單個批次的 SAC 訓練步驟，包括 Critic 和 Actor 的更新。

    Returns:
        tuple: 一個包含 (actor_loss, critic_loss) 的元組。
    """
    states, actions, _, next_states, dones = batch
    
    with torch.no_grad():
        env_input = torch.cat([states, actions], dim=-1)
        reward_features = environment_model(env_input)
        reward_features_np = reward_features.cpu().numpy()
        
        scalar_rewards = np.apply_along_axis(compute_reward, 1, reward_features_np, reward_keys=reward_keys, **reward_kwargs).astype(np.float32)
        scalar_rewards = torch.FloatTensor(scalar_rewards).unsqueeze(-1).to(DEVICE)

        next_actions = sac(next_states)
        q1_next, q2_next = sac.evaluate_action(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next)
        log_prob_next = sac.log_prob(next_states[:, -1, :], next_actions[:, -1, :])
        target_q = scalar_rewards + gamma * (1 - dones[:, -1, :].float()) * (q_next - alpha * log_prob_next)

    q1, q2 = sac.evaluate_action(states, actions)
    critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)

    optimizers['critic'].zero_grad()
    critic_loss.backward()
    optimizers['critic'].step()
    schedulers['critic'].step()

    actor_loss = torch.tensor(0.0)
    if not is_warmup:
        actions_pred = sac(states)
        p1_pred, p2_pred = sac.evaluate_action(states, actions_pred)
        log_prob_pred = sac.log_prob(states[:, -1, :], actions_pred[:, -1, :])
        actor_loss = (alpha * log_prob_pred - torch.min(p1_pred, p2_pred)).mean()

        optimizers['actor'].zero_grad()
        actor_loss.backward()
        optimizers['actor'].step()
        schedulers['actor'].step()

    return actor_loss.item(), critic_loss.item()

def save_best_models(sac, save_dir, model_type, actor_loss, critic_loss, best_losses, timestamp):
    """
    檢查當前損失是否優於歷史最佳紀錄，如果是，則儲存對應的模型權重。
    """
    actor_best_loss, critic_best_loss = best_losses

    if actor_loss < actor_best_loss:
        actor_state = {
            'actor': sac.actor.state_dict(),
            'encoder': sac.encoder.state_dict()
        }
        path = os.path.join(save_dir, f"{timestamp}_{model_type}_ACTOR.pth")
        torch.save(actor_state, path)
        actor_best_loss = actor_loss
        print(f"  -> New best actor model saved with loss: {actor_loss:.4f}")

    if critic_loss < critic_best_loss:
        path1 = os.path.join(save_dir, f"{timestamp}_{model_type}_CRITIC1.pth")
        path2 = os.path.join(save_dir, f"{timestamp}_{model_type}_CRITIC2.pth")
        torch.save(sac.critic_1.state_dict(), path1)
        torch.save(sac.critic_2.state_dict(), path2)
        critic_best_loss = critic_loss
        print(f"  -> New best critic models saved with loss: {critic_loss:.4f}")
    return actor_best_loss, critic_best_loss

def train_sac_agent(
        sac: SoftActorCritic,
        environment_model: nn.Module,
        initial_states: np.ndarray,
        reward_keys: list,
        train_buffer: ReplayBuffer,
        save_dir: str,
        model_type: str = "SAC_LSTM",
        batch_size: int = 128,
        epochs: int = 100,
        steps: int = 1000,
        warmup_steps: int = 10000,
        gamma: float = 0.99,
        alpha: float = 0.2,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        early_stop_patience: int = 10
    ):
    """
    訓練 SAC Agent 的主循環。這是一個生成器函式，會在每個訓練週期結束後 `yield` 當前的訓練狀態。

    Args:
        sac (SoftActorCritic): 要訓練的 SAC 模型。
        environment_model (nn.Module): 已訓練的環境模型。
        initial_states (np.ndarray): 用於生成初始狀態的數據集。
        reward_keys (list): 獎勵相關的欄位名稱。
        train_buffer (ReplayBuffer): 回放緩衝區。
        save_dir (str): 模型儲存目錄。
        model_type (str): 模型類型名稱，用於儲存檔名。
        batch_size (int), epochs (int), steps (int), warmup_steps (int): 訓練過程控制參數。
        gamma (float), alpha (float), lr_actor (float), lr_critic (float): SAC 演算法超參數。
        early_stop_patience (int): 早停的耐心值（基於 Critic 損失）。

    Yields:
        tuple: 每個週期結束時，產生一個包含 (actor_loss, critic_loss, status_message) 的元組。
               訓練結束或早停時，產生 (None, None, final_message)。
    """
    loss_fn = nn.MSELoss()
    optimizers = {
        'actor': optim.Adam(sac.actor.parameters(), lr=lr_actor, weight_decay=1e-4),
        'critic': optim.Adam(list(sac.critic_1.parameters()) + list(sac.critic_2.parameters()), lr=lr_critic, weight_decay=1e-4)
    }
    schedulers = {
        'actor': optim.lr_scheduler.StepLR(optimizers['actor'], step_size=10, gamma=0.1),
        'critic': optim.lr_scheduler.StepLR(optimizers['critic'], step_size=10, gamma=0.1)
    }

    sac.to(DEVICE)
    environment_model.to(DEVICE).eval()

    best_losses = (float('inf'), float('inf'))
    total_steps = 0
    no_improvement_count = 0
    training_timestamp = time.strftime('%Y%m%d-%H%M%S')

    for episode in range(epochs):
        episode_actor_loss, episode_critic_loss, episode_steps = 0.0, 0.0, 0
        initial_idx = np.random.randint(0, len(initial_states) - 1)
        state = initial_states[initial_idx]
        
        for t in range(steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action_tensor = sac(state_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()

            with torch.no_grad():
                env_input = torch.cat([state_tensor, action_tensor], dim=-1)
                reward_features = environment_model(env_input)
                reward_features_np = reward_features.cpu().numpy()[0]
                scalar_reward = compute_reward(reward_features_np, reward_keys=reward_keys)

                next_state_idx = initial_idx + t + 1
                if next_state_idx < len(initial_states):
                    next_state = initial_states[next_state_idx]
                    done = False
                else:
                    next_state = state
                    done = True

            train_buffer.add(state, action.flatten(), scalar_reward, next_state, done)
            state = next_state
            total_steps += 1

            if len(train_buffer.buffer) >= batch_size + train_buffer.sequence_length:
                batch = train_buffer.sample(batch_size)
                if batch is None:
                    continue
                
                is_warmup = total_steps < warmup_steps
                actor_loss, critic_loss = _train_sac_step(
                    sac, optimizers, schedulers, environment_model, loss_fn, 
                    reward_keys, batch, gamma, alpha, is_warmup=is_warmup
                )
                    
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                episode_steps += 1

            if done:
                break

        
        avg_actor_loss = episode_actor_loss / episode_steps if episode_steps > 0 else 0.0
        avg_critic_loss = episode_critic_loss / episode_steps if episode_steps > 0 else 0.0

        status = (
            f"Episode {episode+1}/{epochs} - "
            f"Actor Loss: {avg_actor_loss:.4f} - "
            f"Critic Loss: {avg_critic_loss:.4f}"
        )
        
        prev_best_critic_loss = best_losses[1]
        best_losses = save_best_models(sac, save_dir, model_type, avg_actor_loss, avg_critic_loss, best_losses, training_timestamp)
        
        if best_losses[1] < prev_best_critic_loss:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        yield avg_actor_loss, avg_critic_loss, status

        if no_improvement_count >= early_stop_patience:
            yield None, None, f"Early stopping at episode {episode+1} due to no improvement."
            break

    yield None, None, "Training finished."
def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler,
        num_epochs:int,
        save_dir: str,
        early_stopping:int
    ):
    """
    訓練監督式學習模型（如 LSTM, GRU）的主循環。這是一個生成器函式，會在每個訓練週期結束後 `yield` 當前的訓練狀態。

    Args:
        model (nn.Module): 要訓練的 PyTorch 模型。
        train_loader (DataLoader): 訓練數據加載器。
        val_loader (DataLoader): 驗證數據加載器。
        criterion: 損失函數。
        optimizer: 優化器。
        scheduler: 學習率排程器。
        scaler: 用於在評估 R2 分數時反標準化數據的縮放器。
        num_epochs (int): 訓練週期總數。
        save_dir (str): 模型儲存目錄。
        early_stopping (int): 早停的耐心值（基於驗證集損失）。

    Yields:
        tuple: 每個週期結束時，產生一個包含 (avg_loss, val_loss, val_r2, lr_now, status_message) 的元組。
    """
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model.to(DEVICE)

    for epoch in range(num_epochs):

        model.train()
        best_loss = float('inf')
        record = 0
        total_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = evaluate_model(model, val_loader, criterion)
        val_r2 = evaluate_r2(model, val_loader, scaler)
        avg_loss = total_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]['lr']
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/MODEL_{timestamp}.pth")
            record = 0
        else:
            record += 1
            if record >= early_stopping:
                yield avg_loss, val_loss, val_r2, lr_now, f'Early stopping at epoch {epoch+1}'
                break
        
        scheduler.step(val_loss)
        
        yield avg_loss, val_loss, val_r2, lr_now, (
            f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - val_Loss: {val_loss:.4f} - val_r2: {val_r2:.4f} - LR: {lr_now:.6f}"
        )

def sac_get_action(
        sac_model:nn.Module,
        current_state: np.ndarray
    ) -> np.ndarray:
    """
    使用已訓練的 SAC 模型，根據當前狀態獲取一個確定性的最佳動作。

    Args:
        sac_model (nn.Module): 已載入權重的 SoftActorCritic 模型。
        current_state (np.ndarray): 當前的環境狀態，應為 1D NumPy 陣列。

    Returns:
        np.ndarray: 模型預測的最佳動作，為 1D NumPy 陣列。
    """
    sac_model = sac_model.to(DEVICE)
    sac_model.eval()

    state_tensor = torch.FloatTensor(current_state).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        action_tensor = sac_model(state_tensor)

    return action_tensor[:, -1, :].squeeze(0).cpu().numpy()

def golden_sample_get_action(
        golden_sample_expert: Golden_Sample,
        current_state: np.ndarray
    ) -> np.ndarray:
    """
    使用 Golden_Sample 專家系統，根據當前狀態從歷史數據中找到最佳動作。

    Args:
        golden_sample_expert (Golden_Sample): 已初始化的 Golden_Sample 實例。
        current_state (np.ndarray): 當前的環境狀態 (未標準化)。

    Returns:
        np.ndarray: 找到的最佳歷史動作。
    """
    return golden_sample_expert.find_best_action(current_state)

def predict(
        model:torch.nn.Module,
        data_loader:torch.utils.data.DataLoader,
        scaler
    )-> torch.Tensor:
    """
    使用訓練好的模型對數據進行預測，並將預測結果和真實標籤反標準化。

    Args:
        model (nn.Module): 訓練好的 PyTorch 模型。
        data_loader (DataLoader): 包含預測數據的 DataLoader。
        scaler: 用於反標準化的縮放器。

    Returns:
        tuple: 一個包含 (反標準化後的預測值, 反標準化後的真實值) 的元組。
    """
    model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(DEVICE)
            output = model(data)
            predictions.append(scaler.inverse_transform(output.cpu().numpy()))
            real_values.append(scaler.inverse_transform(target.cpu().numpy()))
    
    return np.concatenate(predictions, axis=0), np.concatenate(real_values, axis=0) 