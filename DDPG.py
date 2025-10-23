import os
import time
import wandb
from tqdm import tqdm
from collections import deque
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

import models



# 狀態向量示例：選取室內與外部環境多項溫度、濕度、電表功率等
STATE_KEYS = [
    "dP_Sys", "dP_Head",
    "Weather.T", "Weather.H",
    "T_CHW_out",  "T_CHW_in",
    "T_CW_out", "T_CW_in",
    "F_CHW_in", "F_CW_in",
    "dT_PowerOn", "dT_Shutdown",
    "Load", "Signal", "Status",
    "I_left", "I_right",
    "Status_left", "Status_right",
    "VLN_R", "VLN_S", "VLN_T", "VLN_avg",
    "I_R", "I_S", "I_T", "I_avg",
    "PF_avg",
    "KW_tot", "Kvar_tot", "KVA_tot"

]

# 動作定義示例：冰水機溫度設定點和主機啟停信號
ACTION_KEYS = [
    "T_SP"
]

REWARD_KEYS = [
    "PF_avg",
    "KW_tot", "Kvar_tot"
]

MACHINE_LIST = [
    "Chiller_1", "Chiller_2",
    "CWP_1", "CWP_SP", "CWP_2",
    "CHP_1", "CHP_SP", "CHP_2",
    "CT_1", "CT_2"
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_TYPE = "SAC_LSTM"
ENVIRONMENT_PATH = "/mnt/c/UEO_AI/model_record/environment/model_LSTM_20250921-180228.pth" # 假設這是預訓練好的環境模型

# 超參數
GAMMA = 0.99
ALPHA = 0.2
BATCH_SIZE = 128
EPOCHS = 100
MAX_STEPS = 1000
WARMUP_EPOCHS = 10  # 預熱階段的epoch數
EARLY_STOP = 10
LEARNING_RATE = 3e-4
REPLAY_BUFFER_CAPACITY = 10000
SEQUENCE_LENGTH = 60



class ReplayBuffer:
    def __init__(self, capacity=1000, sequence_length=SEQUENCE_LENGTH):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 確保緩衝區中有足夠的數據來採樣一個完整的序列
        if len(self.buffer) < self.sequence_length + 1:
            return None

        # 起點必須留夠序列長度
        indices = np.random.choice(len(self.buffer) - self.sequence_length, batch_size, replace=False)
        batch = []
        for idx in indices:
            # 取出一段 sequence_length 的 trajectory
            trajectory = list(self.buffer)[idx:idx + self.sequence_length]
            states, actions, rewards, next_states, dones = zip(*trajectory)
            batch.append((
                np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones)
            ))
        # 組 batch
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),      # (batch, seq_len, state_dim)
            torch.FloatTensor(np.array(actions)).to(DEVICE),
            torch.FloatTensor(np.array(rewards)).unsqueeze(-1).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(np.array(dones)).unsqueeze(-1).to(DEVICE) # Ensure it's float for calculations
        )


class LSTMCritic(nn.Module):
    def __init__(self, state_dim, action_dim, dense_units=32, dropout=0.2):
        super().__init__()
        # Q-value head, 輸入 state 和 action
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1)
        )

    def forward(self, state, action):
        out = self.q_network(torch.cat([state, action], dim=-1))
        return out


class LSTMActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=4):
        super().__init__()
        # LSTM主體
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        # 尾部全連接
        self.fc = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state_emb, sequence=True):
        """
        state_seq: (batch, seq_len, state_dim)
        return: (batch, seq_len, action_dim) 或 (batch, action_dim)
        """
        out, _ = self.lstm(state_emb)  # (batch, seq_len, state_dim)
        if not sequence:
            out = out[:, -1, :]    # (batch, state_dim)
        action = torch.tanh(self.fc(out))  # (batch, action_dim)
        return action


class SoftActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, encode_dim=64):
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
        encoded = self.encoder(state)
        mu = self.mu_layer(encoded)
        log_sigma = self.log_sigma_layer(encoded).clamp(-20, 2)
        sigma = log_sigma.exp()
        dist = D.Normal(mu, sigma)
        return dist.log_prob(action).sum(-1, keepdim=True)

    def forward(self, state):
        encoded = self.encoder(state)
        actions = self.actor(encoded)
        return actions
    
    def evaluate_action(self, state, action):
        # Critic 現在接收最後一個時間步的 state 和 action
        last_state = state[:, -1, :]
        last_action = action[:, -1, :]
        q1 = self.critic_1(last_state, last_action)
        q2 = self.critic_2(last_state, last_action)
        return q1, q2
    


# reward計算函數示例
def compute_reward(
        df,
        machine_prefixes=MACHINE_LIST,
        w_pf=1.0,
        w_p=1.0,
        w_kw_ratio=1.0,
        epsilon=1e-6
    ):
    """
    df: DataFrame，含所有機器數據
    machine_prefixes: 機器名列表，如 ['Chiller_1', 'Chiller_2', 'CWP_1']
    欄位規則假設為 '{machine_prefix}.PF_avg', '{machine_prefix}.KW_tot', '{machine_prefix}.Kvar_tot'

    回傳浮點數reward
    """

    reward = 0.0

    for m in machine_prefixes:
        pf = df[f"{m}_PF_avg"]
        kw = df[f"{m}_KW_tot"]
        kvar = df[f"{m}_Kvar_tot"]
        power_sum = kw + kvar + epsilon

        reward += w_pf * pf
        reward -= w_p * power_sum
        reward += w_kw_ratio * (kw / power_sum)

    return reward


def train_sac_step(
        sac: SoftActorCritic,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        actor_scheduler: optim.lr_scheduler,
        critic_scheduler: optim.lr_scheduler,
        environment_model: nn.Module,
        loss_fn: nn.Module,
        full_reward_keys: list,
        batch,
        gamma=GAMMA,
        alpha=ALPHA
    ):
    states, actions, rewards, next_states, dones = batch
    
    # 1. 使用環境模型預測獎勵
    with torch.no_grad():
        # 將 state 和 action 序列拼接，輸入環境模型
        # 我們關心的是執行 action 後得到的 reward，所以用 (states, actions) 預測
        env_input = torch.cat([states, actions], dim=-1)
        reward_features = environment_model(env_input) # 預期 LSTM_Model 輸出 (batch, reward_dim)
        
        # 根據 environment_model 的實際輸出維度處理
        if reward_features.dim() == 2: # [batch, reward_dim]
            last_reward_features = reward_features.cpu().numpy()
        else: # [batch, seq_len, reward_dim]
            # 如果環境模型被修改為輸出完整序列，則取最後一個時間步
            last_reward_features = reward_features[:, -1, :].cpu().numpy()
        
        # 2. 將多維獎勵特徵轉換為純量獎勵
        scalar_rewards = pd.DataFrame(last_reward_features, columns=full_reward_keys).apply(compute_reward, axis=1).values.astype(np.float32)
        scalar_rewards = torch.FloatTensor(scalar_rewards).unsqueeze(-1).to(DEVICE)

    # 3. 計算 Target Q value
    with torch.no_grad():
        next_actions = sac(next_states)
        q1_next, q2_next = sac.evaluate_action(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next)
        target_q = scalar_rewards + gamma * (1 - dones[:, -1, :].float()) * (q_next - alpha * sac.log_prob(next_states[:, -1, :], next_actions[:, -1, :])) # log_prob 應作用於最後一個時間步

    # 4. Critic Loss
    q1, q2 = sac.evaluate_action(states, actions) # evaluate_action 內部會取最後一個時間步
    critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # 5. Actor Loss
    actions_pred = sac(states) # sac(states) 返回 (batch, seq_len, action_dim)
    p1_pred, p2_pred = sac.evaluate_action(states, actions_pred) # evaluate_action 內部會取最後一個時間步
    actor_loss = (alpha * sac.log_prob(states[:, -1, :], actions_pred[:, -1, :]) - torch.min(p1_pred, p2_pred)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    actor_scheduler.step()
    critic_scheduler.step()

    return actor_loss.item(), critic_loss.item()


def save_best_models(sac, timestamp, model_type, actor_loss, critic_loss, 
                     actor_best_loss, critic_best_loss):
    """統一處理模型保存，避免重複"""
    current_actor_best_loss = actor_best_loss
    current_critic_best_loss = critic_best_loss

    if actor_loss < actor_best_loss:
        torch.save(sac.encoder.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_ENCODER.pth')
        torch.save(sac.actor.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_ACTOR.pth')
        torch.save(sac.mu_layer.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_MU.pth')
        torch.save(sac.log_sigma_layer.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_SIGMA.pth')
        current_actor_best_loss = actor_loss
        #print(f"Actor models saved at epoch {timestamp} with loss {actor_loss:.4f}")
    
    if critic_loss < critic_best_loss:
        torch.save(sac.critic_1.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_CRITIC1.pth')
        torch.save(sac.critic_2.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_CRITIC2.pth')
        torch.save(sac.encoder.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_ENCODER.pth') # Encoder也與Critic相關
        torch.save(sac.mu_layer.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_MU.pth')
        torch.save(sac.log_sigma_layer.state_dict(), f'model_record/DDPG/{timestamp}_{model_type}_SIGMA.pth')
        current_critic_best_loss = critic_loss
        #print(f"Critic models saved at epoch {timestamp} with loss {critic_loss:.4f}")
    
    return current_actor_best_loss, current_critic_best_loss


def _perform_warmup_step(
    sac,
    batch,
    critic_optimizer,
    critic_scheduler,
    environment_model,
    loss_fn,
    full_reward_keys,
    gamma,
    alpha,
):
    """Performs a single training step for the critic during the warmup phase."""
    states, actions, _, next_states, dones = batch
    with torch.no_grad():
        env_input = torch.cat([states, actions], dim=-1)
        reward_features = environment_model(env_input)

        if reward_features.dim() == 2:  # [batch, reward_dim]
            last_reward_features = reward_features.cpu().numpy()
        else:
            last_reward_features = reward_features[:, -1, :].cpu().numpy()

        scalar_rewards = pd.DataFrame(last_reward_features, columns=full_reward_keys).apply(compute_reward, axis=1).values.astype(np.float32)
        scalar_rewards = torch.FloatTensor(scalar_rewards).unsqueeze(-1).to(DEVICE)

        next_actions = sac(next_states)
        q1_next, q2_next = sac.evaluate_action(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next)
        target_q = scalar_rewards + gamma * (1 - dones[:, -1, :].float()) * (q_next - alpha * sac.log_prob(next_states[:, -1, :], next_actions[:, -1, :]))

    q1, q2 = sac.evaluate_action(states, actions)
    critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    critic_scheduler.step()
    return 0.0, critic_loss.item()


def _perform_training_step(
    sac,
    batch,
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    environment_model,
    loss_fn,
    full_reward_keys,
    gamma,
    alpha,
):
    """Performs a single training step for both actor and critic."""
    return train_sac_step(
        sac,
        actor_optimizer,
        critic_optimizer,
        actor_scheduler,
        critic_scheduler,
        environment_model,
        loss_fn,
        full_reward_keys,
        batch,
        gamma=gamma,
        alpha=alpha,
    )


def train(
        sac: SoftActorCritic,
        environment_model: nn.Module,
        initial_states: np.ndarray,
        full_reward_keys: list,
        train_buffer: object,
        batch_size:int=BATCH_SIZE,
        epochs:int=EPOCHS,
        steps:int=MAX_STEPS,
        warmup_epochs:int=WARMUP_EPOCHS,
        device:str=DEVICE,
        gamma:float=GAMMA,
        alpha:float=ALPHA,
        lr_actor:float=LEARNING_RATE,
        lr_critic:float=LEARNING_RATE,
        early_stop:int=EARLY_STOP
    ):

    loss_fn = nn.MSELoss()

    actor_optimizer = optim.Adam(sac.actor.parameters(), lr=lr_actor, weight_decay=1e-4)
    critic_optimizer = optim.Adam(list(sac.critic_1.parameters()) + list(sac.critic_2.parameters()), lr=lr_critic, weight_decay=1e-4)
    
    actor_scheduler = optim.lr_scheduler.StepLR(actor_optimizer, step_size=warmup_epochs, gamma=0.1)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_optimizer, step_size=warmup_epochs, gamma=0.1)

    sac.to(device)
    environment_model.to(device).eval() # 環境模型設為評估模式

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    wandb.init(
        project="UEO_AI DDPG",
        name=f"Training_{MODEL_TYPE}_{timestamp}",
        config={
        "lr_actor": lr_actor,
        "lr_critic": lr_critic,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "batch_size": batch_size,
        "gamma": gamma,
        "alpha": alpha,
        "device": device
    })

    actor_best_loss, critic_best_loss = float('inf'), float('inf')
    is_warmed_up = False

    total_steps = 0
    for episode in range(epochs):
        episode_actor_loss = 0.0
        episode_critic_loss = 0.0
        episode_steps = 0
        # 每一回合開始時，從初始狀態數據中隨機選擇一個起點
        initial_idx = np.random.randint(0, len(initial_states) - 1)
        state = initial_states[initial_idx]
        
        # 假設每回合最多走 1000 步
        for t in tqdm(range(steps), desc=f"Training [{episode} / {epochs}] :"):
            # 1. 選擇動作
            # 將單步 state 擴展為符合模型輸入的序列 (batch=1, seq=1, dim)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                # 由於 Actor 是 LSTM，即使只有一步，也需要序列輸入
                action_tensor = sac(state_tensor) # (1, 1, action_dim)
            action = action_tensor.squeeze(0).cpu().numpy() # (1, action_dim)

            # 2. 與模擬環境互動
            with torch.no_grad():
                # 預測獎勵
                env_input = torch.cat([state_tensor, action_tensor], dim=-1)
                reward_features = environment_model(env_input) # (1, reward_dim)
                reward_df = pd.DataFrame(reward_features.cpu().numpy(), columns=full_reward_keys)
                scalar_reward = reward_df.apply(compute_reward, axis=1).iloc[0]

                # 模擬下一個狀態 (簡化：從數據集中取下一個時間點的狀態)
                next_state_idx = initial_idx + t + 1
                if next_state_idx < len(initial_states):
                    next_state = initial_states[next_state_idx]
                    done = False
                else: # 如果超出數據集，則回合結束
                    next_state = state # 用當前狀態填充
                    done = True

            # 3. 將經驗存入 Replay Buffer
            # state, action, next_state 都是 (dim,) 的 numpy array
            train_buffer.add(state, action.flatten(), scalar_reward, next_state, done)

            # 4. 更新狀態
            state = next_state
            total_steps += 1

            # 5. 檢查是否可以開始訓練
            # The number of possible start indices must be >= batch_size
            if len(train_buffer.buffer) >= batch_size + SEQUENCE_LENGTH:
                batch = train_buffer.sample(batch_size)
                # Ensure batch is not None before proceeding
                if batch is None:
                    continue
                
                if total_steps < warmup_epochs:
                    if not is_warmed_up:
                        print("Entering warmup phase...")
                        # Freeze Actor parameters only once at the beginning of warmup
                        for param in sac.actor.parameters():
                            param.requires_grad = False
                        is_warmed_up = True

                    # Warmup 階段只更新 Critic
                    actor_loss, critic_loss = _perform_warmup_step(
                        sac, batch, critic_optimizer, critic_scheduler, environment_model,
                        loss_fn, full_reward_keys, gamma, alpha
                    )
                else:
                    if is_warmed_up:
                        print("Warmup complete. Unfreezing actor parameters...")
                        # Unfreeze Actor parameters only once after warmup
                        for param in sac.actor.parameters():
                            param.requires_grad = True
                        is_warmed_up = False # Prevent this block from running again

                    # 正常訓練
                    actor_loss, critic_loss = _perform_training_step(
                        sac, batch, actor_optimizer, critic_optimizer, actor_scheduler,
                        critic_scheduler, environment_model, loss_fn, full_reward_keys,
                        gamma, alpha
                    )
                    
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                episode_steps += 1

            if done:
                break

        # 在每個 episode 結束後，記錄平均 loss 並儲存模型
        if episode_steps > 0:
            avg_actor_loss = episode_actor_loss / episode_steps
            avg_critic_loss = episode_critic_loss / episode_steps

            # 每個 episode 結束後才記錄一次 Wandb
            wandb.log({
                "episode": episode,
                "avg_actor_loss": avg_actor_loss,
                "avg_critic_loss": avg_critic_loss
            })

            actor_best_loss, critic_best_loss = save_best_models(
                sac, timestamp, MODEL_TYPE, avg_actor_loss, avg_critic_loss, actor_best_loss, critic_best_loss
            )

        if total_steps > WARMUP_EPOCHS and (actor_best_loss == float('inf') or critic_best_loss == float('inf')):
            if total_steps > WARMUP_EPOCHS + early_stop:
                print(f"Early stopping due to no improvement after warmup.")
                break

    wandb.finish()


if __name__ == '__main__':
    import Preprocessing 
    
    train_csv_file = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train.csv'))


    state_cols = []
    action_cols = []
    reward_cols = []

    # 假設特徵列是所有非目標列
    for col in train_csv_file.columns:

        if col == 'DateTime':
            continue

        if any([col.endswith(i) for i in REWARD_KEYS]):
            reward_cols.append(col)

        if any([col.endswith(i) for i in ACTION_KEYS]):
            action_cols.append(col)

        if any([col.endswith(i) for i in STATE_KEYS]):
            state_cols.append(col)

    
    state_dim = len(state_cols)
    action_dim = len(action_cols)
    reward_dim = len(reward_cols)

    # 為了初始化環境，我們需要從真實數據中讀取一些初始狀態
    # 這裡我們只做一次性的數據預處理來獲取標準化後的狀態數據
    # 1. 清洗數據：處理時間格式、離群值和無窮大/缺失值
    cleaned_df = Preprocessing.clean_data(train_csv_file, datetime_col='DateTime')
    cleaned_df = Preprocessing.remove_outliers_iqr(cleaned_df)
    cleaned_df = Preprocessing.fill_missing(cleaned_df, strategy='mean')
    # 2. 特徵縮放
    initial_states_df, state_scaler = Preprocessing.scale_features(cleaned_df[state_cols], method='minmax')
    initial_states_np = initial_states_df.values.astype(np.float32)

    # 由於現在獎勵由環境模型產生，ReplayBuffer不再需要預先填充獎勵
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY) 

    # 1. 載入作為環境的 LSTM 模型
    environment_model = models.LSTM_Model(
        input_size=state_dim + action_dim,
        output_size=reward_dim
    ).to(DEVICE)
    if os.path.exists(ENVIRONMENT_PATH):
        environment_model.load_state_dict(torch.load(ENVIRONMENT_PATH, map_location=DEVICE))
        print("Environment model loaded successfully.")
    else:
        print(f"Warning: Environment model not found at {ENVIRONMENT_PATH}. Training with uninitialized model.")

    # 檢查環境模型輸出形狀
    with torch.no_grad():
        dummy_states = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, state_dim).to(DEVICE)
        dummy_actions = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, action_dim).to(DEVICE)
        dummy_input = torch.cat([dummy_states, dummy_actions], dim=-1)
        dummy_output = environment_model(dummy_input)
        print(f"Environment model output shape for dummy input: {dummy_output.shape}")
        # 期望: [batch_size, reward_dim] 因為 models.LSTM_Model 總是取最後一個時間步

    # 2. 建立 SAC Agent
    sac = SoftActorCritic(state_dim, action_dim, reward_dim).to(DEVICE)
    
    train(sac, environment_model, initial_states_np, reward_cols, replay_buffer)