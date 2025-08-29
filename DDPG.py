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

import Preprocessing
import environment



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
ACTOR_PATH = ""
ENVIRONMENT_PATH = "/mnt/c/UEO_AI/model_record/environment/model__20250829-000459.pth"
CRITIC1_REWARD_PATH = "/mnt/c/UEO_AI/model_record/DDPG/20250829-021916_SAC_LSTM_REWARD1.pth"
CRITIC2_REWARD_PATH = "/mnt/c/UEO_AI/model_record/DDPG/20250829-021916_SAC_LSTM_REWARD2.pth"
ENCODER_PATH = ""
MU_LAYER_PATH = "/mnt/c/UEO_AI/model_record/DDPG/20250829-021916_SAC_LSTM_MU.pth"
LOG_SIGMA_LAYER_PATH = "/mnt/c/UEO_AI/model_record/DDPG/20250829-021916_SAC_LSTM_SIGMA.pth"

# 超參數
GAMMA = 0.99
ALPHA = 0.2
BATCH_SIZE = 512
EPOCHS = 10000
WARMUP_EPOCHS = 1000  # 預熱階段的epoch數
EARLY_STOP = 1000
LEARNING_RATE = 3e-4
SEQUENCE_LENGTH = 60



class ReplayBuffer:
    def __init__(self, capacity=1000, sequence_length=SEQUENCE_LENGTH):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
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
            torch.FloatTensor(np.array(dones)).unsqueeze(-1).to(DEVICE)
        )


class LSTMCritic(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim, dense_units=32, dropout=0.2):
        super().__init__()
        
        # LSTM 用於捕捉序列資訊
        self.predict = environment.Predict_Model(state_dim+action_dim, reward_dim)
        # Q-value head
        self.reward = nn.Sequential(
            nn.Linear(reward_dim, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, 1)
        )

    def forward(self, state_seq, action_seq):
        out = self.predict(torch.cat([state_seq, action_seq], dim=-1))
        out = self.reward(out)
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

        self.critic_1 = LSTMCritic(state_dim, action_dim, reward_dim)
        self.critic_2 = LSTMCritic(state_dim, action_dim, reward_dim)

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
        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)
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


def preprocess_data(
        df:pd.DataFrame,
        replay_buffer:ReplayBuffer,
        state_keys, action_keys, reward_keys,
        datetime_col='DateTime',
        strategy='mean',
        scalers:dict={}
    ):

    df = Preprocessing.clean_data(df, datetime_col)
    df = Preprocessing.remove_outliers_iqr(df, factor=1.5)
    df = Preprocessing.fill_missing(df, strategy=strategy)

    # 明確保持欄位順序，用list串接，且避免set破壞順序
    all_cols = state_keys + action_keys
    seen = set()
    all_cols_unique = [x for x in all_cols if not (x in seen or seen.add(x))]
    
    df_filtered = df[all_cols_unique].copy()
    states, state_scaler = Preprocessing.scale_features(
        df_filtered[state_keys],
        method='minmax',
        scaler=scalers.get('state', None)
    )
    actions, action_scaler = Preprocessing.scale_features(
        df_filtered[action_keys],
        method='minmax',
        scaler=scalers.get('action', None)
    )
    rewards, reward_scaler = Preprocessing.scale_features(
        df_filtered[reward_keys],
        method='minmax',
        scaler=scalers.get('reward', None)
    )

    # 計算reward列
    rewards = rewards.apply(compute_reward, axis=1)

    next_states = states.values[1:]
    states = states.values[:-1]
    actions = actions.values[:-1]
    rewards = rewards.values[:-1]
    dones = np.zeros(len(states))

    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
        replay_buffer.add(s, a, r, ns, d)
    
    return state_scaler, action_scaler, reward_scaler

    
def train_sac_step(
        sac: SoftActorCritic,
        actor_optimizer: optim.Optimizer,
        critic_optimizer: optim.Optimizer,
        actor_scheduler: optim.lr_scheduler,
        critic_scheduler: optim.lr_scheduler,
        loss_fn: nn.Module,
        batch,
        gamma=GAMMA,
        alpha=ALPHA
    ):
    states, actions, rewards, next_states, dones = batch

    with torch.no_grad():
        next_actions = sac(next_states[:, :-1, :])
        q1_next, q2_next = sac.evaluate_action(next_states[:, :-1, :], next_actions)
        q_next = torch.min(q1_next, q2_next)
        target_q = rewards[:, -1, :] + gamma * (1 - dones[:, -1, :]) * (q_next - alpha * sac.log_prob(next_states[:, -1, :], next_actions[:, -1, :]))

    q1, q2 = sac.evaluate_action(states[:, :-1, :], actions[:, :-1, :])

    critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actions_pred = sac(states[:, :-1, :])
    p1_pred, p2_pred = sac.evaluate_action(states[:, :-1, :], actions_pred)
    actor_loss = (alpha * sac.log_prob(states[:, -1, :], actions_pred[:, -1, :]) - torch.min(p1_pred, p2_pred)).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    actor_scheduler.step()
    critic_scheduler.step()

    return actor_loss.item(), critic_loss.item()


def train(
        sac: SoftActorCritic,
        train_buffer: object,
        batch_size:int=BATCH_SIZE,
        epochs:int=EPOCHS,
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
    actor_record, critic_record = 0, 0

    for epoch in tqdm(range(epochs)):
        if len(train_buffer.buffer) < batch_size:
            continue

        batch = train_buffer.sample(batch_size)
        batch = tuple(t.to(device) for t in batch)

        if epoch < warmup_epochs:
            # Warmup階段：只訓練Critic，凍結Actor
            for param in sac.actor.parameters():
                param.requires_grad = False
            for param in sac.critic_1.parameters():
                param.requires_grad = True
            for param in sac.critic_2.parameters():
                param.requires_grad = True
            
            # 只做Critic優化，Actor不更新，避免actor_optimizer step
            states, actions, rewards, next_states, dones = batch

            with torch.no_grad():
                next_actions = sac(next_states[:, :-1, :])
                q1_next, q2_next = sac.evaluate_action(next_states[:, :-1, :], next_actions)
                q_next = torch.min(q1_next, q2_next)
                target_q = rewards[:, -1, :] + gamma * (1 - dones[:, -1, :]) * (q_next - alpha * sac.log_prob(next_states[:, -1, :], next_actions[:, -1, :]))
            q1, q2 = sac.evaluate_action(states[:, :-1, :], actions[:, :-1, :])
            critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            critic_scheduler.step()# 更新Scheduler
            actor_loss = 0.0  # 無Actor更新

        else:
            # 正常訓練階段：Actor和Critic同時訓練
            for param in sac.actor.parameters():
                param.requires_grad = True
            # 執行完整train_sac_step
            actor_loss, critic_loss = train_sac_step(
                sac,
                actor_optimizer, critic_optimizer,
                actor_scheduler, critic_scheduler,
                loss_fn,
                batch, 
                gamma=gamma, alpha=alpha
            )
        
        # 檢查是否早停
        if actor_loss < actor_best_loss:
            torch.save(sac.encoder.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_ENCODER.pth')
            torch.save(sac.actor.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_ACTOR.pth')
            torch.save(sac.mu_layer.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_MU.pth')
            torch.save(sac.log_sigma_layer.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_SIGMA.pth')
            actor_best_loss = actor_loss
            actor_record = 0
        else:
            actor_record = actor_record + 1

        if critic_loss < critic_best_loss:
            torch.save(sac.encoder.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_ENCODER.pth')
            torch.save(sac.critic_1.reward.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_REWARD1.pth')
            torch.save(sac.critic_2.reward.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_REWARD2.pth')
            torch.save(sac.mu_layer.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_MU.pth')
            torch.save(sac.log_sigma_layer.state_dict(), f'model_record/DDPG/{timestamp}_{MODEL_TYPE}_SIGMA.pth')
            critic_best_loss = critic_loss
            critic_record = 0
        else:
            critic_record = critic_record + 1

        if actor_record >= early_stop and critic_record >= early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break

        # 紀錄Wandb
        wandb.log({
            "epoch": epoch,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "actor_lr": actor_scheduler.get_last_lr()[0],
            "critic_lr": critic_scheduler.get_last_lr()[0]
        })

    wandb.finish()




if __name__ == '__main__':
    
    train_csv_file = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train.csv'))
    test_csv_file = pd.read_csv(os.path.join(os.getcwd(), 'data', 'test.csv'))
    val_csv_file = pd.read_csv(os.path.join(os.getcwd(), 'data', 'val.csv'))


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

    replay_buffer = ReplayBuffer()
    state_scaler, action_scaler, reward_scaler = preprocess_data(train_csv_file, replay_buffer, state_cols, action_cols, reward_cols)

    sac = SoftActorCritic(state_dim, action_dim, reward_dim).to(DEVICE)
    sac.encoder.load_state_dict(torch.load(ENCODER_PATH)) if os.path.exists(ENCODER_PATH) else None
    sac.actor.load_state_dict(torch.load(ACTOR_PATH)) if os.path.exists(ACTOR_PATH) else None
    sac.critic_1.predict.load_state_dict(torch.load(ENVIRONMENT_PATH)) if os.path.exists(ENVIRONMENT_PATH) else None
    sac.critic_1.reward.load_state_dict(torch.load(CRITIC1_REWARD_PATH)) if os.path.exists(CRITIC1_REWARD_PATH) else None
    sac.critic_2.reward.load_state_dict(torch.load(CRITIC2_REWARD_PATH)) if os.path.exists(CRITIC2_REWARD_PATH) else None
    sac.mu_layer.load_state_dict(torch.load(MU_LAYER_PATH)) if os.path.exists(MU_LAYER_PATH) else None
    sac.log_sigma_layer.load_state_dict(torch.load(LOG_SIGMA_LAYER_PATH)) if os.path.exists(LOG_SIGMA_LAYER_PATH) else None

    train(sac, replay_buffer)

