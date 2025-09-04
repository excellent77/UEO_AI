import os
import time
from typing import Literal

import torch
import torch.nn as nn



#獲取 models.py 中定義的所有類別名稱
MODEL_LIST = ("LSTM_Model", "GRU_Model", "Transformer_Model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class LSTM_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64, dense_units=64, num_layers=2, dropout=0.2):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dense1 = nn.Linear(hidden_size, dense_units)
        self.bn1 = nn.BatchNorm1d(dense_units)          # 第一層BatchNorm

        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dense_units, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]            # 取最後時間步輸出

        out = self.dense1(out)
        out = self.bn1(out)                # BatchNorm
        out = self.activation(out)
        out = self.drop(out)

        out = self.output_layer(out)      # 輸出層無激活函數

        return out


class GRU_Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=64, dense_units=64, num_layers=2, dropout=0.2):
        super(GRU_Model, self).__init__()
        self.lstm = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dense1 = nn.Linear(hidden_size, dense_units)
        self.bn1 = nn.BatchNorm1d(dense_units)          # 第一層BatchNorm

        self.activation = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)
        self.output_layer = nn.Linear(dense_units, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]            # 取最後時間步輸出

        out = self.dense1(out)
        out = self.bn1(out)                # BatchNorm
        out = self.activation(out)
        out = self.drop(out)

        out = self.output_layer(out)      # 輸出層無激活函數

        return out
    

class Transformer_Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64, num_layers=2, nhead=4, dense_units=64, dropout=0.1):
        super(Transformer_Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 將輸入投影到 hidden_size 方便給 Transformer
        self.input_fc = nn.Linear(input_size, hidden_size)
        
        # 位置編碼 (可簡單用 learnable 或 sinusoidal)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        # Transformer Encoder
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
        
        # 輸出層
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, dense_units),
            nn.BatchNorm1d(dense_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, output_size)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = self.input_fc(x)                    # [batch, seq_len, hidden_size]
        x = self.pos_encoder(x)                 # 增加位置資訊
        
        out = self.transformer_encoder(x)       # [seq_len, batch, hidden_size]
        
        # 取最後一個時間步
        out = out[:, -1, :]                     # [batch, hidden_size]
        out = self.fc_out(out)                  # [batch, output_size]
        return out

class PositionalEncoding(nn.Module):
    # 經典 sine-cosine 位置編碼
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)
    def forward(self, x):
        # x: [seq_len, batch, dim]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

    


def build_model(
        model_name:Literal["LSTM_Model", "GRU_Model", "Transformer_Model"], # 模型名稱
        input_size:int, # 輸入特徵大小
        hidden_size:int, # 隱藏層大小
        num_layers:int, # LSTM/GRU 層數
        output_size:int, # 輸出特徵大小
        *args, # 其他模型特定參數，例如 Transformer 的 nhead, dropout 等
        **kwargs # 其他關鍵字參數
        )-> torch.nn.Module:
    """
    根據模型名稱創建相應的模型實例。
    
    Args:
        model_name (str): 模型名稱，必須是 'LSTM_Model', 'GRU_Model' 或 'Transformer_Model'。
        input_size (int): 輸入特徵的大小。
        hidden_size (int): 隱藏層的大小。
        num_layers (int): LSTM/GRU 的層數。
        output_size (int): 輸出特徵的大小。
        **kwargs: 其他模型特定參數，例如 Transformer 的 nhead, dropout 等。
    
    Returns:
        torch.nn.Module: 相應的模型實例。
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


def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_epochs:int,
        save_dir: str
    ):

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model.to(DEVICE)
    loss_history = []
    lr_history = []

    for epoch in range(num_epochs):

        model.train()
        total_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        lr_now = optimizer.param_groups[0]['lr']
        
        scheduler.step()
        
        loss_history.append(avg_loss)
        lr_history.append(lr_now)

        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/MODEL_{timestamp}.pth")
        
        yield loss_history, lr_history, (
            f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {lr_now:.6f}"
        )
        
def predict(
        model:torch.nn.Module,
        data_loader:torch.utils.data.DataLoader
    )-> torch.Tensor:
    '''
    使用訓練好的模型對數據進行預測。
    Args:
        model (torch.nn.Module): 訓練好的 LSTM 模型。
        data_loader (torch.utils.data.DataLoader): 用於預測的數據加載器。
    Returns:
        torch.Tensor: 預測結果。
    '''

    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            predictions.append(outputs)
            
    return torch.cat(predictions, dim=0)