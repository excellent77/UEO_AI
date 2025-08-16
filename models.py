import time
import torch
import torch.nn as nn
from typing import Literal



#獲取 models.py 中定義的所有類別名稱
MODEL_LIST = ("LSTM_Model", "GRU_Model", "Transformer_Model")



class LSTM_Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 層
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全連接層
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隱藏狀態和細胞狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM 前向傳播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最後一個時間步的輸出
        out = out[:, -1, :]
        
        # 全連接層輸出
        out = self.fc(out)
        
        return out


class GRU_Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU 層
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全連接層
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隱藏狀態 (GRU 沒有 cell state)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU 前向傳播
        out, _ = self.gru(x, h0)
        
        # 取最後一個時間步的輸出
        out = out[:, -1, :]
        
        # 全連接層輸出
        out = self.fc(out)
        
        return out
    

class Transformer_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, nhead=4, dropout=0.1):
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
            batch_first=False  # 要配合下面轉置
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 輸出層
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        x = self.input_fc(x)                    # [batch, seq_len, hidden_size]
        x = x.transpose(0,1)                    # [seq_len, batch, hidden_size] for transformer
        x = self.pos_encoder(x)                 # 增加位置資訊
        
        out = self.transformer_encoder(x)       # [seq_len, batch, hidden_size]
        
        # 取最後一個時間步
        out = out[-1, :, :]                     # [batch, hidden_size]
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
        return LSTM_Model(input_size, hidden_size, num_layers, output_size)
    elif model_name == 'GRU_Model':
        return GRU_Model(input_size, hidden_size, num_layers, output_size)
    elif model_name == 'Transformer_Model':
        return Transformer_Model(input_size, hidden_size, num_layers, output_size, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


import torch
import time
import os

def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        num_epochs: int,
        save_dir: str,
        early_stopping: int = 10
    ):
    '''
    訓練模型，並在每個 epoch 結束時返回損失和學習率歷史記錄。
    '''
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    
    best_loss = float('inf')
    record = 0
    loss_history = []
    lr_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            output = model(inputs.to(device))
            loss = criterion(output, targets.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        
        if scheduler is not None:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()
        
        lr_now = optimizer.param_groups[0]['lr']
        loss_history.append(avg_loss)
        lr_history.append(lr_now)

        # 儲存模型檔案，確保資料夾存在
        if avg_loss < best_loss:
            record = 0
            best_loss = avg_loss
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{save_dir}/MODEL_{timestamp}.pth")
        else:
            record += 1
            if record >= early_stopping:
                yield loss_history, lr_history, "Early stopping triggered."
                break
        
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