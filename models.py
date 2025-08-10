import time
import torch
import torch.nn as nn



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
    


def train_model(
        model:torch.nn.Module, # LSTM 模型
        train_loader:torch.utils.data.DataLoader, # 訓練數據加載器
        criterion:torch.nn.Module, # 損失函數
        optimizer:torch.optim.Optimizer, # 優化器
        scheduler:torch.optim.lr_scheduler._LRScheduler, # 調度器
        num_epochs:int, # 訓練輪數
        early_stopping:int=10 # 早停輪數
    ):
    '''
    訓練 LSTM 模型，並在每個 epoch 結束時返回損失和學習率歷史記錄。
    
    Args:
        model (torch.nn.Module): LSTM 模型實例。
        train_loader (torch.utils.data.DataLoader): 訓練數據加載器。
        criterion (torch.nn.Module): 損失函數。
        optimizer (torch.optim.Optimizer): 優化器。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 學習率調度器。
        num_epochs (int): 訓練輪數。
        early_stopping (int): 早停輪數，默認為 10。
    Yields:
        tuple: 每個 epoch 的損失歷史和學習率歷史，以及當前狀態的字符串。
    '''

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model.train()
    best_loss = float('inf')
    record = 0

    # 用以記錄每個 epoch 的 loss 和 lr
    loss_history = []
    lr_history = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        if scheduler is not None:
            # ReduceLROnPlateau 類型需傳入 loss 作調整，其他直接調整
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']

        # 將本次 epoch loss 和 lr 累加記錄
        loss_history.append(avg_loss)
        lr_history.append(lr_now)

        if avg_loss < best_loss:
            record = 0
            best_loss = avg_loss
            torch.save(model.state_dict(), f"model_record/LSTM_{timestamp}.pth")
        else:
            record += 1
            if record >= early_stopping:
                yield loss_history, lr_history, "Early stopping triggered."
                break

        # yield 時同時返回歷史列表以及文字訊息，方便前端實時更新與繪圖
        yield loss_history, lr_history, f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - LR: {lr_now:.6f}"

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