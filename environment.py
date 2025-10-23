import time
import wandb
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import Preprocessing
from models import LSTM_Model


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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
EPOCHS = 100
WARMUP_EPOCHS = 30  # 預熱階段的epoch數
BATCH_SIZE = 128
SEQUENCE_LENGTH = 60
PRE_TRAINED_MODEL = r"/mnt/c/UEO_AI/model_record/environment/model_LSTM_20250921-180228.pth"  # 預訓練模型路徑，如果有的話
MODEL_TYPE = "LSTM"
KEY_WORDS = ['Kvar_tot', 'KW_tot', 'PF_avg']



def train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        scaler,
        main_scheduler,
        early_stopping=10
    ):

    timestamp = time.strftime('%Y%m%d-%H%M%S')
    wandb.init(
        project="UEO_AI environment",
        name=f"Training_{MODEL_TYPE}_{timestamp}",
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "warmup_epochs": WARMUP_EPOCHS,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQUENCE_LENGTH,
            "device": DEVICE
        }
    )

    for epoch in tqdm(range(EPOCHS), desc="Training Epochs"):

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

        # 訓練一輪後做：驗證集評價
        val_loss = evaluate_model(model, val_loader, criterion)
        val_r2 = evaluate_r2(model, val_loader, scaler)

        # 檢查是否早停
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'model_record/environment/model_{MODEL_TYPE}_{timestamp}.pth')
            record = 0
        else:
            record += 1
            if record >= early_stopping:
                print(f'Early stopping at epoch {epoch+1}')
                break


        # === 調整學習率 ===
        if epoch < WARMUP_EPOCHS:
            scheduler.step()
        else:
            main_scheduler.step(val_loss)
        

        wandb.log({
            "epoch": epoch + 1,
            "loss": total_loss / len(train_loader),
            "val_loss": val_loss,
            "val_r2":val_r2,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

    wandb.finish()


def predict(model, data_loader, scaler):
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


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss/len(data_loader)

def evaluate_r2(model, data_loader, scaler):
    predictions, real_values = predict(model, data_loader, scaler)
    return metrics.r2_score(real_values, predictions)



if __name__ == "__main__":
    os.makedirs('model_record/environment', exist_ok=True)


    ## 資料處理 ##
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


    # print("predict targets: ", state_cols+action_cols)
    train_feature, train_target, x_scaler, y_scaler = Preprocessing.preprocess_for_lstm(
        train_csv_file,
        datetime_col='DateTime',
        feature_cols=state_cols+action_cols,
        target_cols=reward_cols,
        fill_strategy='mean',
        scale_method='minmax',
        sequence_length=SEQUENCE_LENGTH
    )

    val_feature, val_target, _, _ = Preprocessing.preprocess_for_lstm(
        val_csv_file,
        datetime_col='DateTime',
        feature_cols=state_cols+action_cols,
        target_cols=reward_cols,
        fill_strategy='mean',
        scale_method='minmax',
        sequence_length=SEQUENCE_LENGTH,
        apply_scaler={
            "feature":x_scaler,
            "target":y_scaler
        }  # 使用訓練集的標準化器
    )

    test_feature, test_target, _, _ = Preprocessing.preprocess_for_lstm(
        test_csv_file,
        datetime_col='DateTime',
        feature_cols=state_cols+action_cols,
        target_cols=reward_cols,
        fill_strategy='mean',
        scale_method='minmax',
        sequence_length=SEQUENCE_LENGTH,
        apply_scaler={
            "feature":x_scaler,
            "target":y_scaler
        }  # 使用訓練集的標準化器
    )


    train_loader = Preprocessing.process_to_dataloader(
        train_feature,
        train_target,
        batch_size=BATCH_SIZE
    )

    val_loader = Preprocessing.process_to_dataloader(
        val_feature,
        val_target,
        batch_size=BATCH_SIZE
    )
    
    test_loader = Preprocessing.process_to_dataloader(
        test_feature,
        test_target,
        batch_size=BATCH_SIZE
    )
    print("train data shape: ", train_feature.shape, train_target.shape)
    print("test data shape: ", test_feature.shape, test_target.shape)
    print("val data shape: ", val_feature.shape, val_target.shape)

    

    ## 模型訓練 ##
    model = LSTM_Model(
        input_size=len(state_cols+action_cols),
        output_size=len(reward_cols)
    ).to(DEVICE)

    model.load_state_dict(torch.load(PRE_TRAINED_MODEL)) if os.path.exists(PRE_TRAINED_MODEL) else None

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    ## Warm_up Block ##
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,     # 從 lr * 1e-3 開始
        end_factor=1.0,        # 線性增加到 lr * 1.0（即 LEARNING_RATE）
        total_iters=WARMUP_EPOCHS
    )

    # Main Scheduler (ReduceLROnPlateau)
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, 
        #verbose=True
    )

    # Combine with SequentialLR (但要包一層 LambdaLR 來轉接 ReduceLROnPlateau)
    scheduler = torch.optim.lr_scheduler.SequentialLR (
        optimizer,
        schedulers=[warmup_scheduler, torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)],
        milestones=[WARMUP_EPOCHS]
    )
    
    '''train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        main_scheduler=main_scheduler,
        scaler=y_scaler,
        early_stopping=10
    )
    
    ## 模型評估 ##
    print("train loss: ", evaluate_model(model, train_loader, criterion))
    print("train r2", evaluate_r2(model, train_loader, y_scaler))

    print("test loss: ", evaluate_model(model, test_loader, criterion))
    print("test r2", evaluate_r2(model, test_loader, y_scaler))

    print("val loss: ", evaluate_model(model, val_loader, criterion))
    print("val r2", evaluate_r2(model, val_loader, y_scaler))

    '''
    predicts = predict(model, val_loader, y_scaler)
    import matplotlib.pyplot as plt
    length = 20
    idx = 1
    print(reward_cols)
    plt.plot(range(len(predicts[0][:length,idx])), predicts[0][:length,idx], label='predict', linestyle='--')
    plt.plot(range(len(predicts[1][:length,idx])), predicts[1][:length,idx], label='real', linestyle='--')
    plt.legend()
    plt.show()