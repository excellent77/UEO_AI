import os
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch

import models
import Preprocessing
from utils import losses, optimizers, schedulers



# 設備與資料特性對應表
MACHINE_FEATURES = {
    "空壓機": ["倒U型曲線"],
    "HVAC系統": ["依負載變動", "週期波動"],
    "冷卻塔": ["依負載變動", "倒U型曲線"],
    "馬達": ["倒U型", "脈衝波動"],
    "加熱爐": ["脈衝波動"],
    "鍋爐": ["依負載變動", "脈衝波動"],
    "泵浦": ["依負載變動", "週期波動"],
    "風機": ["週期波動"]
}

MACHINE_TYPES = list(MACHINE_FEATURES.keys())
MODEL_DIR = os.path.join(os.getcwd(), 'model_record')
SEQUENCE_LENGTH = 60
BATCH_SIZE = 128



def create_matplotlib_figure(
        datasets:list, # 包含多組數據的列表
        title:str # 圖形標題
    )-> plt.Figure:
    '''
    根據傳入的多組數據生成 matplotlib 圖形，並設置標題、標籤和圖例。
    參數:
        - datasets: list of dicts. 每個 dict 應包含 'x', 'y', 'label'。
          範例: [{'x': [1,2], 'y': [1,2], 'label': 'train'}, {'x': [1,2], 'y': [3,4], 'label': 'val'}]
        - title: 圖形標題
    返回:
        - matplotlib 圖形對象
    '''

    fig, ax = plt.subplots(figsize=(6,4))
    
    for data in datasets:
        ax.plot(data['x'], data['y'], marker='o', linestyle='-', label=data['label'])

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.grid(True)
    ax.legend()

    return fig



if __name__ == "__main__":

    for mode in models.MODEL_LIST:
        # 確保模型目錄存在
        os.makedirs(os.path.join(MODEL_DIR, mode), exist_ok=True)

    with gr.Blocks() as demo:
        gr.Markdown("## 耗能設備的通用性能源操作優化框架 ")

        with gr.Tabs(selected=0):
            
            with gr.Tab("上傳資料"):
                train_data = gr.File(label="上傳訓練資料集（CSV）", file_types=[".csv"])
                train_load = gr.Dataframe(label="訓練資料集預覽", interactive=False)
                val_data = gr.File(label="上傳驗證資料集（CSV）", file_types=[".csv"])
                val_load = gr.Dataframe(label="驗證資料集預覽", interactive=False)
            
            with gr.Tab("選擇欄位"):
                DP_machine = gr.Dropdown(choices=MACHINE_TYPES, label="請選擇設備類別")
                RD_datetime = gr.Radio(choices=[], label="請選擇「時間戳記」的欄位")
                CBG_feature = gr.CheckboxGroup(choices=[], label="請選擇要用於「判斷」的欄位")
                CBG_target = gr.CheckboxGroup(choices=[], label="請選擇要用於「預測」的欄位")

                BTN_select = gr.Button("確認選擇", interactive=False)
                output_select = gr.Textbox(label="選擇結果", interactive=False)
                

                def show_selection(
                    machine:str, # 選擇的設備類型
                    datetime_col:str, # 時間欄位名稱
                    train_cols:list, # 用於判斷的特徵欄位
                    label_cols:list  # 用於預測的標籤欄位
                )->str:
                    """
                    組裝及回傳當前欄位及設備的使用者選擇摘要，於 Gradio Textbox 顯示。
                    """
                    result = f"已選設備：{machine}\n"
                    result += f"時間欄位：{datetime_col}\n"
                    result += f"用於判斷欄位：{train_cols}\n"
                    result += f"用於預測欄位：{label_cols}\n"
                    return result
                
                BTN_select.click(
                    fn=show_selection,
                    inputs=[
                        DP_machine,
                        RD_datetime,
                        CBG_feature,
                        CBG_target
                    ],
                    outputs=output_select
                )

            with gr.Tab("資料清洗"):
                DP_fill = gr.Dropdown(choices=Preprocessing.FILL_STRATEGIES, label="請選擇缺失值填補策略")
                DP_scale = gr.Dropdown(choices=Preprocessing.SCALE_METHODS, label="請選擇正規化方式")

                BTN_clean = gr.Button("確認選擇", interactive=False)
                output_clean = gr.Textbox(label="資料 shape", interactive=False)
                clean_features = gr.State()
                clean_targets = gr.State()
                scalers = gr.State()


                def preprocess_and_export(
                    train_df:pd.DataFrame, # 原始資料
                    val_df:pd.DataFrame, # 驗證資料
                    datetime_col:str, # 時間欄位名稱
                    feature_cols:list, # 特徵(輸入)欄位清單
                    target_cols:list, # 標籤(預測目標)欄位清單
                    fill_strategy:str, # 缺失值填補策略
                    scale_method:str # 特徵正規化方式
                )->list:
                    """
                    調用自訂 Preprocessing 模組的預處理流程，產生 LSTM 可用的特徵與標籤及標準化器。
                    傳回:
                        [資料形狀資訊, X特徵, y標籤, 標準化器物件]
                    """
                    # 進行資料清洗與轉換
                    train_feature, train_target, x_scaler, y_scaler = Preprocessing.preprocess_for_lstm(
                        train_df,
                        datetime_col=datetime_col, feature_cols=feature_cols, target_cols=target_cols,
                        fill_strategy=fill_strategy, scale_method=scale_method,
                        sequence_length=SEQUENCE_LENGTH
                    )

                    val_feature, val_target, _, _ = Preprocessing.preprocess_for_lstm(
                        val_df,
                        datetime_col=datetime_col, feature_cols=feature_cols, target_cols=target_cols,
                        fill_strategy=fill_strategy, scale_method=scale_method,
                        sequence_length=SEQUENCE_LENGTH,
                        apply_scaler={
                            "feature": x_scaler,
                            "target": y_scaler
                        }  # 使用訓練集的標準化器
                    )

                    shape_str = f'''
                    train特徵 shape: {train_feature.shape}; train標籤 shape: {train_target.shape}
                    val特徵 shape: {val_feature.shape}; val標籤 shape: {val_target.shape}'''

                    return [
                        shape_str,
                        {"train" : train_feature, "val" : val_feature},
                        {"train" : train_target, "val" : val_target},
                        {"feature" : x_scaler, "target" : y_scaler}
                    ]

                BTN_clean.click(
                    fn=preprocess_and_export,
                    inputs=[
                        train_data,
                        val_data,
                        RD_datetime,
                        CBG_feature,
                        CBG_target,
                        DP_fill,
                        DP_scale
                    ],
                    outputs=[
                        output_clean,
                        clean_features,
                        clean_targets,
                        scalers
                    ]
                )


            with gr.Tab("超參數設定"):
                DP_model_name = gr.Dropdown(choices=models.MODEL_LIST, label="選擇使用模型")
                DP_pre_model = gr.Dropdown(choices=os.listdir(f"{MODEL_DIR}/{models.MODEL_LIST[0]}")+[None], label="請選擇預訓練模型", value=None)
                NUM_epochs = gr.Number(label="訓練週期數 (Epochs)", value=100, precision=0)
                NUM_lr = gr.Slider(label="學習率 (Learning Rate)", minimum=1e-5, maximum=1e-3, step=1e-5, value=1e-4, interactive=True)
                DP_loss = gr.Dropdown(choices=losses.LOSS_LIST, label="Loss Function", value=losses.LOSS_LIST[0])
                DP_opt = gr.Dropdown(choices=optimizers.OPTIM_LIST, label="Optimizer", value=optimizers.OPTIM_LIST[0])
                DP_sch = gr.Dropdown(choices=schedulers.SCH_LIST, label="Scheduler", value=schedulers.SCH_LIST[0])

                BTN_train = gr.Button("開始訓練模型", interactive=False)
                output_hyp = gr.Textbox(lines=5, label="訓練進度")
                loss_plot = gr.Plot(label="Loss 變化")
                r2_plot = gr.Plot(label="val_R square 變化")
                lr_plot = gr.Plot(label="Learning Rate 變化")


                def start_to_train(
                    epochs:int, # 訓練週期數
                    lr:int, # 學習率
                    loss:str, # 損失函數
                    opt:str, # 優化器
                    sch:str, # 調度器
                    features, # 特徵資料
                    targets, # 標籤資料
                    model_name:str, # 模型名稱
                    pre_model_name:str, # 預訓練模型路徑 (可選)
                    scalers_dict:dict # 從 gr.State 傳入的 scaler 字典
                ):
                    """
                    開始訓練模型，並返回訓練過程中的損失和學習率曲線。
                    參數:
                        - epochs: 訓練週期數
                        - lr: 學習率
                        - loss: 損失函數名稱
                        - opt: 優化器名稱
                        - sch: 調度器名稱
                        - feature: 特徵資料
                        - labels: 標籤資料
                        - model_name: 模型名稱
                        - pre_model_name: 預訓練模型路徑 (可選)
                        - scalers_dict: 包含 'feature' 和 'target' scaler 的字典
                    返回:
                        - loss_fig: 損失曲線圖形
                        - lr_fig: 學習率曲線圖形
                        - status_record: 訓練狀態記錄
                    """
                    save_dir = os.path.join(MODEL_DIR, model_name)

                    train_loader = Preprocessing.process_to_dataloader(
                        features['train'],
                        targets['train'],
                        batch_size=BATCH_SIZE
                    )

                    val_loader = Preprocessing.process_to_dataloader(
                        features['val'],
                        targets['val'],
                        batch_size=BATCH_SIZE
                    )

                    model = models.build_model(
                        model_name=model_name,
                        input_size=int(features['train'].shape[-1]),
                        hidden_size=64,
                        num_layers=2,
                        output_size=int(targets['train'].shape[-1])
                    )
                    
                    loss_function = losses.build_loss(loss)
                    optimizer = optimizers.build_optimizer(opt, model, lr)
                    scheduler = schedulers.build_scheduler(sch, optimizer)
                    

                    if type(pre_model_name) == str and pre_model_name:
                        print("Loading pre-trained model...")
                        model.load_state_dict(torch.load(os.path.join(save_dir, pre_model_name)))

                    status_record = f'''
                    ---Training details---
                    Model: {model.__class__.__name__}
                    Loss function: {loss}
                    Optimizer: {opt}
                    Scheduler: {sch}
                    Learning Rate: {lr}\n\n'''
                    
                    loss_hist = []
                    val_loss_hist = []
                    r2_hist = []
                    lr_hist = []

                    for avg_loss, val_loss, val_r2, lr_now, status in models.train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        criterion=loss_function,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scalers_dict['target'],
                        save_dir=save_dir,
                        num_epochs=epochs,
                        early_stopping=10
                    ):
                        loss_hist.append(avg_loss)
                        val_loss_hist.append(val_loss)
                        r2_hist.append(val_r2)
                        lr_hist.append(lr_now)
                        # 將 loss 和 lr 歷史記錄轉換為 matplotlib 圖形
                        epochs_range = list(range(1, len(loss_hist) + 1))
                        loss_fig = create_matplotlib_figure(
                            datasets=[
                                {'x': epochs_range, 'y': loss_hist, 'label': 'Train Loss'},
                                {'x': epochs_range, 'y': val_loss_hist, 'label': 'Validation Loss'}
                            ],
                            title="Loss Curves"
                        )

                        r2_fig = create_matplotlib_figure(
                            datasets=[
                                {'x': epochs_range, 'y': r2_hist, 'label': 'Validation R2'}
                            ],
                            title="Validation R2"
                        )

                        lr_fig = create_matplotlib_figure(
                            datasets=[
                                {'x': epochs_range, 'y': lr_hist, 'label': 'learning rate'}
                            ],
                            title="LR curve"
                        )
                        status_record += status+'\n'
                        yield loss_fig, r2_fig, lr_fig, status_record
                        plt.close(loss_fig)
                        plt.close(r2_fig)
                        plt.close(lr_fig)


                BTN_train.click(
                    fn=start_to_train,
                    inputs=[
                        NUM_epochs,
                        NUM_lr,
                        DP_loss,
                        DP_opt,
                        DP_sch,
                        clean_features,
                        clean_targets,
                        DP_model_name,
                        DP_pre_model,
                        scalers
                    ],
                    outputs=[loss_plot, r2_plot, lr_plot, output_hyp]
                )
                        
            with gr.Tab("規劃求解器"):
                gr.Markdown("正在開發中...")



        def update_columns(file)->list:
            """
            依據上傳的 CSV 檔案，讀取欄位並更新前端選單選項，若無檔案則重置欄位。
            傳回:
                - 時間欄位單選選單 (Radio)
                - 判斷欄位多選 (CheckboxGroup)
                - 預測欄位多選 (CheckboxGroup)
                - CSV 前五列資料預覽 (DataFrame)
            """
            if file is None:
                # 無檔案時, 回傳空選項及無預覽
                return [
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    gr.update(choices=[], value=[]),
                    gr.update(interactive=False), 
                    None
                ]
            
            df = pd.read_csv(file.name, encoding="utf-8")
            cols = df.columns.tolist()
            # CheckboxGroup 用 gr.update，Dataframe 傳資料
            return [
                gr.update(choices=cols, value=[]),
                gr.update(choices=cols, value=[]),
                gr.update(choices=cols, value=[]),
                gr.update(interactive=True),
                df.head()
            ]
        
        train_data.change(
            fn=update_columns,
            inputs=train_data,
            outputs=[RD_datetime, CBG_feature, CBG_target, BTN_select, train_load]
        )

        val_data.change(
            fn=lambda file: pd.read_csv(file.name, encoding="utf-8").head(),
            inputs=val_data,
            outputs=[val_load]
        )



        def get_model_files(model_name):
            save_dir = os.path.join(MODEL_DIR, model_name)
            files = [None] + os.listdir(save_dir)
            return gr.update(choices=files, value=None)
                
        DP_model_name.change(
            fn=get_model_files,
            inputs=DP_model_name,
            outputs=DP_pre_model
        )
        

        output_select.change(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[BTN_clean]
        )


        output_clean.change(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[BTN_train]
        )


        output_hyp.change(
            fn=lambda model_name: gr.update(choices=os.listdir(os.path.join(MODEL_DIR, model_name))+[None], value=None),
            inputs=[DP_model_name],
            outputs=[DP_pre_model]
        )

    print("Starting Gradio demo...")
    demo.launch()
