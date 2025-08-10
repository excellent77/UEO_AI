import os
import gradio as gr
import pandas as pd
import Preprocessing
import torch
import models
from utils import losses, optimizers, schedulers
import matplotlib.pyplot as plt



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
CONFIG = {
    "warmup_scheduler": {
        "warmup_epochs": 5
    },
    "train_scheduler": {
        "scheduler_type": "reduceonplateau",
        "mode": "min",
        "patience": 10,
        "scheduler_args": {
            "min_lr": 1e-6,
            "t_0_epochs": 10,
            "t_mult": 1
        }
    }
}
MODEL_DIR = os.path.join(os.getcwd(), 'model_record')



def create_matplotlib_figure(
        x,
        y,
        title
    )-> plt.Figure:
    '''
    根據 x 和 y 數據生成 matplotlib 圖形，並設置標題和標籤。
    參數:
        - x: x 軸數據
        - y: y 軸數據
        - title: 圖形標題
    返回:
        - matplotlib 圖形對象
    '''

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(x, y, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title)
    ax.grid(True)

    return fig

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


def preprocess_and_export(
        df:pd.DataFrame, # 原始資料
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
    X, y, scaler = Preprocessing.preprocess_for_lstm(
        df, datetime_col, feature_cols, target_cols, fill_strategy, scale_method
    )
    shape_str = f"特徵 shape: {X.shape}; 標籤 shape: {y.shape}"
    return [shape_str, X, y, scaler]

def start_to_train(
        epochs:int,
        lr:int,
        loss:str,
        opt:str,
        sch:str,
        feature,
        labels,
        model_path:str=None
    ):
    loss_function = losses.build_loss(loss)
    dataloader = Preprocessing.process_to_dataloader(feature, labels)

    model = models.LSTM_Model(
        input_size=int(feature.shape[-1]),
        hidden_size=64,
        num_layers=2,
        output_size=int(labels.shape[-1])
    )
    
    optimizer = optimizers.build_optimizer(opt, model, lr)
    scheduler = schedulers.build_scheduler(sch, optimizer, CONFIG)

    if model_path is not None:
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, model_path)))

    status_record = f'''
    ---Training details---
    Model: {model.__class__.__name__}
    Loss function: {loss}
    Optimizer: {opt}
    Scheduler: {sch}
    Learning Rate: {lr}\n\n'''
    
    for loss_hist, lr_hist, status in models.train_model(
        model=model,
        train_loader=dataloader,
        criterion=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=MODEL_DIR,
        num_epochs=epochs
    ):
        # 將 loss 和 lr 歷史記錄轉換為 matplotlib 圖形
        loss_fig = create_matplotlib_figure(
            list(range(1, len(loss_hist)+1)),
            loss_hist,
            "Loss curve"
        )

        lr_fig = create_matplotlib_figure(
            list(range(1, len(lr_hist)+1)),
            lr_hist,
            "LR curve"
        )
        status_record += status+'\n'
        yield loss_fig, lr_fig, status_record
        plt.close(loss_fig)
        plt.close(lr_fig)



if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    with gr.Blocks() as demo:
        gr.Markdown("## 耗能設備的通用性能源操作優化框架 ")

        with gr.Tabs(selected=0):
            
            with gr.Tab("上傳資料"):
                csv_file = gr.File(label="上傳感測器資料（CSV）", file_types=[".csv"])
                preview = gr.Dataframe(label="資料預覽", interactive=False)
            
            with gr.Tab("選擇欄位"):
                machine_dropdown = gr.Dropdown(choices=MACHINE_TYPES, label="請選擇設備類別")
                datetime_col = gr.Radio(choices=[], label="請選擇「時間戳記」的欄位")
                feature_cols = gr.CheckboxGroup(choices=[], label="請選擇要用於「判斷」的欄位")
                target_cols = gr.CheckboxGroup(choices=[], label="請選擇要用於「預測」的欄位")
                output_text = gr.Textbox(label="選擇結果", interactive=False)
                btn_select = gr.Button("確認選擇", interactive=False)
                btn_select.click(
                    fn=show_selection,
                    inputs=[
                        machine_dropdown,
                        datetime_col,
                        feature_cols,
                        target_cols
                    ],
                    outputs=output_text
                )

            with gr.Tab("資料清洗"):
                fill_dropdown = gr.Dropdown(choices=Preprocessing.FILL_STRATEGIES, label="請選擇缺失值填補策略")
                scale_dropdown = gr.Dropdown(choices=Preprocessing.SCALE_METHODS, label="請選擇正規化方式")
                btn_clean = gr.Button("確認選擇", interactive=False)
                shape_info = gr.Textbox(label="資料 shape", interactive=False)
                clean_feature = gr.Numpy(label="清洗後特徵資料預覽", interactive=False)
                clean_labels = gr.Numpy(label="清洗後特徵資料預覽", interactive=False)

                btn_clean.click(
                    fn=preprocess_and_export,
                    inputs=[
                        csv_file,
                        datetime_col,
                        feature_cols,
                        target_cols,
                        fill_dropdown,
                        scale_dropdown
                    ],
                    outputs=[
                        shape_info,
                        clean_feature,
                        clean_labels,
                        gr.State()  # 用於保存 scaler 狀態
                    ]
                )
            
            with gr.Tab("超參數設定"):
                pre_model_path = gr.Dropdown(choices=os.listdir(MODEL_DIR), label="請選擇預訓練模型", value=None)
                epochs_input = gr.Number(label="訓練週期數 (Epochs)", value=50, precision=0)
                lr_input = gr.Slider(label="學習率 (Learning Rate)", minimum=1e-4, maximum=1e-3, step=1e-5, value=1e-3, interactive=True)
                loss_dropdown = gr.Dropdown(choices=losses.LOSS_LIST, label="Loss Function", value=losses.LOSS_LIST[0])
                opt_dropdown = gr.Dropdown(choices=optimizers.OPTIM_LIST, label="Optimizer", value=optimizers.OPTIM_LIST[0])
                sch_dropdown = gr.Dropdown(choices=schedulers.SCH_LIST, label="Scheduler", value=schedulers.SCH_LIST[0])

                btn_train = gr.Button("開始訓練模型", interactive=False)
                out_text = gr.Textbox(lines=5, label="訓練進度")
                loss_plot = gr.Plot(label="Loss 變化")
                lr_plot = gr.Plot(label="Learning Rate 變化")

                btn_train.click(
                    fn=start_to_train,
                    inputs=[
                        epochs_input,
                        lr_input,
                        loss_dropdown,
                        opt_dropdown,
                        sch_dropdown,
                        clean_feature,
                        clean_labels,
                        pre_model_path
                    ],
                    outputs=[loss_plot, lr_plot, out_text]
                )
                        
            with gr.Tab("規劃求解器"):
                gr.Markdown("正在開發中...")


        csv_file.change(
            fn=update_columns,
            inputs=csv_file,
            outputs=[datetime_col, feature_cols, target_cols, btn_select, preview]
        )

        output_text.change(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[btn_clean]
        )

        shape_info.change(
            fn=lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[btn_train]
        )

    print("Starting Gradio demo...")
    demo.launch()
