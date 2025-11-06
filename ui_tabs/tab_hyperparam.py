import os
import gradio as gr
import matplotlib.pyplot as plt
import torch

import ui_tabs.models as models
import ui_tabs.Preprocessing as Preprocessing
from utils import losses, optimizers, schedulers
from . import tab_shared

def create_hyperparam_tab(clean_features, clean_targets, scalers, output_clean):
    """
    創建並返回「超參數設定」分頁的 UI 元件和相關的事件處理。
    """
    with gr.Tab("超參數設定") as hyperparam_tab:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 設定 Model 超參數")
                DP_model_name = gr.Dropdown(choices=models.MODEL_LIST, label="選擇使用模型", value=models.MODEL_LIST[0])
                DP_pre_model = gr.Dropdown(choices=[], label="請選擇預訓練模型", value=None)
                NUM_epochs = gr.Number(label="訓練週期數 (Epochs)", value=100, precision=0)
                NUM_lr = gr.Slider(label="學習率 (Learning Rate)", minimum=1e-5, maximum=1e-3, step=1e-5, value=1e-4, interactive=True)
                DP_loss = gr.Dropdown(choices=losses.LOSS_LIST, label="Loss Function", value=losses.LOSS_LIST[0])
                DP_opt = gr.Dropdown(choices=optimizers.OPTIM_LIST, label="Optimizer", value=optimizers.OPTIM_LIST[0])
                DP_sch = gr.Dropdown(choices=schedulers.SCH_LIST, label="Scheduler", value=schedulers.SCH_LIST[0])
                BTN_train = gr.Button("開始訓練模型", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 2. 訓練進度")
                output_hyp = gr.Textbox(lines=5, label="訓練進度")
                loss_plot = gr.Plot(label="Loss 變化")
                r2_plot = gr.Plot(label="val_R square 變化")
                lr_plot = gr.Plot(label="Learning Rate 變化")

    def start_to_train(epochs, lr, loss, opt, sch, features, targets, model_name, pre_model_name, scalers_dict):
        """
        開始訓練模型，並返回訓練過程中的損失和學習率曲線。
        """
        if features is None or targets is None:
            raise gr.Error("請先完成資料清洗步驟！")

        save_dir = os.path.join(tab_shared.MODEL_DIR, model_name)
        train_loader = Preprocessing.process_to_dataloader(features['train'], targets['train'], batch_size=tab_shared.BATCH_SIZE)
        val_loader = Preprocessing.process_to_dataloader(features['val'], targets['val'], batch_size=tab_shared.BATCH_SIZE)

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

        if isinstance(pre_model_name, str) and pre_model_name:
            print("Loading pre-trained model...")
            model.load_state_dict(torch.load(os.path.join(save_dir, pre_model_name)))

        status_record = f'''---Training details---
Model: {model.__class__.__name__}
Loss function: {loss}
Optimizer: {opt}
Scheduler: {sch}
Learning Rate: {lr}\n\n'''
        
        loss_hist, val_loss_hist, r2_hist, lr_hist = [], [], [], []

        for avg_loss, val_loss, val_r2, lr_now, status in models.train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            criterion=loss_function, optimizer=optimizer, scheduler=scheduler,
            scaler=scalers_dict['target'], save_dir=save_dir, num_epochs=epochs,
            early_stopping=tab_shared.EARLY_STOPPING_PREDICT
        ):
            if avg_loss is None: # Early stopping
                status_record += status + '\n'
                yield loss_fig, r2_fig, lr_fig, status_record
                break

            loss_hist.append(avg_loss)
            val_loss_hist.append(val_loss)
            r2_hist.append(val_r2)
            lr_hist.append(lr_now)
            
            epochs_range = list(range(1, len(loss_hist) + 1))
            loss_fig = tab_shared.create_matplotlib_figure(
                datasets=[
                    {'x': epochs_range, 'y': loss_hist, 'label': 'Train Loss'},
                    {'x': epochs_range, 'y': val_loss_hist, 'label': 'Validation Loss'}
                ], title="Loss Curves"
            )
            r2_fig = tab_shared.create_matplotlib_figure(
                datasets=[{'x': epochs_range, 'y': r2_hist, 'label': 'Validation R2'}], title="Validation R2"
            )
            lr_fig = tab_shared.create_matplotlib_figure(
                datasets=[{'x': epochs_range, 'y': lr_hist, 'label': 'Learning Rate'}], title="LR Curve"
            )
            status_record += status + '\n'
            yield loss_fig, r2_fig, lr_fig, status_record
            plt.close(loss_fig)
            plt.close(r2_fig)
            plt.close(lr_fig)

    BTN_train.click(
        fn=start_to_train,
        inputs=[NUM_epochs, NUM_lr, DP_loss, DP_opt, DP_sch, clean_features, clean_targets, DP_model_name, DP_pre_model, scalers],
        outputs=[loss_plot, r2_plot, lr_plot, output_hyp]
    )

    # 當模型名稱改變時，更新預訓練模型列表
    DP_model_name.change(fn=tab_shared.get_model_files, inputs=DP_model_name, outputs=DP_pre_model)
    # 當進入此分頁時，也更新一次
    hyperparam_tab.select(fn=tab_shared.get_model_files, inputs=DP_model_name, outputs=DP_pre_model)
    # 當訓練完成後，也更新一次
    output_hyp.change(fn=tab_shared.get_model_files, inputs=DP_model_name, outputs=DP_pre_model)

    # 當資料清洗完成後，啟用訓練按鈕
    output_clean.change(fn=lambda: gr.update(interactive=True), outputs=[BTN_train])

    return hyperparam_tab, DP_model_name, DP_pre_model