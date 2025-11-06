import os
import gradio as gr
import pandas as pd
import numpy as np
import torch
import tqdm
import datetime # 新增：導入 datetime 模組
import matplotlib.pyplot as plt

import ui_tabs.models as models
import ui_tabs.Preprocessing as Preprocessing
from . import tab_shared

def create_inference_tab(train_data, solver_state_cols, solver_action_cols, state_scaler_obj):
    """
    創建並返回「執行推論」分頁的 UI 元件和相關的事件處理。
    """
    with gr.Tab("執行推論") as inference_tab:
        gr.Markdown("### 使用已訓練的 SAC Actor 模型或 Golden Sample 進行推論")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### 1. 選擇模型與資料")
                DP_inference_method = gr.Dropdown(choices=["SAC Actor", "Golden Sample"], label="選擇推論方法", value="SAC Actor")
                DP_actor_model = gr.Dropdown(choices=[], label="選擇 Actor 模型檔案", visible=True)
                inference_data = gr.File(label="上傳推論資料集 (CSV)", file_types=[".csv"])
                BTN_run_inference = gr.Button("生成最佳動作", interactive=True)

            with gr.Column(scale=2):
                gr.Markdown("#### 2. 推論結果")
                inference_output_df = gr.DataFrame(label="生成的動作")
                inference_output_plot = gr.Plot(label="動作變化圖")
                inference_output_csv = gr.File(label="下載生成的動作 (CSV)") # 新增：用於下載 CSV 檔案的 Gradio 元件

    def run_inference(method, actor_model_file, inference_file, historical_train_file, state_cols, action_cols, state_scaler):
        if not all([inference_file, state_cols, action_cols]):
            raise gr.Error("請確保已在先前頁籤完成所有設定，並在此處上傳推論資料。")
        
        if method == "SAC Actor" and not actor_model_file:
            raise gr.Error("使用 SAC Actor 方法時，必須選擇一個模型檔案。")
        
        if method == "Golden Sample" and not historical_train_file:
            raise gr.Error("使用 Golden Sample 方法時，必須已上傳訓練資料集。")

        df = pd.read_csv(inference_file.name)
        df_cleaned = Preprocessing.clean_data(df, datetime_col=None)
        df_cleaned = Preprocessing.remove_outliers_iqr(df_cleaned)
        df_cleaned = Preprocessing.fill_missing(df_cleaned, strategy='mean')

        if not all(col in df_cleaned.columns for col in state_cols):
            missing = [col for col in state_cols if col not in df_cleaned.columns]
            raise gr.Error(f"推論資料缺少必要的狀態欄位: {missing}")

        generated_actions = []

        if method == "SAC Actor":
            if state_scaler is None:
                raise gr.Error("找不到用於標準化狀態的 Scaler，請先執行求解器分頁的'確認欄位'步驟。")
            scaled_states_df, _ = Preprocessing.scale_features(df_cleaned[state_cols], scaler=state_scaler)
            inference_states = scaled_states_df.values

            actor_model_path = os.path.join(tab_shared.SOLVER_MODEL_DIR, actor_model_file)
            sac_model = models.SoftActorCritic(state_dim=len(state_cols), action_dim=len(action_cols))
            try:
                state_dict = torch.load(actor_model_path, map_location=models.DEVICE)
                sac_model.actor.load_state_dict(state_dict['actor'])
                sac_model.encoder.load_state_dict(state_dict['encoder'])
            except KeyError:
                print("警告：模型檔案不含 'encoder' 權重，僅載入 'actor'。")
                sac_model.actor.load_state_dict(torch.load(actor_model_path, map_location=models.DEVICE))

            for state in tqdm.tqdm(inference_states, desc="Running SAC Inference"):
                action = models.sac_get_action(sac_model, current_state=state)
                generated_actions.append(action)

        elif method == "Golden Sample":
            historical_df = pd.read_csv(historical_train_file.name)
            golden_expert = models.Golden_Sample(df=historical_df, state_cols=state_cols, action_cols=action_cols)
            inference_states = df_cleaned[state_cols].values

            for state in tqdm.tqdm(inference_states, desc="Running Golden Sample Inference"):
                action = models.golden_sample_get_action(golden_expert, current_state=state)
                generated_actions.append(action)

        actions_df = pd.DataFrame(np.array(generated_actions), columns=action_cols)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"inference_actions_{timestamp}.csv"
        output_filepath = os.path.join(tab_shared.OUTPUT_DIR, filename)
        actions_df.to_csv(output_filepath, index=False)

        fig, axes = plt.subplots(len(action_cols), 1, figsize=(8, 2 * len(action_cols)), sharex=True)
        if len(action_cols) == 1:
            axes = [axes] # make it iterable
        for i, col in enumerate(action_cols):
            axes[i].plot(actions_df.index, actions_df[col])
            axes[i].set_ylabel(col)
        axes[-1].set_xlabel("Time Step")
        fig.suptitle("Generated Actions Over Time")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        return actions_df, fig, output_filepath # 返回 DataFrame、圖表和 CSV 檔案路徑

    BTN_run_inference.click(
        fn=run_inference, 
        inputs=[DP_inference_method, DP_actor_model, inference_data, train_data, solver_state_cols, solver_action_cols, state_scaler_obj], 
        outputs=[inference_output_df, inference_output_plot, inference_output_csv] # 更新輸出，包含 CSV 檔案
    )

    DP_inference_method.change(
        fn=lambda method: gr.update(visible=method == "SAC Actor"),
        inputs=DP_inference_method, outputs=DP_actor_model
    )

    def update_actor_model_list():
        """讀取求解器模型目錄並更新 Actor 模型列表"""
        return gr.update(choices=tab_shared.get_solver_model_files())

    inference_tab.select(fn=update_actor_model_list, outputs=DP_actor_model)

    return inference_tab