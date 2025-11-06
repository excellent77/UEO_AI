import os
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import ui_tabs.models as models
import ui_tabs.Preprocessing as Preprocessing
from . import tab_shared

def create_solver_tab(train_data, val_data, CBG_act, CBG_reward, DP_fill, DP_scale):
    """
    創建並返回「求解器」分頁的 UI 元件和相關的事件處理。
    """
    with gr.Tab("求解器") as solver_tab:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. 選擇環境與欄位")
                DP_env_model_name = gr.Dropdown(choices=list(models.MODEL_LIST), label="選擇環境模型類別", value=models.MODEL_LIST[0])
                DP_env_model_file = gr.Dropdown(choices=[], label="選擇已訓練的環境模型檔案")
                BTN_solver_cols = gr.Button("確認欄位")

                gr.Markdown("### 2. 設定 Model 超參數")
                NUM_solver_epochs = gr.Number(label="訓練週期數 (Epochs)", value=100, precision=0)
                NUM_solver_steps = gr.Number(label="每週期最大步數 (Steps)", value=1000, precision=0)
                NUM_warmup_steps = gr.Number(label="預熱步數 (Warmup Steps)", value=10000, precision=0)
                NUM_solver_lr_actor = gr.Slider(label="Actor 學習率", minimum=1e-5, maximum=1e-3, step=1e-5, value=3e-4)
                NUM_solver_lr_critic = gr.Slider(label="Critic 學習率", minimum=1e-5, maximum=1e-3, step=1e-5, value=3e-4)
                NUM_solver_gamma = gr.Slider(label="折扣因子 (Gamma)", minimum=0.9, maximum=0.999, step=0.001, value=0.99)
                NUM_solver_alpha = gr.Slider(label="溫度係數 (Alpha)", minimum=0.0, maximum=1.0, step=0.05, value=0.2)
                BTN_solver_train = gr.Button("開始訓練求解器", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 3. 訓練進度")
                output_solver_hyp = gr.Textbox(lines=10, label="訓練日誌", interactive=False)
                solver_loss_plot = gr.Plot(label="Solver Loss 變化")

        # 求解器相關的狀態儲存
        solver_state_cols = gr.State()
        solver_action_cols = gr.State()
        solver_reward_cols = gr.State()
        initial_states_np = gr.State()
        state_scaler_obj = gr.State()

    def set_solver_columns(action_cols, reward_cols, train_df_file, val_df_file, fill_strategy, scale_method):
        """設定求解器所需欄位，並準備初始狀態數據"""
        if train_df_file is None or val_df_file is None:
            raise gr.Error("請先上傳訓練與驗證資料集！")

        train_df = pd.read_csv(train_df_file.name)
        val_df = pd.read_csv(val_df_file.name)
        
        feature_cols = action_cols + reward_cols
        state_cols = sorted(list(set(feature_cols) - set(action_cols)))
        
        full_df = pd.concat([train_df, val_df], ignore_index=True)
        cleaned_df = Preprocessing.clean_data(full_df, datetime_col=None)
        cleaned_df = Preprocessing.remove_outliers_iqr(cleaned_df)
        cleaned_df = Preprocessing.fill_missing(cleaned_df, strategy=fill_strategy)
        
        initial_states_df, state_scaler = Preprocessing.scale_features(cleaned_df[state_cols], method=scale_method)
        
        return state_cols, action_cols, reward_cols, initial_states_df.values.astype(np.float32), state_scaler

    BTN_solver_cols.click(
        fn=set_solver_columns,
        inputs=[CBG_act, CBG_reward, train_data, val_data, DP_fill, DP_scale],
        outputs=[solver_state_cols, solver_action_cols, solver_reward_cols, initial_states_np, state_scaler_obj]
    )

    def start_solver_train(env_model_name, env_model_file, state_cols, action_cols, reward_cols, initial_states, epochs, steps, warmup_steps, lr_actor, lr_critic, gamma, alpha):
        if not all([env_model_file, state_cols, action_cols, reward_cols, initial_states is not None]):
            yield None, "請先在前面分頁完成資料處理，並在此頁籤確認所有欄位與模型選擇！"
            return

        state_dim, action_dim, reward_dim = len(state_cols), len(action_cols), len(reward_cols)

        env_model_path = os.path.join(tab_shared.MODEL_DIR, env_model_name, env_model_file)
        environment_model = models.build_model(
            model_name=env_model_name, input_size=state_dim + action_dim, output_size=reward_dim,
            hidden_size=64, num_layers=2
        )
        environment_model.load_state_dict(torch.load(env_model_path, map_location=models.DEVICE))

        sac_agent = models.SoftActorCritic(state_dim, action_dim)
        replay_buffer = models.ReplayBuffer(capacity=tab_shared.REPLAY_BUFFER_CAPACITY, sequence_length=tab_shared.SEQUENCE_LENGTH)

        status_record = "---Starting Reinforce Training---\n"
        actor_loss_hist, critic_loss_hist = [], []

        train_generator = models.train_sac_agent(
            sac=sac_agent, environment_model=environment_model, initial_states=initial_states,
            reward_keys=reward_cols, train_buffer=replay_buffer, save_dir=tab_shared.SOLVER_MODEL_DIR,
            model_type=tab_shared.SOLVER_MODEL_TYPE, epochs=epochs, steps=steps, warmup_steps=warmup_steps,
            lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, alpha=alpha,
            early_stop_patience=tab_shared.EARLY_STOPPING_SOLVER
        )

        for actor_loss, critic_loss, status in train_generator:
            if actor_loss is None: # 訓練結束或早停
                status_record += status + '\n'
                yield loss_fig, status_record
                break

            status_record += status + '\n'
            actor_loss_hist.append(actor_loss)
            critic_loss_hist.append(critic_loss)
            
            epochs_range = list(range(1, len(actor_loss_hist) + 1))
            loss_fig = tab_shared.create_matplotlib_figure(
                datasets=[
                    {'x': epochs_range, 'y': actor_loss_hist, 'label': 'Actor Loss'},
                    {'x': epochs_range, 'y': critic_loss_hist, 'label': 'Critic Loss'}
                ], title="Solver Loss"
            )
            yield loss_fig, status_record
            plt.close(loss_fig)

    BTN_solver_train.click(
        fn=start_solver_train,
        inputs=[DP_env_model_name, DP_env_model_file, solver_state_cols, solver_action_cols, solver_reward_cols, initial_states_np, NUM_solver_epochs, NUM_solver_steps, NUM_warmup_steps, NUM_solver_lr_actor, NUM_solver_lr_critic, NUM_solver_gamma, NUM_solver_alpha],
        outputs=[solver_loss_plot, output_solver_hyp]
    )

    # 當進入求解器分頁時，更新環境模型列表
    solver_tab.select(fn=tab_shared.get_model_files, inputs=DP_env_model_name, outputs=DP_env_model_file)
    # 當環境模型類別改變時，也更新環境模型檔案列表
    DP_env_model_name.change(
        fn=tab_shared.get_model_files,
        inputs=DP_env_model_name,
        outputs=DP_env_model_file
    )
    # 當求解器欄位確認後，啟用訓練按鈕
    BTN_solver_cols.click(fn=lambda: gr.update(interactive=True), outputs=[BTN_solver_train])

    return solver_tab, DP_env_model_file, solver_state_cols, solver_action_cols, solver_reward_cols, initial_states_np, state_scaler_obj