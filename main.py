import os
import gradio as gr
import matplotlib

matplotlib.use('Agg')

import ui_tabs.models as models
from ui_tabs import tab_upload, tab_select_cols, tab_dataclean, tab_hyperparam, tab_solver, tab_inference, tab_shared

if __name__ == "__main__":
    for mode in models.MODEL_LIST:
        os.makedirs(os.path.join(tab_shared.MODEL_DIR, mode), exist_ok=True)
    os.makedirs(tab_shared.SOLVER_MODEL_DIR, exist_ok=True)
    os.makedirs(tab_shared.OUTPUT_DIR, exist_ok=True)

    with gr.Blocks() as demo:
        gr.Markdown("## 耗能設備的通用性能源操作優化框架 ")

        with gr.Tabs(selected=0):
            train_data, train_load, val_data, val_load = tab_upload.create_upload_tab()

            (DP_machine, RD_datetime, CBG_act, CBG_reward, BTN_select, output_select) = \
                tab_select_cols.create_select_cols_tab(train_data, val_data)

            (DP_fill, DP_scale, NUM_sequence_length, BTN_clean, output_clean,
             clean_features, clean_targets, scalers) = \
                tab_dataclean.create_dataclean_tab(train_data, val_data, RD_datetime, CBG_act, CBG_reward, BTN_select)

            hyperparam_tab, DP_model_name, DP_pre_model = \
                tab_hyperparam.create_hyperparam_tab(clean_features, clean_targets, scalers, output_clean)

            (solver_tab, DP_env_model_file, solver_state_cols, solver_action_cols,
             solver_reward_cols, initial_states_np, state_scaler_obj) = \
                tab_solver.create_solver_tab(train_data, val_data, CBG_act, CBG_reward, DP_fill, DP_scale)

            inference_tab = tab_inference.create_inference_tab(train_data, solver_state_cols, solver_action_cols, state_scaler_obj)

    print("Starting Gradio...")
    demo.launch()
