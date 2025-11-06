import gradio as gr
import pandas as pd
import ui_tabs.Preprocessing as Preprocessing
from . import tab_shared

def create_dataclean_tab(train_data, val_data, RD_datetime, CBG_act, CBG_reward, BTN_select):
    """
    創建並返回「資料清洗」分頁的 UI 元件和相關的事件處理。
    """
    with gr.Tab("資料清洗"):
        DP_fill = gr.Dropdown(choices=Preprocessing.FILL_STRATEGIES, label="請選擇缺失值填補策略", value='mean')
        NUM_fill_constant = gr.Number(label="請輸入填補常數", value=0, visible=False, interactive=True)
        DP_scale = gr.Dropdown(choices=Preprocessing.SCALE_METHODS, label="請選擇正規化方式", value='minmax')
        NUM_sequence_length = gr.Slider(label="序列長度 (以多少個時間點為一個序列間隔)", minimum=1, maximum=100, step=1, value=tab_shared.SEQUENCE_LENGTH, interactive=True)

        BTN_clean = gr.Button("確認選擇", interactive=False)
        output_clean = gr.Textbox(label="資料 shape", interactive=False)
        clean_features = gr.State()
        clean_targets = gr.State()
        scalers = gr.State()

    def preprocess_and_export(train_df_file, val_df_file, datetime_col, act_cols, reward_cols, fill_strategy, scale_method, sequence_length, fill_value=0):
        """
        調用 Preprocessing 模組進行資料預處理。
        """
        try:
            train_df = pd.read_csv(train_df_file.name)
            val_df = pd.read_csv(val_df_file.name)
        except AttributeError:
            raise gr.Error("請先上傳訓練與驗證資料集！")

        preprocess_kwargs = {}
        if fill_strategy == 'constant':
            preprocess_kwargs['fill_value'] = fill_value

        train_feature, train_target, x_scaler, y_scaler = Preprocessing.preprocess_for_lstm(
            train_df,
            datetime_col=datetime_col, feature_cols=act_cols + reward_cols, target_cols=reward_cols,
            fill_strategy=fill_strategy, scale_method=scale_method,
            sequence_length=sequence_length,
            **preprocess_kwargs
        )

        val_feature, val_target, _, _ = Preprocessing.preprocess_for_lstm(
            val_df,
            datetime_col=datetime_col, feature_cols=act_cols + reward_cols, target_cols=reward_cols,
            fill_strategy=fill_strategy, scale_method=scale_method,
            sequence_length=sequence_length,
            apply_scaler={"feature": x_scaler, "target": y_scaler},
            **preprocess_kwargs
        )

        shape_str = f'''
        train特徵 shape: {train_feature.shape}; train標籤 shape: {train_target.shape}
        val特徵 shape: {val_feature.shape}; val標籤 shape: {val_target.shape}'''

        return [
            shape_str,
            {"train": train_feature, "val": val_feature},
            {"train": train_target, "val": val_target},
            {"feature": x_scaler, "target": y_scaler}
        ]

    BTN_clean.click(
        fn=preprocess_and_export,
        inputs=[train_data, val_data, RD_datetime, CBG_act, CBG_reward, DP_fill, DP_scale, NUM_sequence_length, NUM_fill_constant],
        outputs=[output_clean, clean_features, clean_targets, scalers]
    )

    def update_fill_constant_visibility(strategy):
        return gr.update(visible=strategy == 'constant')

    DP_fill.change(
        fn=update_fill_constant_visibility,
        inputs=DP_fill,
        outputs=NUM_fill_constant
    )

    BTN_select.click(
        fn=lambda: gr.update(interactive=True), outputs=[BTN_clean]
    )

    return DP_fill, DP_scale, NUM_sequence_length, BTN_clean, output_clean, clean_features, clean_targets, scalers