import gradio as gr
import pandas as pd
from . import tab_shared

def create_select_cols_tab(train_data, val_data):
    """
    創建並返回「選擇欄位」分頁的 UI 元件和相關的事件處理。
    """
    with gr.Tab("選擇欄位"):
        DP_machine = gr.Dropdown(choices=tab_shared.MACHINE_TYPES, label="請選擇設備類別")
        RD_datetime = gr.Radio(choices=[], label="請選擇「時間戳記」的欄位")
        CBG_act = gr.CheckboxGroup(choices=[], label="請選擇要用於「動作(Action)」的欄位")
        CBG_reward = gr.CheckboxGroup(choices=[], label="請選擇要用於「獎勵Reward」的欄位")

        BTN_select = gr.Button("確認選擇", interactive=False)
        output_select = gr.Textbox(label="選擇結果", interactive=False)

    def show_selection(machine: str, datetime_col: str, act_cols: list, reward_cols: list) -> str:
        """
        組裝及回傳當前欄位及設備的使用者選擇摘要，於 Gradio Textbox 顯示。
        """
        result = f"已選設備：{machine}\n"
        result += f"時間欄位：{datetime_col}\n"
        result += f"用於執行動作的欄位：{act_cols}\n"
        result += f"用於預測獎勵欄位：{reward_cols}\n"
        return result

    BTN_select.click(
        fn=show_selection,
        inputs=[DP_machine, RD_datetime, CBG_act, CBG_reward],
        outputs=output_select
    )

    def update_columns(file):
        """
        依據上傳的 CSV 檔案，讀取欄位並更新前端選單選項。
        """
        if file is None:
            return (
                gr.update(choices=[], value=None),
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=[]),
                gr.update(interactive=False)
            )
        
        df = pd.read_csv(file.name, encoding="utf-8")
        cols = df.columns.tolist()
        return (
            gr.update(choices=cols, value=None),
            gr.update(choices=cols, value=[]),
            gr.update(choices=cols, value=[]),
            gr.update(interactive=True)
        )

    # 當訓練資料上傳時，更新所有欄位選擇器
    train_data.change(
        fn=update_columns,
        inputs=train_data,
        outputs=[RD_datetime, CBG_act, CBG_reward, BTN_select]
    )

    return DP_machine, RD_datetime, CBG_act, CBG_reward, BTN_select, output_select