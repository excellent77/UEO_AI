import gradio as gr
import pandas as pd

def create_upload_tab():
    """
    創建並返回「上傳資料」分頁的 UI 元件和相關的事件處理。
    """
    with gr.Tab("上傳資料"):
        train_data = gr.File(label="上傳訓練資料集（CSV）", file_types=[".csv"])
        train_load = gr.Dataframe(label="訓練資料集預覽", interactive=False)
        val_data = gr.File(label="上傳驗證資料集（CSV）", file_types=[".csv"])
        val_load = gr.Dataframe(label="驗證資料集預覽", interactive=False)

    def update_preview(file):
        """
        讀取上傳的 CSV 檔案並顯示前幾行作為預覽。
        """
        if file is None:
            return None
        return pd.read_csv(file.name, encoding="utf-8").head()

    train_data.change(
        fn=update_preview,
        inputs=train_data,
        outputs=train_load
    )

    val_data.change(
        fn=update_preview,
        inputs=val_data,
        outputs=val_load
    )

    return train_data, train_load, val_data, val_load