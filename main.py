import gradio as gr
import pandas as pd
import Preprocessing



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


def update_columns(file):
    if file is None:
        return [
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]), 
            None
        ]
    
    df = pd.read_csv(file.name, encoding="utf-8")
    cols = df.columns.tolist()
    # CheckboxGroup 用 gr.update，Dataframe 傳資料
    return [
        gr.update(choices=cols, value=[]),
        gr.update(choices=cols, value=[]),
        gr.update(choices=cols, value=[]),
        df.head()
    ]


def show_selection(
        machine,
        datetime_col,
        train_cols,
        label_cols )->str:
    result = f"已選設備：{machine}\n"
    result += f"時間欄位：{datetime_col}\n"
    result += f"用於判斷欄位：{train_cols}\n"
    result += f"用於預測欄位：{label_cols}\n"
    return result


def preprocess_and_export(
    df,
    datetime_col, 
    feature_cols,
    target_cols,
    fill_strategy,
    scale_method ):
    # 讀取資料
    X, y, scaler = Preprocessing.preprocess_for_lstm(
        df, datetime_col, feature_cols, target_cols, fill_strategy, scale_method
    )
    shape_str = f"特徵 shape: {X.shape}; 標籤 shape: {y.shape}"

    return [shape_str, X, y, scaler]



if __name__ == "__main__":
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
                btn_select = gr.Button("確認選擇")
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
                btn_clean = gr.Button("確認選擇")
                shape_info = gr.Textbox(label="資料 shape", interactive=False)
                clean_feature = gr.Dataframe(label="清洗後特徵資料預覽", interactive=False)
                clean_labels = gr.Dataframe(label="清洗後標籤資料預覽", interactive=False)

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

        # 上傳CSV時，更新欄位選項與預覽
        csv_file.change(
            fn=update_columns,
            inputs=csv_file,
            outputs=[datetime_col, feature_cols, target_cols, preview]
        )

    print("Starting Gradio demo...")
    demo.launch()
