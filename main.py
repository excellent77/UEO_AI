import gradio as gr
import pandas as pd

# 設備與資料特性對應表
machine_features = {
    "空壓機": ["倒U型曲線"],
    "HVAC系統": ["依負載變動", "週期波動"],
    "冷卻塔": ["依負載變動", "倒U型曲線"],
    "馬達": ["倒U型", "脈衝波動"],
    "加熱爐": ["脈衝波動"],
    "鍋爐": ["依負載變動", "脈衝波動"],
    "泵浦": ["依負載變動", "週期波動"],
    "風機": ["週期波動"]
}

machine_types = list(machine_features.keys())

'''# 整理所有資料特性（去重）
all_features = []
for feats in machine_features.values():
    for f in feats:
        if f not in all_features:
            all_features.append(f)'''

def update_columns(file):
    if file is None:
        return [gr.update(choices=[], value=[]), gr.update(choices=[], value=[]), None]
    df = pd.read_csv(file.name, encoding="utf-8")
    cols = df.columns.tolist()
    # CheckboxGroup 用 gr.update，Dataframe 傳資料
    return [gr.update(choices=cols, value=[]), gr.update(choices=cols, value=[]), df.head()]

def show_selection(machine, train_cols, label_cols):
    result = f"已選設備：{machine}\n"
    result += f"用於判斷欄位：{train_cols}\n"
    result += f"用於預測欄位：{label_cols}\n"
    return result

with gr.Blocks() as demo:
    gr.Markdown("## **耗能設備的通用性能源操作優化框架** ")

    with gr.Tabs(selected=0):  # 預設顯示第一個 Tab（上傳資料）
        with gr.Tab("上傳資料"):
            csv_file = gr.File(label="上傳感測器資料（CSV）", file_types=[".csv"])
            preview = gr.Dataframe(label="資料預覽", interactive=False)
        with gr.Tab("選擇欄位"):
            machine_dropdown = gr.Dropdown(choices=machine_types, label="請選擇設備類別")
            train_cols = gr.CheckboxGroup(choices=[], label="請選擇要用於「判斷」的欄位")
            label_cols = gr.CheckboxGroup(choices=[], label="請選擇要用於「預測」的欄位")
            output_text = gr.Textbox(label="選擇結果", interactive=False)
            btn = gr.Button("確認選擇")
            btn.click(
                fn=show_selection,
                inputs=[machine_dropdown, train_cols, label_cols],
                outputs=output_text
            )

    # 上傳CSV時，所有特性欄位都更新選項，Dataframe只傳資料
    csv_file.change(
        fn=update_columns,
        inputs=csv_file,
        outputs=[train_cols, label_cols, preview]
    )

demo.launch()
