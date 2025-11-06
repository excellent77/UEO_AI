import os
import gradio as gr
import matplotlib.pyplot as plt

# 共享常數
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
MODEL_DIR = os.path.join(os.getcwd(), 'model_record')
SEQUENCE_LENGTH = 60
EARLY_STOPPING_PREDICT = 10
EARLY_STOPPING_SOLVER = 10
SOLVER_MODEL_TYPE = "SAC_LSTM"
SOLVER_MODEL_DIR = os.path.join(MODEL_DIR, 'RL_models')
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
REPLAY_BUFFER_CAPACITY = 10000
BATCH_SIZE = 128

def create_matplotlib_figure(datasets: list, title: str) -> plt.Figure:
    """
    根據傳入的多組數據生成 matplotlib 圖形。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    for data in datasets:
        ax.plot(data.get('x'), data.get('y'), marker='o', linestyle='-', label=data.get('label'))
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(title.split(' ')[0]) # e.g., "Loss Curves" -> "Loss"
    ax.grid(True)
    ax.legend()
    return fig

def get_model_files(model_name):
    """
    根據模型名稱獲取對應目錄下的模型檔案列表。
    """
    if not model_name:
        return gr.update(choices=[], value=None)
    save_dir = os.path.join(MODEL_DIR, model_name)
    files = [None] + os.listdir(save_dir) if os.path.exists(save_dir) else [None]
    return gr.update(choices=files, value=None)

def get_solver_model_files():
    """
    獲取求解器模型目錄下的 Actor 模型檔案列表。
    """
    if os.path.exists(SOLVER_MODEL_DIR):
        return [f for f in os.listdir(SOLVER_MODEL_DIR) if 'ACTOR.pth' in f]
    return []