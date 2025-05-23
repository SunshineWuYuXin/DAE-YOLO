import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.rcParams['font.family'] = ['Microsoft YaHei']  # 设置中文字体

def plot_yolo_model(file_list, save_dir='results'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    colors = ['orange', 'olive', 'green', 'red', 'purple', 'pink', 'brown', 'gray', 'blue', 'cyan']

    # 确保save_dir存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for f, color in zip(file_list, colors):
        try:
            data = pd.read_csv(f)
            x = data.iloc[:, 0].values
            y = data.iloc[:, 1].values.astype('float')  # 假设第二列是损失值
            ax.plot(x, y, marker='.', label=Path(f).stem, linewidth=2, markersize=8, color=color)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')

    ax.set_title('改进的Loss曲线对比', fontsize=14)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=10)
    fig.savefig(Path(save_dir) / 'loss_merge.png', dpi=1000)  # 保存合并的模型图
    plt.close()

if __name__ == '__main__':
    # 定义文件列表和保存目录
    file_list = [
        'yolov8_Dlou.csv', 'yolov8_Focal-WLou.csv', 'yolov8_GLou.csv',
        'yolov8_InnerIou.csv', 'yolov8.csv', 'yolov8_Shape Iou.csv',
        'yolov8_SLou.csv', 'yolov8_WLou.csv'
    ]
    save_dir = 'results'  # 你可以修改这个路径来保存你的图表
    plot_yolo_model(file_list, save_dir)