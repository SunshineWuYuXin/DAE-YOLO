import contextlib
import math
import os
from copy import copy
from pathlib import Path
from urllib.error import URLError
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw, ImageFont

plt.rcParams['font.family'] = ['Microsoft YaHei']  # 或者其他您系统中支持的中文字体

def plot_yolo_model(dir=''):
    file = 'ultralytics/script/print_plot/yolov8_origin/yolov8_InnerIou.csv'
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    colors = ['orange', 'olive', 'green', 'red', 'purple', 'pink', 'brown', 'gray', 'blue', 'cyan']
    # 要对比的csv文件
    files = list(save_dir.glob('yolov8_*.csv'))
    #assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f, color in zip(files, colors):
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            # 1代表 results文件里面的train/box_loss 2代表train/cls_loss 
            for i, j in enumerate([1]):
            # for i, j in enumerate([2]):
                y = data.values[:, j].astype('float')
                ax.plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8, color=color)
                ax.set_title('改进的Loss曲线对比', fontsize=14)
                # ax.set_title(s[j], fontsize=14)
        except Exception as e:
            print(f'Warning: Plotting error for {f}: {e}')
    ax.legend(fontsize=10)
    ax.set_xlabel('epoch')
    ax.set_ylabel('Loss')
    fig.savefig(save_dir / 'loss_merge.png', dpi=1000) # merge models
    plt.close()

if __name__ == '__main__':
    plot_yolo_model()