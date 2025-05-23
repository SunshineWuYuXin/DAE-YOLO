#coding: utf-8
from ultralytics import YOLO
import matplotlib


if __name__ == '__main__':

    model = YOLO('/root/autodl-tmp/runs/train/exp/weights/best.pt')
    metrics = model.val(data = '/root/autodl-tmp/ultralytics/cfg/datasets/DOTA.yaml')
