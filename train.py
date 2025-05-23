import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model='/root/autodl-tmp/ultralytics/cfg/models/cfg2024/YOLOv8-Backbone/BiFormer/C2f_Biformer-InerIOU-ARA.yaml')
    model.train(data='/root/autodl-tmp/ultralytics/cfg/datasets/ROSD.yaml',
                imgsz=640,
                epochs=200,
                batch=16,
                workers=20,
                device='',
                optimizer='SGD',
                close_mosaic=10,
                resume=True,
                project='runs/train',
                name='exp',
                single_cls=False,
                cache=False,
                )
