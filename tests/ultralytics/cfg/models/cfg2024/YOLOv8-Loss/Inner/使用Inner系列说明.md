当使用 ultralytics\cfg\models\cfg2024\YOLOv8-Loss\Inner 文件下的内容时

需要将ultralytics\utils\NewLoss\iouloss.py文件里面的
代码
Inner = False
改成
Inner = True
即可


改成这样之后，调用比如 python train_v8.py --cfg ultralytics\cfg\models\cfg2024\YOLOv8-Loss\YOLOv8-CIoU.yaml
就使用的是Inner-CIoU函数，其他ultralytics\cfg\models\cfg2024\YOLOv8-Loss目录下的其他一级函数（如SIoU、DIoU、PIoU）也是同理