调用方式：

将ultralytics\utils\loss.py

里面的

self.use_angle_dfl = False

改成

self.use_angle_dfl = True

即可使用angloss损失函数



这个改进点在多个数据集上已经验证 有效涨点