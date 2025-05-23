### 文件准备

关于 ultralytics\script\print_plot\yolov8_origin_conv.csv 这些文件 是yolov8训练自动生成的,在runs文件里面

将其复制到这个文件夹下，然后原始名称results.csv 改为 yolov8_xxx.csv的格式即可运行


### 要打印将多个results文件的图画出来

执行命令：
python ultralytics\script\print_plot\将多个results文件的图画出来.py


### 要单独打印多个map对比的图

执行命令：
python ultralytics\script\print_plot\multi_map_result.py


### 要单独打印多个loss对比的图

执行命令：
python ultralytics\script\print_plot\multi_loss_result.py


### 打印mAP精度性能图

执行命令：
python ultralytics\script\打印mAP精度性能图.py