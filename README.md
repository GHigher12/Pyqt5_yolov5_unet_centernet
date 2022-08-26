# Pyqt5_yolov5_unet_centernet
集yolov5、centernet、unet算法的pyqt5界面，可实现图片目标检测和语义分割

## 环境配置

### python

**Python>=3.6.0**，可以用anconda创建python虚拟环境

`conda create -n your_env_name python=x.x`

然后进入所创建的环境，下载所需安装包

`activate your_env_name`

### pytorch

pytorch可前往[官网下载](https://pytorch.org/get-started/previous-versions/)，选择符合自己的版本下载

注意!!!**PyTorch>=1.7**

### requirements.txt

在自己的创建虚拟环境下输入

`pip install -r requirements.txt`

直接下载requirements.txt中所需功能包：

> matplotlib>=3.2.2
>
> numpy>=1.18.5
>
> opencv-python>=4.1.2
>
> Pillow>=7.1.2
>
> PyYAML>=5.3.1
>
> requests>=2.23.0
>
> scipy>=1.4.1
>
> torch>=1.7.0
>
> torchvision>=0.8.1
>
> tqdm>=4.41.0
>
> pandas>=1.1.4
>
> seaborn>=0.11.0

## 运行程序

安装完环境后，直接运行**`main_qt.py`**文件即可

model_data中的权重文件下载

链接：https://pan.baidu.com/s/1vVPrdvuzCWyaXCnB-_6AeA 
提取码：z54u

运行结果

![image-20220822204610201](https://ghigher-picture-bed.oss-cn-qingdao.aliyuncs.com/img/image-20220822204610201.png)

***注意导入文件不要有中文路径和中文名称！！！***

否则会报错

![image-20220822205019336](https://ghigher-picture-bed.oss-cn-qingdao.aliyuncs.com/img/image-20220822205019336.png)

若文件路径有中文一定要更改路径或修改名称。

## 其他文件

- `Qt_yolo.ui`为Qtdesigner生成的UI文件
- `Qt_yolo.py`为PyUIC将Qt_yolo.ui转换的py文件
- `main_qt_qthead.py`为使用QT多线程编写，有bug
- `predict_yolo.py`为yolov5的检测文件，可直接运行
- `predict_cen.py`为centernet的检测文件，可直接运行
- `predict_unet.py`为unet的检测文件，可直接运行
- `centernet.py`为centernet的网络结构文件
- `unet.py`为unet的网络结构文件
- `icon`文件夹储存了qt界面的背景和图标
- `data`储存了待检测的图片及视频文件
- `model_data`储存了yolov5、centernet、unet的权重文件
- `md`储存了yolov5和centernet的官方说明文档
