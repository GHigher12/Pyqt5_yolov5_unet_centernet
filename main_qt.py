# -*- coding: UTF-8 -*-  
# @Time : 2022/8/8 12:55
# @File : main_qt.py
# @Software: PyCharm
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget, QMessageBox
from Qt_yolo import Ui_MainWindow
from PyQt5.QtCore import pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import os

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
# LoadWebcam 的最后一个返回值改为 self.cap
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

from PIL import Image
from centernet import CenterNet
from unet import Unet


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = 'model_data/yolov5_weights/yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25

    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):

        # Initialize
        device = select_device(device)
        half &= device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        # Dataloader
        if self.source.isnumeric():
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        for path, img, im0s, self.vid_cap in dataset:
            statistic_dic = {name: 0 for name in names}
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                im0 = im0s.copy()

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        statistic_dic[names[c]] += 1
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

            time.sleep(1 / 20)
            # print(type(im0s))
            self.send_img.emit(im0)
            self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
            self.send_statistic.emit(statistic_dic)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        
        self.det_thread = DetThread()   # 多线程yolov5对象
        self.centernet = CenterNet()    # Centernet对象
        self.unet = Unet()  # Unet对象
       
        self.crop = False   # Centernet检测是否裁剪
        self.count = True   # Centernet检测是否计数   
        self.name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
                             "chair", "cow",
                             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                             "train",
                             "tvmonitor"]   # unet检测名称类别
        
        self.model = 'model_data/yolov5_weights/yolov5s.pt'  # yolov5模型默认路径
        self.centernet.model_path = 'model_data/centernet_weights/centernet_resnet50_voc.pth'   # centernet模型默认路径
        self.unet.model_path = 'model_data/unet_weights/unet_vgg_voc.pth' # unet模型路径文件 
        
        self.flag = False   # 视频或摄像头运行标志位
        self.det_thread.source = '0'    # yolov5检测文件的默认路径
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.label_show))  # 原始图像的显示
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_detect))   # yolov5检测图像的显示
        
        self.det_thread.send_statistic.connect(self.show_statistic)  # yolov5类别数目显示
        self.detectButton.clicked.connect(self.term_or_con)  # ’检测文件‘按钮被点击发出信号
        self.stopButton.clicked.connect(self.detect_stop)  # ‘停止检测’按钮被点击发出信号
        
        self.importButton.clicked.connect(self.open_file)   # ’导入文件‘按钮被点击发出信号
        self.camopenButton.clicked.connect(self.camera_open)    # ’打开摄像头‘按钮被点击发出信号
        self.camcloseButton.clicked.connect(self.camera_close)  # ’关闭摄像头‘按钮被点击发出信号
        self.weightButton.clicked.connect(self.open_model)  # ‘yolov5权重’按钮被点击发出信号
        self.weight_cenButton.clicked.connect(self.open_model_cen)  # ‘centernet权重’按钮被点击发出信号
        self.weight_cenButton_2.clicked.connect(self.open_model_unet)   # ’unet权重‘按钮被点击发出信号

        self.status_bar_init()  # 界面状态栏初始化
        self.horizontalSlider.valueChanged.connect(lambda: self.conf_change(self.horizontalSlider)) # 置信度滑块改变发出信号
        self.spinBox.valueChanged.connect(lambda: self.conf_change(self.spinBox))   # spinBox根据slider的值改变而改变
        self.camcloseButton.setEnabled(False)   # 禁用关闭摄像头按钮
        self.stopButton.setEnabled(False)   # 禁用停止检测按钮

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("./icon/background.jpg")   # 导入背景图片
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)

    # 更改置信度
    def conf_change(self, method):
        if method == self.horizontalSlider:
            self.spinBox.setValue(self.horizontalSlider.value())
        if method == self.spinBox:
            self.horizontalSlider.setValue(self.spinBox.value())
        self.det_thread.conf_thres = self.horizontalSlider.value() / 100    # 赋值给yolov5置信度
        self.centernet.confidence = self.horizontalSlider.value() / 100    # # 赋值给centernet置信度
        self.statusbar.showMessage("置信度已更改为：" + str(self.det_thread.conf_thres))    # 状态栏显示置信度

    def status_bar_init(self):
        self.statusbar.showMessage('界面已准备')

    # 导入文件处理函数
    def open_file(self):
        source = QFileDialog.getOpenFileName(self, '选取视频或图片', "data/", "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                       "*.jpg *.png)")
        self.centernet_img_path = source[0]  # 提取文件路径
        # print(self.centernet_img_path)
        if source[0]:
            self.det_thread.source = source[0]  # 赋值给yolov5检测文件路径
        self.statusbar.showMessage('加载文件：{}'.format(os.path.basename(self.det_thread.source)
                                                    if os.path.basename(self.det_thread.source) != '0'
                                                    else '摄像头设备'))   # 状态栏显示 

    # 检测文件处理函数
    def term_or_con(self):
        self.detectButton.setEnabled(False)  # 禁用检测文件按钮
        self.stopButton.setEnabled(True)    # 启用停止检测按钮
        self.flag = True    # 设置标志位为真
        self.det_thread.start()  # 开启多线程
        self.statusbar.showMessage('正在检测 >> yolov5模型：{}，centernet模型：{}，unet模型{}，文件：{}'.
                                   format(os.path.basename(self.det_thread.weights),
                                          os.path.basename(self.centernet.model_path),
                                          os.path.basename(self.unet.model_path),
                                          os.path.basename(self.det_thread.source)
                                          if os.path.basename(self.det_thread.source) != '0'
                                          else '摄像头设备'))   # 状态栏显示
        if self.centernet_img_path:     # 判断是为文件路径
            file_name = self.centernet_img_path.split('/')[-1]  # 提取文件名称
            # 判断文件类型
            if file_name[-3:] in ['jpg', 'png']:    
                self.show_net_img()
            else:
                self.show_net_video()
        else:
            self.show_net_video()

    # 显示图像
    def show_net_img(self):
        image = Image.open(self.centernet_img_path)  # 导入图片
        r_image = self.centernet.detect_image(image, crop=self.crop, count=self.count)  # centernet检测
        n_image = self.unet.detect_image(image, count=self.count, name_classes=self.name_classes)   # unet检测
        if self.centernet.class_dic:    # 判断centernet检测的类别数目是否为空
            self.listWidget_cen.clear()   # 清空显示窗口
            cen_class = [i + '：' + str(k) for i, k in self.centernet.class_dic.items()]  # 字典类型转为列表
            self.listWidget_cen.addItems(cen_class)     # 显示centernet检测的类别数目
            print(cen_class)  
        # print("模型为{}".format(self.centernet.model_path))

        r_image = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)  # PIL类型转换为BGR图片格式
        r_image = self.resize_img(r_image, self.label_centernet.width(), self.label_centernet.height())  # 使图片适应label窗口大小
        
        qt_img = QImage(r_image[:], r_image.shape[1], r_image.shape[0], r_image.shape[1] * 3, QImage.Format_RGB888)  # 转换为QImage格式
        pixmap_img = QPixmap.fromImage(qt_img)
        self.label_centernet.setPixmap(pixmap_img)  # label 控件显示centernet检测图片

        n_image = cv2.cvtColor(np.asarray(n_image), cv2.COLOR_RGB2BGR)
        n_image = self.resize_img(n_image, self.label_unet.width(), self.label_unet.height())
        qt_img_unet = QImage(n_image[:], n_image.shape[1], n_image.shape[0], n_image.shape[1] * 3, QImage.Format_RGB888)
       
        pixmap_img_unet = QPixmap.fromImage(qt_img_unet)
        self.label_unet.setPixmap(pixmap_img_unet)  # label 控件显示unet检测图片

    # 显示视频
    def show_net_video(self):
        self.capture = cv2.VideoCapture(self.centernet_img_path)    # 获取视频路径
        fps = self.capture.get(cv2.CAP_PROP_FPS)  # 获得视频帧率
        while self.capture.isOpened():
            ret, frame = self.capture.read()  # 读取视频一帧
            frame = Image.fromarray(np.uint8(frame))
            c_frame = np.array(self.centernet.detect_image(frame, crop=self.crop, count=self.count))  # centernet检测
            n_frame = np.array(self.unet.detect_image(frame, count=self.count, name_classes=self.name_classes))  # unet检测

            if self.centernet.class_dic:
                self.listWidget_cen.clear()
                cen_class = [i + '：' + str(k) for i, k in self.centernet.class_dic.items()]
                self.listWidget_cen.addItems(cen_class)
                print(cen_class)
            
            # 转换图片格式及显示
            c_frame = self.resize_img(c_frame, self.label_centernet.width(), self.label_centernet.height())
            qt_frame = QImage(c_frame[:], c_frame.shape[1], c_frame.shape[0], c_frame.shape[1] * 3,
                              QImage.Format_RGB888)
            pixmap_frame = QPixmap.fromImage(qt_frame)
            self.label_centernet.setPixmap(pixmap_frame)

            n_frame = self.resize_img(n_frame, self.label_unet.width(), self.label_unet.height())
            qt_net_frame = QImage(n_frame[:], n_frame.shape[1], n_frame.shape[0], n_frame.shape[1] * 3,
                                  QImage.Format_RGB888)
            pixmap_net_frame = QPixmap.fromImage(qt_net_frame)
            self.label_unet.setPixmap(pixmap_net_frame)

            self.label_centernet.show()
            self.label_unet.show()
            cv2.waitKey(int(500 / fps))

    #   调整图片大小适应label
    def resize_img(self, img, label_w, label_h):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] > label_h or img.shape[1] > label_w:
            size = min(label_h / img.shape[0], label_w / img.shape[1])
        else:
            size = 1
        # print(size)
        img = cv2.resize(img, dsize=(int(img.shape[1] * size), int(img.shape[0] * size)), interpolation=cv2.INTER_AREA)
        return img

    # 停止检测
    def detect_stop(self):  
        self.flag = False   # 标志位设为假
        self.det_thread.terminate()
        if hasattr(self.det_thread, 'vid_cap'):
            if self.det_thread.vid_cap:
                # 释放视频
                self.det_thread.vid_cap.release()  
                self.capture.release()
        self.statusbar.showMessage('结束检测')
        self.detectButton.setEnabled(True)  # 启用检测文件按钮
        self.stopButton.setEnabled(False)   # 禁用停止检测按钮

    # 导入yolov5权重
    def open_model(self):
        self.model = QFileDialog.getOpenFileName(self, '选取模型', "model_data/yolov5_weights", "Model File(*.pt)")[0]
        if self.model: 
            self.det_thread.weights = self.model
            print(self.det_thread.weights)
        self.statusbar.showMessage('加载yolov5模型：' + os.path.basename(self.det_thread.weights))

    # 导入centernet权重
    def open_model_cen(self):
        self.model_cen = QFileDialog.getOpenFileName(self, '选取模型', 'model_data/centernet_weights', "Model File(*.pth)")[0]
        print(self.model_cen)
        if self.model_cen:
            self.centernet.model_path = self.model_cen   # 赋值centernet模型文件
        self.statusbar.showMessage('加载centernet模型：' + os.path.basename(self.centernet.model_path))

    # 导入unet权重
    def open_model_unet(self):
        self.model_unet = QFileDialog.getOpenFileName(self, '选取模型', 'model_data/unet_weights', "Model File(*.pth)")[0]
        print(self.model_unet)
        if self.model_unet:
            self.unet.model_path = self.model_unet   #  赋值unet模型文件
        self.statusbar.showMessage('加载unet模型：' + os.path.basename(self.unet.model_path))

    # 打开摄像头
    def camera_open(self):
        self.flag = True  # 标志位设置为真
        # 设置摄像头路径
        self.det_thread.source = '0'   
        self.centernet_img_path = 0
        self.statusbar.showMessage('摄像头已打开')
        self.camcloseButton.setEnabled(True)  # 启用关闭摄像头按钮
        self.camopenButton.setEnabled(False)  # 禁用打开摄像头按钮
    
    # 关闭摄像头
    def camera_close(self):
        self.flag = False
        self.det_thread.terminate() 
        if hasattr(self.det_thread, 'vid_cap'):
            self.det_thread.vid_cap.release()
        if self.detectButton.isChecked():
            self.detectButton.setChecked(False)
        self.statusbar.showMessage('摄像头已关闭')
        self.camcloseButton.setEnabled(False)  # 禁用关闭摄像头按钮
        self.camopenButton.setEnabled(True)  # 启用打开摄像头按钮

    # 显示检测类别数目
    def show_statistic(self, statistic_dic):
        try:
            self.listWidget.clear()  # 情况显示框
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)  # 按数量排序
            statistic_dic = [i for i in statistic_dic if i[1] > 0]  # 保留数量大于0的类别
            results = [str(i[0]) + '：' + str(i[1]) for i in statistic_dic]  # 字典类型转为列表
            self.listWidget.addItems(results)  # 显示检测类别
            print(results)

        except Exception as e:
            print(repr(e))

    # 添加中文的确认退出提示框
    def closeEvent(self, event):
        # 创建一个消息盒子（提示框）
        quitMsgBox = QMessageBox()
        # 设置提示框的标题
        quitMsgBox.setWindowTitle('确认提示')
        # 设置提示框的内容
        quitMsgBox.setText('你确认退出吗？')
        # 设置按钮标准，一个yes一个no
        quitMsgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        # 获取两个按钮并且修改显示文本
        buttonY = quitMsgBox.button(QMessageBox.Yes)
        buttonY.setText('确定')
        buttonN = quitMsgBox.button(QMessageBox.No)
        buttonN.setText('取消')
        quitMsgBox.exec_()
        # 判断返回值，如果点击的是Yes按钮，我们就关闭组件和应用，否则就忽略关闭事件
        if quitMsgBox.clickedButton() == buttonY:
            if self.flag:   # 判断摄像头和视频是否运行
                self.detect_stop()
            event.accept()
        else:
            event.ignore()

    @staticmethod
    def show_image(img_src, label):
        try:
            # 保持纵横比
            # 找出长边
            img_h, img_w = img_src.shape[0], img_src.shape[1]
            label_w = label.geometry().width()
            label_h = label.geometry().height()
            if img_h > label_h or img_w > label_w:
                size = min(label_h / img_h, label_w / img_w)
            else:
                size = 1
            # print(size)
            img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(img_src, dsize=(int(img_w * size), int(img_h * size)), interpolation=cv2.INTER_AREA)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
