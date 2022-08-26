# -*- coding: UTF-8 -*-  
# @Time : 2022/8/8 12:55
# @File : main_qt.py
# @Software: PyCharm
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
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
        imgsz = check_img_size(imgsz, s=stride)  # check image size 32
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
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)  # 加载图片文件

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

            time.sleep(1 / 15)
            # print(type(im0s))
            self.send_img.emit(im0)
            self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
            self.send_statistic.emit(statistic_dic)


class New_Qthead(QThread):
    centernet_img = pyqtSignal(np.ndarray)
    unet_img = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)

    def __init__(self):
        super(New_Qthead, self).__init__()
        self.path = '0'
        self.capture = None
        self.centernet = ''
        self.unet = ''
        self.crop = False
        self.count = False
        self.name_classes = []

    def run(self):  # 在使用这个类的时候直接调用这个接口就行
        if self.path:
            file_name = self.path.split('/')[-1]
            if file_name[-3:] in ['jpg', 'png']:
                image = Image.open(self.path)
                r_image = self.centernet.detect_image(image, crop=self.crop, count=self.count)
                n_image = self.unet.detect_image(image, count=self.count, name_classes=self.name_classes)
                r_image = cv2.cvtColor(np.array(r_image), cv2.COLOR_BGR2RGB)
                n_image = cv2.cvtColor(np.array(n_image), cv2.COLOR_BGR2RGB)
                self.centernet_img.emit(r_image)
                self.unet_img.emit(n_image)
                self.send_statistic.emit(self.centernet.class_dic)
            else:
                self.show_video(self.path)
        else:
            self.show_video(0)

    def show_video(self, filepath):
        self.capture = cv2.VideoCapture(filepath)
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            frame = Image.fromarray(np.uint8(frame))
            c_frame = np.array(self.centernet.detect_image(frame, crop=self.crop, count=self.count))
            n_frame = np.array(self.unet.detect_image(frame, count=self.count, name_classes=self.name_classes))
            self.centernet_img.emit(c_frame)
            self.unet_img.emit(n_frame)
            self.send_statistic.emit(self.centernet.class_dic)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.det_thread = DetThread()
        self.qthead_net = New_Qthead()
        self.centernet = CenterNet()
        self.unet = Unet()
        # self.mode = "predict"

        self.model = './yolov5s.pt'
        self.centernet.model_path = "model_data/centernet_resnet50_voc.pth"
        self.unet.model_path = "model_data/unet_vgg_voc.pth"
        self.det_thread.source = '0'
        self.qthead_net.path = '0'
        self.qthead_net.centernet = self.centernet
        self.qthead_net.unet = self.unet
        self.qthead_net.count = True
        self.qthead_net.crop = False
        self.qthead_net.name_classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
                                        "cat", "chair", "cow",
                                        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                        "sofa", "train",
                                        "tvmonitor"]
        self.qthead_net.centernet_img.connect(lambda x: self.show_image(x, self.label_centernet))
        self.qthead_net.unet_img.connect(lambda x: self.show_image(x, self.label_unet))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_detect))
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.label_show))

        self.det_thread.send_statistic.connect(lambda x: self.show_statistic(x, self.listWidget))
        self.qthead_net.send_statistic.connect(lambda x: self.show_statistic(x, self.listWidget_cen))
        self.detectButton.clicked.connect(self.term_or_con)
        self.stopButton.clicked.connect(self.detect_stop)
        self.importButton.clicked.connect(self.open_file)
        self.weightButton.clicked.connect(self.open_model)

        self.weight_cenButton.clicked.connect(self.open_model_cen)
        self.weight_cenButton_2.clicked.connect(self.open_model_unet)

        self.status_bar_init()
        self.camopenButton.clicked.connect(self.camera_open)
        self.camcloseButton.clicked.connect(self.camera_close)
        self.horizontalSlider.valueChanged.connect(lambda: self.conf_change(self.horizontalSlider))
        self.spinBox.valueChanged.connect(lambda: self.conf_change(self.spinBox))
        self.camcloseButton.setEnabled(False)
        self.stopButton.setEnabled(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        pixmap = QPixmap("./icon/background06.jpg")
        # 绘制窗口背景，平铺到整个窗口，随着窗口改变而改变
        painter.drawPixmap(self.rect(), pixmap)

    # 更改置信度
    def conf_change(self, method):
        if method == self.horizontalSlider:
            self.spinBox.setValue(self.horizontalSlider.value())
        if method == self.spinBox:
            self.horizontalSlider.setValue(self.spinBox.value())
        self.det_thread.conf_thres = self.horizontalSlider.value() / 100
        self.centernet.confidence = self.horizontalSlider.value() / 100
        self.statusbar.showMessage("置信度已更改为：" + str(self.det_thread.conf_thres))

    def status_bar_init(self):
        self.statusbar.showMessage('界面已准备')

    def open_file(self):
        source = QFileDialog.getOpenFileName(self, '选取视频或图片', "data/", "Pic File(*.mp4 *.mkv *.avi *.flv "
                                                                       "*.jpg *.png)")
        self.qthead_net.path = source[0]
        # print(self.qthead_net.path)
        if source[0]:
            self.det_thread.source = source[0]
        self.statusbar.showMessage('加载文件：{}'.format(os.path.basename(self.det_thread.source)
                                                    if os.path.basename(self.det_thread.source) != '0'
                                                    else '摄像头设备'))

    def term_or_con(self):
        self.qthead_net.start()
        time.sleep(0.01)
        self.det_thread.start()
        self.statusbar.showMessage('正在检测 >> yolov5模型：{}，centernet模型：{}，unet模型{}，文件：{}'.
                                   format(os.path.basename(self.det_thread.weights),
                                          os.path.basename(self.centernet.model_path),
                                          os.path.basename(self.unet.model_path),
                                          os.path.basename(self.det_thread.source)
                                          if os.path.basename(self.det_thread.source) != '0'
                                          else '摄像头设备'))
        self.detectButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def detect_stop(self):
        self.det_thread.terminate()
        self.qthead_net.terminate()
        if hasattr(self.det_thread, 'vid_cap'):
            if self.det_thread.vid_cap:
                self.det_thread.vid_cap.release()
                self.qthead_net.capture.release()
        self.statusbar.showMessage('结束检测')
        self.detectButton.setEnabled(True)
        self.stopButton.setEnabled(False)

    def open_model(self):
        self.model = QFileDialog.getOpenFileName(self, '选取模型', os.getcwd(), "Model File(*.pt)")[0]
        if self.model:
            self.det_thread.weights = self.model
        self.statusbar.showMessage('加载yolov5模型：' + os.path.basename(self.det_thread.weights))

    def open_model_cen(self):
        self.model_cen = QFileDialog.getOpenFileName(self, '选取模型', os.getcwd(), "Model File(*.pth)")[0]
        print(self.model_cen)
        if self.model_cen:
            self.centernet.model_path = self.model_cen
        self.statusbar.showMessage('加载centernet模型：' + os.path.basename(self.centernet.model_path))

    def open_model_unet(self):
        self.model_unet = QFileDialog.getOpenFileName(self, '选取模型', os.getcwd(), "Model File(*.pth)")[0]
        print(self.model_unet)
        if self.model_unet:
            self.unet.model_path = self.model_unet
        self.statusbar.showMessage('加载unet模型：' + os.path.basename(self.unet.model_path))

    def camera_open(self):
        self.det_thread.source = '0'
        self.qthead_net.path = 0
        self.statusbar.showMessage('摄像头已打开')
        self.camcloseButton.setEnabled(True)
        self.camopenButton.setEnabled(False)

    def camera_close(self):
        self.det_thread.terminate()
        self.qthead_net.terminate()
        if hasattr(self.det_thread, 'vid_cap'):
            self.det_thread.vid_cap.release()
            self.qthead_net.capture.release()
        if self.detectButton.isChecked():
            self.detectButton.setChecked(False)
        self.statusbar.showMessage('摄像头已关闭')
        self.camcloseButton.setEnabled(False)
        self.camopenButton.setEnabled(True)

    def show_statistic(self, statistic_dic, widget):
        try:
            widget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            widget.addItems(results)
            print(results)

        except Exception as e:
            print(repr(e))

    @staticmethod
    def show_image(img_src, label):
        try:
            label.setPixmap(QPixmap(""))
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
