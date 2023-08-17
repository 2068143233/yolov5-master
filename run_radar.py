import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLineEdit
from PyQt5.QtGui import *
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from radar2_ui import Ui_MainWindow
from utils.augmentations import letterbox
from utils.plots import Annotator
from utils.general import check_img_size, non_max_suppression, scale_segments
import argparse
import torch
from models.common import DetectMultiBackend
import random
from utils.torch_utils import select_device, smart_inference_mode
import torch.backends.cudnn as cudnn
import numpy as np
from cameracalibration import camera2

import cv2
tvec=np.array([[-25.59121107],
[ 9.24853872],
[ 57.91991721]])

rvec=np.array([[-0.03545517],
[-0.93580246],
[-0.00872234]])

mtx=np.array([[529.88898474,0,336.35141422],
    [0.0,532.89792279,249.99553094],
    [0.0,0.0,1]])
dist=np.array([[ 0.05603174,1.05575023,-0.0029776,-0.04539679,-5.03365241]])
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):  # 初始化函数
        super(MainWindow, self).__init__(parent)
        # UI界面
        self.setupUi(self)
        self.CAM_NUM = 0
        self.cap = cv2.VideoCapture()
        self.cap.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
        self.background()
        # 在label中播放视频
        self.init_timer()
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='runs/train/exp17/weights/best.pt', help='model.pt path(s)')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,
                            default='1', help='source')
        parser.add_argument('--img-size', type=int,
                            default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,
                            default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,
                            default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument(
            '--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',
                            help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',
                            help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',
                            help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,
                            help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument(
            '--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',
                            help='augmented inference')
        parser.add_argument('--update', action='store_true',
                            help='update all models')
        parser.add_argument('--project', default='runs/detect',
                            help='save results to project/name')
        parser.add_argument('--name', default='exp',
                            help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',
                            help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)

        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True

        # # Load model
        # self.model = attempt_load(
        #     weights, map_location=self.device)  # load FP32 model
        # stride = int(self.model.stride.max())  # model stride
        # self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        # if self.half:
        #     self.model.half()  # to FP16
            
            
        # Load model
        self.model = DetectMultiBackend(weights, device=self.device)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
            

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

    def background(self):
        # 文件选择按钮
        self.pushButton.clicked.connect(self.open_camera)
        self.pushButton_2.clicked.connect(self.close_camera)

        self.pushButton.setEnabled(True)
        # 初始状态不能关闭摄像头
        self.pushButton_2.setEnabled(False)

    # 打开相机采集视频
    def open_camera(self):
        # 获取选择的设备名称
        index = self.comboBox.currentIndex()
        print(index)
        self.CAM_NUM = index
        # 检测该设备是否能打开
        flag = self.cap.open(self.CAM_NUM)
        print(flag)
        if flag is False:
            QMessageBox.information(self, "警告", "该设备未正常连接", QMessageBox.Ok)
        else:
            # 幕布可以播放
            self.label.setEnabled(True)
            # 打开摄像头按钮不能点击
            self.pushButton.setEnabled(False)
            # 关闭摄像头按钮可以点击
            self.pushButton_2.setEnabled(True)
            self.timer.start()
            print("beginning！")

    # 关闭相机
    def close_camera(self):
        self.cap.release()
        self.pushButton.setEnabled(True)
        self.pushButton_2.setEnabled(False)
        self.timer.stop()

    # 播放视频画面
    def init_timer(self):
        self.timer = QTimer(self)
        #每次循环都都用一下show_pic这个函数
        self.timer.timeout.connect(self.show_pic)

    # 显示视频图像
    def show_pic(self):
        simage=cv2.imread('realsmallmap.png')
        name_list=[]
        ret, img = self.cap.read()
        if ret:
            # cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # # 视频流的长和宽
            # height, width = cur_frame.shape[:2]
            # pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            # pixmap = QPixmap.fromImage(pixmap)
            # # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            # ratio = max(width / self.label.width(),
            #             height / self.label.height())
            # pixmap.setDevicePixelRatio(ratio)
            # # 视频流置于label中间部分播放
            # self.label.setAlignment(Qt.AlignCenter)
            # #将视频流显示在label上
            # self.label.setPixmap(pixmap)
            showimg=img 
            with torch.no_grad():
                img = letterbox(img)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]
                
                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                        agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_segments(
                            img.shape[2:], det[:, :4], showimg.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            print(label)
                            c=int(cls)
                            annotator = Annotator(showimg, line_width=3, example=str(self.model.names))
                            annotator.box_label(
                                xyxy,  label)
                            # print(*xyxy)
                            a=(xyxy[0]+xyxy[1])/2
                            b=(xyxy[1]+xyxy[3])/2
                            imgPoints=np.array([[[a,b]]])
                            worldpt=camera2.cameraToWorld(mtx, rvec, tvec, imgPoints)
                            print('世界坐标：',worldpt)
            # self.out.write(showimg)
            show = cv2.resize(showimg, (640, 480))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())