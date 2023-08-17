import sys
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QLineEdit
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from radar2_ui import Ui_MainWindow
import cv2


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):  # 初始化函数
        super(MainWindow, self).__init__(parent)
        # UI界面
        self.setupUi(self)
        self.CAM_NUM = 0
        self.cap = cv2.VideoCapture()
        self.background()
        # 在label中播放视频
        self.init_timer()

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
        self.timer.timeout.connect(self.show_pic)

    # 显示视频图像
    def show_pic(self):
        ret, img = self.cap.read()
        if ret:
            cur_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 视频流的长和宽
            height, width = cur_frame.shape[:2]
            pixmap = QImage(cur_frame, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(pixmap)
            # 获取是视频流和label窗口的长宽比值的最大值，适应label窗口播放，不然显示不全
            ratio = max(width / self.label.width(),
                        height / self.label.height())
            pixmap.setDevicePixelRatio(ratio)
            # 视频流置于label中间部分播放
            self.label.setAlignment(Qt.AlignCenter)
            #将视频流显示在label上
            self.label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
