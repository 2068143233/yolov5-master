# -*- coding=utf-8 -*-
# name: nan chen
# date: 2021/5/18 10:50

import cv2
import numpy as np
import glob

np.set_printoptions(suppress=True)

# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 18, 0.001)
# 棋盘格模板规格
w = 10  # 内角点个数，内角点是和其他格子连着的点
h = 6

# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点

images = glob.glob('cameracalibration\\count11.jpg')
i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    # 棋盘图像(8位灰度或彩色图像)  棋盘尺寸  存放角点的位置
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        # 角点精确检测
        # 输入图像 角点初始坐标 搜索窗口为2*winsize+1 死区 求角点的迭代终止条件
        i += 1
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.imshow('findCorners', img)
        cv2.imwrite('count11' + str(i) + '.jpg', img)
        cv2.waitKey(10)
cv2.destroyAllWindows()
# 标定、去畸变
# 输入：世界坐标系里的位置 像素坐标 图像的像素尺寸大小 3*3矩阵，相机内参数矩阵 畸变矩阵
# 输出：标定结果 相机的内参数矩阵 畸变系数 旋转矩阵 平移向量
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# mtx：内参数矩阵
# dist：畸变系数
# rvecs：旋转向量 （外参数）
# tvecs ：平移向量 （外参数）
print(("ret:"), ret)   #重投影误差
print(("mtx:\n"), mtx)  # 内参数矩阵
print(("dist:\n"), dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print(("rvecs:\n"), rvecs)  # 旋转向量  # 外参数
print(("tvecs:\n"), tvecs)  # 平移向量  # 外参数
# 去畸变
img2 = cv2.imread('count11.jpg')
h, w = img2.shape[:2]
# 我们已经得到了相机内参和畸变系数，在将图像去畸变之前，
# 我们还可以使用cv.getOptimalNewCameraMatrix()优化内参数和畸变系数，
# 通过设定自由自由比例因子alpha。当alpha设为0的时候，
# 将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
# 当alpha设为1的时候，将会返回一个包含额外黑色像素点的内参数和畸变系数，并返回一个ROI用于将其剪裁掉
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # 自由比例参数

dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
cv2.imshow("video",dst)
cv2.waitKey(0)
# 根据前面ROI区域裁剪图片
# x, y, w, h = roi
# dst = dst[y:y + h, x:x + w]
cv2.imwrite('count/calibresult.jpg', dst)

# 反投影误差
# 通过反投影误差，我们可以来评估结果的好坏。越接近0，说明结果越理想。
# 通过之前计算的内参数矩阵、畸变系数、旋转矩阵和平移向量，使用cv2.projectPoints()计算三维点到二维图像的投影，
# 然后计算反投影得到的点与图像上检测到的点的误差，最后计算一个对于所有标定图像的平均误差，这个值就是反投影误差。
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print(("total error: "), total_error / len(objpoints))

